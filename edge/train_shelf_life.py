"""Train a grouped, transfer-learning RSL model from shelf-life image folders."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights

from edge.pipelines.shelf_life_model import (
    IMAGE_MEAN,
    IMAGE_SIZE,
    IMAGE_STD,
    MobileNetShelfLife,
)

LOGGER = logging.getLogger("harvestcast.train")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
LABEL_PATTERN = re.compile(r"^(apple|banana|carrot|tomato)\s*\(\s*(\d+)\s*-\s*(\d+)\s*\)$", re.I)
FRAME_PATTERN = re.compile(r"frame\s*(\d+)", re.I)


@dataclass(frozen=True)
class Sample:
    """One unique image, its interval label, and approximate source-frame group."""

    path: Path
    class_name: str
    product: str | None
    lower_days: float
    upper_days: float
    expired: bool
    group_id: str
    digest: str

    def metadata(self) -> dict[str, Any]:
        """Return JSON-safe sample metadata without leaking local paths."""
        return {
            "name": self.class_name,
            "product": self.product,
            "lower_days": self.lower_days,
            "upper_days": self.upper_days,
            "expired": self.expired,
        }


def parse_folder(folder_name: str) -> tuple[str, str | None, float, float, bool] | None:
    """Map dataset folder names to canonical classes and shelf-life intervals."""
    if folder_name.casefold() == "expired":
        return "Expired", None, 0.0, 0.0, True
    match = LABEL_PATTERN.fullmatch(folder_name.strip())
    if not match:
        return None
    product = match.group(1).title()
    lower, upper = float(match.group(2)), float(match.group(3))
    if lower >= upper:
        return None
    return f"{product}({int(lower)}-{int(upper)})", product, lower, upper, False


def source_group(path: Path, digest: str) -> str:
    """Bucket adjacent extracted frames so neighboring video frames stay together."""
    match = FRAME_PATTERN.search(path.stem)
    if match:
        return f"frame-bucket-{int(match.group(1)) // 100:05d}"
    return f"image-{digest[:16]}"


def load_samples(dataset_root: Path) -> tuple[list[Sample], dict[str, int]]:
    """Load image folders, removing exact duplicates and conflicting labels."""
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")
    by_digest: dict[str, list[Sample]] = defaultdict(list)
    ignored_folders = 0
    for folder in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        label = parse_folder(folder.name)
        if label is None:
            ignored_folders += 1
            LOGGER.warning("Ignoring unrecognized label folder: %s", folder.name)
            continue
        class_name, product, lower, upper, expired = label
        for image_path in sorted(path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS):
            raw = image_path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            by_digest[digest].append(
                Sample(
                    image_path,
                    class_name,
                    product,
                    lower,
                    upper,
                    expired,
                    source_group(image_path, digest),
                    digest,
                )
            )

    conflict_digests = {
        digest
        for digest, samples in by_digest.items()
        if len({sample.class_name.casefold() for sample in samples}) > 1
    }
    samples = [
        min(duplicates, key=lambda item: item.path.as_posix().casefold())
        for digest, duplicates in by_digest.items()
        if digest not in conflict_digests
    ]
    samples.sort(key=lambda item: (item.class_name.casefold(), item.group_id, item.path.as_posix()))
    stats = {
        "raw_files": sum(len(items) for items in by_digest.values()),
        "unique_unambiguous_images": len(samples),
        "conflicting_duplicate_files": sum(
            len(items) for digest, items in by_digest.items() if digest in conflict_digests
        ),
        "same_label_duplicate_files_removed": sum(
            max(0, len(items) - 1)
            for digest, items in by_digest.items()
            if digest not in conflict_digests
        ),
        "ignored_label_folders": ignored_folders,
    }
    return samples, stats


def grouped_split(samples: list[Sample], validation_fraction: float = 0.2) -> tuple[list[Sample], list[Sample], int]:
    """Choose a deterministic frame-group split with every class in both sets."""
    group_counts: dict[str, Counter[str]] = defaultdict(Counter)
    class_counts: Counter[str] = Counter()
    for sample in samples:
        group_counts[sample.group_id][sample.class_name] += 1
        class_counts[sample.class_name] += 1
    groups = sorted(group_counts)
    validation_group_count = max(1, round(len(groups) * validation_fraction))
    if len(groups) < 2:
        raise ValueError("At least two distinct frame groups are required for evaluation")

    best: tuple[float, int, set[str]] | None = None
    for seed in range(10_000):
        shuffled = groups.copy()
        random.Random(seed).shuffle(shuffled)
        validation_groups = set(shuffled[:validation_group_count])
        validation_counts: Counter[str] = Counter()
        for group in validation_groups:
            validation_counts.update(group_counts[group])
        if any(
            validation_counts[name] == 0 or validation_counts[name] >= total
            for name, total in class_counts.items()
        ):
            continue
        class_error = sum(
            abs(validation_counts[name] - total * validation_fraction) / max(total, 1)
            for name, total in class_counts.items()
        )
        size_error = abs(sum(validation_counts.values()) - len(samples) * validation_fraction) / len(samples)
        candidate = (class_error + size_error, seed, validation_groups)
        if best is None or candidate[0] < best[0]:
            best = candidate

    if best is None:
        raise ValueError("Could not make a grouped split containing every class in both sets")
    _, seed, validation_groups = best
    training = [sample for sample in samples if sample.group_id not in validation_groups]
    validation = [sample for sample in samples if sample.group_id in validation_groups]
    return training, validation, seed


class ShelfLifeDataset(Dataset[tuple[torch.Tensor, int, float, float, bool]]):
    """Decode labeled image samples and return their interval-survival targets."""

    def __init__(self, samples: list[Sample], class_to_index: dict[str, int], transform: transforms.Compose) -> None:
        self.samples = samples
        self.class_to_index = class_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, float, float, bool]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image_tensor = self.transform(image.convert("RGB"))
        return (
            image_tensor,
            self.class_to_index[sample.class_name],
            sample.lower_days,
            sample.upper_days,
            sample.expired,
        )


def evaluate(
    model: MobileNetShelfLife,
    loader: DataLoader,
    classes: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    """Compute class, range-containment, and midpoint errors on held-out groups."""
    model.eval()
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    midpoint_errors: list[float] = []
    interval_hits: list[bool] = []
    with torch.inference_mode():
        for images, labels, lower, upper, expired in loader:
            images = images.to(device)
            logits, scale, shape = model(images)
            predicted = logits.argmax(dim=1).cpu()
            expected = scale * torch.exp(torch.lgamma(1.0 + 1.0 / shape))
            lower_values = lower.float()
            upper_values = upper.float()
            for offset, predicted_index in enumerate(predicted.tolist()):
                predicted_class = classes[predicted_index]
                estimate = 0.0 if predicted_class["expired"] else float(expected[offset].item())
                lower_day = float(lower_values[offset].item())
                upper_day = float(upper_values[offset].item())
                if bool(expired[offset]):
                    interval_hits.append(predicted_class["expired"])
                    midpoint_errors.append(0.0 if predicted_class["expired"] else estimate)
                else:
                    interval_hits.append(
                        not predicted_class["expired"] and lower_day <= estimate <= upper_day
                    )
                    midpoint_errors.append(abs(estimate - (lower_day + upper_day) / 2.0))
                true_labels.append(int(labels[offset].item()))
                predicted_labels.append(predicted_index)

    per_class: dict[str, dict[str, int | float]] = {}
    for index, item in enumerate(classes):
        positions = [i for i, label in enumerate(true_labels) if label == index]
        if positions:
            hits = sum(true_labels[i] == predicted_labels[i] for i in positions)
            per_class[item["name"]] = {"count": len(positions), "accuracy": hits / len(positions)}
    balanced_accuracy = sum(float(item["accuracy"]) for item in per_class.values()) / max(len(per_class), 1)
    return {
        "accuracy": sum(actual == predicted for actual, predicted in zip(true_labels, predicted_labels)) / max(len(true_labels), 1),
        "balanced_accuracy": balanced_accuracy,
        "rsl_interval_hit_rate": sum(interval_hits) / max(len(interval_hits), 1),
        "rsl_midpoint_mae_days": sum(midpoint_errors) / max(len(midpoint_errors), 1),
        "per_class": per_class,
        "validation_samples": len(true_labels),
    }


def survival_interval_loss(
    scale: torch.Tensor,
    shape: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    expired: torch.Tensor,
) -> torch.Tensor:
    """Negative log likelihood for interval-censored non-expired observations."""
    valid = ~expired
    if not torch.any(valid):
        return scale.new_zeros(())
    scale = scale[valid].clamp_min(0.05)
    shape = shape[valid].clamp_min(0.05)
    lower = lower[valid]
    upper = upper[valid]
    survival_lower = torch.exp(-torch.pow(lower / scale, shape))
    survival_upper = torch.exp(-torch.pow(upper / scale, shape))
    interval_probability = (survival_lower - survival_upper).clamp_min(1e-7)
    return -torch.log(interval_probability).mean()


def main() -> None:
    """Train, evaluate, and save the best grouped-validation checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("selflife_dataset"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/shelf_life_mobilenet_v3.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--dry-run", action="store_true", help="Validate labels and split without training")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    random.seed(1616)
    np.random.seed(1616)
    torch.manual_seed(1616)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    samples, data_stats = load_samples(args.data_dir)
    training_samples, validation_samples, split_seed = grouped_split(samples)
    classes = [sample.metadata() for sample in sorted(samples, key=lambda item: item.class_name.casefold())]
    classes = list({item["name"]: item for item in classes}.values())
    class_to_index = {item["name"]: index for index, item in enumerate(classes)}
    LOGGER.info("Data stats: %s", data_stats)
    LOGGER.info("Class counts: %s", dict(Counter(sample.class_name for sample in samples)))
    LOGGER.info(
        "Grouped split: train=%s, validation=%s, frame_groups=%s, search_seed=%s",
        len(training_samples),
        len(validation_samples),
        len({sample.group_id for sample in samples}),
        split_seed,
    )
    if args.dry_run:
        return

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.72, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.1, hue=0.025),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )
    train_dataset = ShelfLifeDataset(training_samples, class_to_index, train_transform)
    validation_dataset = ShelfLifeDataset(validation_samples, class_to_index, validation_transform)
    train_counts = Counter(sample.class_name for sample in training_samples)
    weights = [1.0 / train_counts[sample.class_name] for sample in training_samples]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetShelfLife(len(classes), pretrained=True).to(device)
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    for block in list(model.encoder.features.children())[-2:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for layer in list(model.encoder.classifier.children())[:-1]:
        for parameter in layer.parameters():
            parameter.requires_grad = True

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=2e-4,
        weight_decay=1e-4,
    )
    class_criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_balanced_accuracy = -1.0
    best_metrics: dict[str, Any] = {}
    best_epoch = 0
    patience = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels, lower, upper, expired in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            lower = lower.to(device)
            upper = upper.to(device)
            expired = expired.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, scale, shape = model(images)
            class_loss = class_criterion(logits, labels)
            interval_loss = survival_interval_loss(scale, shape, lower, upper, expired)
            loss = class_loss + 0.35 * interval_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            running_loss += float(loss.item())

        metrics = evaluate(model, validation_loader, classes, device)
        LOGGER.info(
            "epoch %d/%d loss=%.4f accuracy=%.3f balanced_accuracy=%.3f interval_hit=%.3f midpoint_MAE=%.2fd",
            epoch,
            args.epochs,
            running_loss / max(len(train_loader), 1),
            metrics["accuracy"],
            metrics["balanced_accuracy"],
            metrics["rsl_interval_hit_rate"],
            metrics["rsl_midpoint_mae_days"],
        )
        if metrics["balanced_accuracy"] > best_balanced_accuracy:
            best_balanced_accuracy = float(metrics["balanced_accuracy"])
            best_metrics = metrics
            best_epoch = epoch
            patience = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "image_size": IMAGE_SIZE,
                    "mean": IMAGE_MEAN,
                    "std": IMAGE_STD,
                    "architecture": "mobilenet_v3_small_interval_weibull",
                    "validation_metrics": metrics,
                    "data_stats": data_stats,
                    "split": {
                        "frame_bucket_size": 100,
                        "search_seed": split_seed,
                        "training_samples": len(training_samples),
                        "validation_samples": len(validation_samples),
                    },
                },
                args.output,
            )
        else:
            patience += 1
            if patience >= 3:
                LOGGER.info("Early stopping after %s non-improving epochs", patience)
                break

    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps({"best_epoch": best_epoch, "validation": best_metrics, "data": data_stats}, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Saved best checkpoint to %s", args.output)
    LOGGER.info("Saved held-out metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
"""Rule-based receding-horizon HVAC setpoint policy."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HVACTargets:
    """Bounded actuator targets selected from predicted remaining life."""

    temperature_c: float
    humidity_pct: float
    fan_pct: float
    mode: str


def optimize_microclimate(expected_rsl_days: float) -> HVACTargets:
    """Select the requested storage profile for a batch's expected RSL."""
    if expected_rsl_days < 0:
        raise ValueError("Expected RSL cannot be negative")
    if expected_rsl_days < 4:
        return HVACTargets(3.5, 92.0, 90.0, "URGENT_LOCAL_DISPATCH")
    if expected_rsl_days < 8:
        return HVACTargets(6.0, 90.0, 60.0, "DOMESTIC_COLD_TRANSIT")
    return HVACTargets(8.5, 85.0, 45.0, "EXPORT_GRADE_LONG_HOLD")


def targets_as_dict(expected_rsl_days: float) -> dict[str, float | str]:
    """Return serializable HVAC target fields."""
    return asdict(optimize_microclimate(expected_rsl_days))

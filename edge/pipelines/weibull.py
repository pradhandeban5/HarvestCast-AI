"""Pure-Python Weibull shelf-life and FEFO calculations."""

from math import exp, gamma


def expected_rsl_days(lambda_days: float, k_shape: float) -> float:
    """Compute E[T] = lambda * Gamma(1 + 1/k) for Weibull parameters."""
    if lambda_days <= 0 or k_shape <= 0:
        raise ValueError("Weibull scale and shape must be positive")
    return lambda_days * gamma(1.0 + 1.0 / k_shape)


def survival_probability(t_days: float, lambda_days: float, k_shape: float) -> float:
    """Compute Weibull survival probability S(t) for nonnegative time."""
    if t_days < 0 or lambda_days <= 0 or k_shape <= 0:
        raise ValueError("Time must be nonnegative and Weibull parameters positive")
    return exp(-((t_days / lambda_days) ** k_shape))


def dispatch_action(rsl_days: float) -> str:
    """Map expected RSL to a FEFO logistics action."""
    if rsl_days < 4:
        return "URGENT_LOCAL_DISPATCH"
    if rsl_days < 8:
        return "DOMESTIC_COLD_TRANSIT"
    return "EXPORT_GRADE_LONG_HOLD"

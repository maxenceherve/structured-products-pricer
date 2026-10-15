"""When a barrier is observed: part of a product's definition, read by every pricer."""

from enum import StrEnum


class Monitoring(StrEnum):
    """Barrier observation."""

    EUROPEAN = "At maturity (European)"
    CONTINUOUS = "Continuously"
    DISCRETE = "On observation dates"


def check_observations(monitoring: Monitoring, observations: int | None) -> None:
    """Discrete monitoring needs a number of dates; the other modes take none."""
    if monitoring is Monitoring.DISCRETE:
        if observations is None or observations < 1:
            raise ValueError("discrete monitoring needs at least one observation date")
    elif observations is not None:
        raise ValueError("observations only apply to discrete monitoring")


def require_european(monitoring: Monitoring, product: str) -> None:
    """Refuse a monitoring that a product's replication does not support."""
    if monitoring is not Monitoring.EUROPEAN:
        raise NotImplementedError(
            f"{monitoring.value} barrier observation is not available for the {product} yet; "
            "only European (at maturity) barriers are priced for it."
        )


def observation_dates(maturity: float, observations: int) -> list[float]:
    """Equally spaced discrete observation dates ``maturity * i / observations``, i = 1..n."""
    if observations < 1:
        raise ValueError("discrete monitoring needs at least one observation date")
    return [maturity * i / observations for i in range(1, observations + 1)]

"""When a barrier is observed: part of a product's definition, read by every pricer."""

from enum import StrEnum


class Monitoring(StrEnum):
    """Barrier observation."""

    EUROPEAN = "At maturity (European)"
    CONTINUOUS = "Continuously"
    DISCRETE = "On observation dates"


def require_european(monitoring: Monitoring) -> None:
    """Refuse a monitoring that has no pricer yet, instead of pricing it as European."""
    if monitoring is not Monitoring.EUROPEAN:
        raise NotImplementedError(
            f"{monitoring.value} barrier observation comes with the barrier module; "
            "only European (at maturity) barriers are priced for now."
        )

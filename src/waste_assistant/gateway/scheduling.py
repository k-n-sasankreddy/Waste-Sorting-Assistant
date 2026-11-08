"""Pure policy selection over an ordered list of worker loads."""

from collections.abc import Sequence


def estimated_completion_ms(
    service_ms: float,
    waiting: int,
    elapsed_ms: float | None,
    residual_floor_fraction: float,
) -> float:
    residual = (
        0.0
        if elapsed_ms is None
        else max(service_ms - elapsed_ms, residual_floor_fraction * service_ms)
    )
    return residual + (waiting + 1) * service_ms


def choose_worker(
    policy: str, loads: Sequence[float | None], cursor: int
) -> int | None:
    """None marks an ineligible worker. Ties rotate from the cursor."""
    eligible = [(cursor + offset) % len(loads) for offset in range(len(loads))]
    eligible = [index for index in eligible if loads[index] is not None]
    if not eligible:
        return None
    if policy == "round_robin":
        return eligible[0]
    if policy in ("least_outstanding", "adaptive"):
        return min(eligible, key=loads.__getitem__)
    raise ValueError(f"Unsupported scheduling policy: {policy}")

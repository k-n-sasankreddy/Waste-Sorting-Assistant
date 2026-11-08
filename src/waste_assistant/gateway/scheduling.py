"""Pure policy selection over an ordered list of worker loads."""

from collections.abc import Sequence


def choose_worker(policy: str, loads: Sequence[int | None], cursor: int) -> int | None:
    """None marks an ineligible worker. Both policies rotate ties from the cursor."""
    eligible = [(cursor + offset) % len(loads) for offset in range(len(loads))]
    eligible = [index for index in eligible if loads[index] is not None]
    if not eligible:
        return None
    if policy == "round_robin":
        return eligible[0]
    if policy == "least_outstanding":
        return min(eligible, key=loads.__getitem__)
    raise ValueError(f"Unsupported Phase 5 scheduling policy: {policy}")

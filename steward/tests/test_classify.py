"""What the robot calls a region, and the one view that used to hide a whole table.

The twin's only camera looks straight down at its table, so the tabletop arrives as a one-centimetre
slab at table height -- its legs are underneath it and its sides are out of view. Judged as a box,
that region fits no prototype, the table was "unknown", and the Ego State showed the bowls resting
on nothing.
"""

import numpy as np

from steward.ego.classify import classify

WOOD = np.array([0.62, 0.50, 0.38])


def _top(probs: dict[str, float]) -> str:
    return max(probs, key=probs.get)


def test_a_tabletop_seen_from_above_is_a_table() -> None:
    probs = classify(np.array([0.99, 0.79, 0.01]), 0.745, WOOD)
    assert _top(probs) == "table", probs


def test_the_same_table_seen_from_the_side_is_still_a_table() -> None:
    probs = classify(np.array([0.99, 0.79, 0.75]), 0.0, WOOD)
    assert _top(probs) == "table", probs


def test_a_plate_on_that_table_is_not_the_table() -> None:
    probs = classify(np.array([0.24, 0.24, 0.02]), 0.75, np.array([0.9, 0.9, 0.9]))
    assert _top(probs) == "plate", probs


def test_a_flat_sheet_on_the_floor_is_not_furniture() -> None:
    probs = classify(np.array([0.99, 0.79, 0.01]), 0.0, WOOD)
    assert _top(probs) != "table", probs


def test_a_counter_top_at_counter_height_is_a_counter() -> None:
    probs = classify(np.array([1.85, 0.62, 0.01]), 0.89, WOOD)
    assert _top(probs) == "counter", probs


def test_stemware_sized_box_with_table_bleed_colour_is_still_a_wine_glass() -> None:
    """Clear glass often averages to table beige with few depth holes; size must still win."""
    beige = np.array([0.76, 0.68, 0.55])
    probs = classify(np.array([0.079, 0.073, 0.188]), 0.75, beige, invalid_frac=0.0)
    assert _top(probs) == "wine_glass", probs


def test_stemware_with_partial_depth_holes_is_a_wine_glass() -> None:
    beige = np.array([0.76, 0.68, 0.55])
    probs = classify(np.array([0.08, 0.08, 0.19]), 0.75, beige, invalid_frac=0.1)
    assert _top(probs) == "wine_glass", probs


def test_a_wide_beige_box_is_not_rescued_as_a_wine_glass() -> None:
    """Geometry-softened clear must not turn an unrelated beige region into stemware."""
    beige = np.array([0.76, 0.68, 0.55])
    probs = classify(np.array([0.22, 0.22, 0.07]), 0.75, beige, invalid_frac=0.0)
    assert _top(probs) != "wine_glass", probs
    assert _top(probs) == "bowl", probs

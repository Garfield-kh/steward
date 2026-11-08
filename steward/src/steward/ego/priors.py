"""Category priors the robot brings to perception: affordances, how likely the object is fixed,
and fragility. They are robot knowledge, not World State facts.

``receptacle`` marks things whose floor holds other objects inside them (drawers, nesting bowls);
``pullable`` marks drawers that open along their front axis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CATEGORY_PRIORS: dict[str, dict] = {
    "wall": {"role": "environment", "root": {"fixed": 1.0}, "affordances": [], "fragility": 0.0},
    "counter": {"role": "object", "root": {"fixed": 0.9, "free": 0.1}, "affordances": ["support-surface"], "fragility": 0.0},
    "table": {"role": "object", "root": {"free": 0.7, "fixed": 0.3}, "affordances": ["support-surface"], "fragility": 0.0},
    "sideboard": {"role": "object", "root": {"fixed": 0.9, "free": 0.1}, "affordances": ["support-surface", "container"],
                  "fragility": 0.0},
    "cabinet": {"role": "object", "root": {"fixed": 0.9, "free": 0.1}, "affordances": ["support-surface", "container"],
                "fragility": 0.0},
    "drawer": {"role": "object", "root": {"fixed": 1.0}, "affordances": ["container", "receptacle", "pullable"],
               "fragility": 0.0},
    "cup": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable", "container"], "fragility": 0.8},
    "bowl": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable", "container", "receptacle", "stackable"],
             "fragility": 0.6},
    "plate": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable", "support-surface"], "fragility": 0.6},
    "wine_glass": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable", "container"], "fragility": 0.95},
    "mug": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable", "container"], "fragility": 0.4},
    "teddy_bear": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable"], "fragility": 0.0},
    "scissors": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable"], "fragility": 0.0},
    "toy_airplane": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable"], "fragility": 0.1},
    "pistol": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable"], "fragility": 0.0},
    "pill_bottle": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable", "container"], "fragility": 0.2},
    "perfume_bottle": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable", "container"], "fragility": 0.8},
    "chair": {"role": "object", "root": {"free": 0.95, "fixed": 0.05}, "affordances": ["graspable", "floor-standing"],
              "fragility": 0.1},
    "plant": {"role": "object", "root": {"free": 1.0}, "affordances": ["graspable", "floor-standing"], "fragility": 0.3},
}

UNKNOWN_PRIOR = {"role": "object", "root": {"free": 0.5, "fixed": 0.5}, "affordances": [], "fragility": 0.5}

# Words that name a listed category in other terms ("desk" for a table).
SYNONYMS: dict[str, str] = {"desk": "table", "bench": "table", "shelf": "cabinet", "cupboard": "cabinet",
                            "dresser": "cabinet", "stool": "chair", "glass": "wine_glass"}


@dataclass(frozen=True)
class CommonSense:
    """Sizes by which an object of an unlisted category gets its affordances: a thing at least
    ``surface_min_m`` across, standing on the floor (base below ``floor_max_m``) with its top at
    least ``surface_top_min_m`` high, is furniture with a ``support-surface``; a thing less than
    ``graspable_max_m`` across is ``graspable``."""

    surface_min_m: float = 0.4
    surface_top_min_m: float = 0.3
    floor_max_m: float = 0.1
    graspable_max_m: float = 0.35


COMMON_SENSE = CommonSense()

COMPATIBLE: dict[str, set[str]] = {"cup": {"cup", "bowl"}, "bowl": {"bowl", "cup"}}


def prior(category: str) -> dict:
    return CATEGORY_PRIORS.get(category, UNKNOWN_PRIOR)


def default_affordances(category: str, footprint: float, bottom: float, top: float,
                        sense: CommonSense = COMMON_SENSE) -> list[str]:
    """Affordances an object has by common sense: its category prior's, else those of a listed
    category its name contains ("dinner table", "photo_frame"), else by its size (see
    :class:`CommonSense`); none for an object that is neither furniture nor small.

    :param category: the category perception or the robot's agent gave it.
    :param footprint: its larger horizontal extent in metres.
    :param bottom: the height of its base in metres.
    :param top: the height of its top in metres.
    :param sense: the sizes the size rule uses.
    :returns: the affordance names.
    """
    if category in CATEGORY_PRIORS:
        return list(CATEGORY_PRIORS[category]["affordances"])
    for word in reversed(re.split(r"[\s_\-]+", category.lower())):
        named = SYNONYMS.get(word, word)
        if named in CATEGORY_PRIORS:
            return list(CATEGORY_PRIORS[named]["affordances"])
    if footprint >= sense.surface_min_m and bottom < sense.floor_max_m and top >= sense.surface_top_min_m:
        return ["support-surface"]
    if footprint < sense.graspable_max_m:
        return ["graspable"]
    return []


def compatible(a: str, b: str) -> bool:
    return a == b or b in COMPATIBLE.get(a, set())

"""Affordances the robot gives objects by common sense when perception names them in its own terms."""

from __future__ import annotations

from steward.ego.priors import default_affordances


def test_a_listed_category_keeps_its_prior_and_a_name_containing_one_takes_its_affordances():
    assert default_affordances("bowl", 0.15, 0.75, 0.82) == ["graspable", "container", "receptacle", "stackable"]
    assert default_affordances("dinner table", 0.9, 0.0, 0.75) == ["support-surface"]
    assert default_affordances("writing desk", 0.9, 0.0, 0.75) == ["support-surface"], "a synonym names the table"


def test_an_unlisted_name_is_furniture_with_a_surface_when_large_and_graspable_when_small():
    assert default_affordances("picture", 0.24, 0.75, 0.82) == ["graspable"], "a puzzle box named by its print"
    assert default_affordances("puzzle_piece", 0.06, 0.75, 0.76) == ["graspable"]
    assert default_affordances("sofa", 2.0, 0.0, 0.45) == ["support-surface"]
    assert default_affordances("rug", 1.5, 0.0, 0.01) == [], "too low to put things on, too large to take"

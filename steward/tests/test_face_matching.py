"""Face correlation against a direct small-image oracle, including missing observations."""

import math

import numpy as np
import pytest

from steward.ego.perception import match_face


def direct_matches(face: np.ndarray, reference: np.ndarray, top: int = 3) -> list[dict]:
    """Independent sliding-window definition; intentionally small inputs keep this oracle cheap."""
    candidates = []
    for turn in range(4):
        patch = np.rot90(face, turn)
        h, w = patch.shape[:2]
        for row in range(reference.shape[0] - h + 1):
            for col in range(reference.shape[1] - w + 1):
                window = reference[row:row + h, col:col + w]
                finite = np.isfinite(patch) & np.isfinite(window)
                if finite.sum() < patch.size / 2:
                    continue
                numerator = norm_a = norm_b = 0.0
                for channel in range(3):
                    mask = finite[..., channel]
                    a, b = patch[..., channel][mask], window[..., channel][mask]
                    if not len(a):
                        continue
                    a, b = a - a.mean(), b - b.mean()
                    numerator += float(a @ b)
                    norm_a += float(a @ a)
                    norm_b += float(b @ b)
                norm = math.sqrt(norm_a * norm_b)
                candidates.append((numerator / norm if norm > 1e-9 else 0.0,
                                   row + h / 2, col + w / 2, turn, max(h, w)))
    candidates.sort(key=lambda item: -item[0])
    chosen = []
    for candidate in candidates:
        if all(math.hypot(candidate[1] - other[1], candidate[2] - other[2]) >= other[4] / 2 for other in chosen):
            chosen.append(candidate)
        if len(chosen) == top:
            break
    return [{"row": row, "col": col, "turns": turn, "score": round(score, 3)}
            for score, row, col, turn, _ in chosen]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("missing", [False, True])
def test_rotated_rectangular_candidates_match_direct_masked_correlations(seed, missing):
    rng = np.random.default_rng(seed)
    reference = rng.uniform(10, 250, (11, 13, 3))
    face = np.rot90(reference[3:7, 5:11], -seed % 4).copy()
    face += np.array([3, -7, 12])
    if missing:
        face[rng.random(face.shape) < 0.15] = np.nan
        reference[rng.random(reference.shape) < 0.15] = np.nan
    assert match_face(face, reference)["matches"] == direct_matches(face, reference)


def test_unobserved_face_has_no_candidates():
    assert match_face(np.full((4, 4, 3), np.nan), np.ones((8, 8, 3)))["matches"] == []


def test_uniform_channels_do_not_create_fft_roundoff_matches():
    reference = np.full((12, 12, 3), [200, 100, 50], dtype=float)
    reference[8:, 8:] = [10, 20, 30]
    face = np.full((4, 4, 3), [200, 100, 50], dtype=float)
    assert all(candidate["score"] == 0 for candidate in match_face(face, reference)["matches"])


def test_exact_half_coverage_is_accepted_and_less_than_half_is_rejected():
    reference = np.arange(8 * 8 * 3, dtype=float).reshape(8, 8, 3)
    face = reference[:4, :4].copy()
    face[:2] = np.nan
    assert match_face(face, reference)["matches"]
    face[2, 0] = np.nan
    assert match_face(face, reference)["matches"] == []

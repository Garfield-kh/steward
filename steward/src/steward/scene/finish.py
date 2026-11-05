"""Procedural finishes for the textured render style.

Flat rgba is one render style. These images are the other: the same geometry, with grain, grout
or plaster so a model trained on photographs has edges and shading to use. Glass stays flat.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np

from steward.assets.generators import (
    BIRCH, BIRCH_LEG, CABINET_BLUE, CERAMIC, COUNTER_TOP, COUNTER_WHITE, FLOOR, LIGHT_WOOD, STEEL, WALL,
    WALNUT, WALNUT_DARK, WOOD,
)

_KIND = (
    ("walnut", WALNUT), ("walnut", WALNUT_DARK),
    ("birch", BIRCH), ("birch", BIRCH_LEG), ("birch", LIGHT_WOOD), ("birch", WOOD),
    # The carcass stays a bright panel. The same tile finish as the grey top pulled it toward the
    # wall, and a frontal view was then named as wall.
    ("cabinet", COUNTER_WHITE),
    ("tile", COUNTER_TOP), ("tile", CERAMIC), ("tile", STEEL),
    ("paint", CABINET_BLUE), ("plaster", WALL), ("plank", FLOOR),
)


def finish_kind(rgba: list[float]) -> str | None:
    """Which procedural finish matches an asset colour, or ``None`` for glass and unlisted colours.

    :param rgba: red, green, blue and optional alpha, each in 0..1.
    """
    if len(rgba) > 3 and float(rgba[3]) < 0.9:
        return None
    rgb = np.asarray(rgba[:3], float)
    for kind, ref in _KIND:
        if float(np.max(np.abs(rgb - np.asarray(ref[:3])))) <= 0.04:
            return kind
    return None


def finish_png(kind: str, rgba: list[float], size: int = 128) -> bytes:
    """A square RGB PNG of ``kind`` tinted by ``rgba``.

    :param kind: one of the names :func:`finish_kind` returns.
    :param size: width and height in pixels.
    """
    base = np.asarray(rgba[:3], float)
    y, x = np.mgrid[0:size, 0:size]
    rng = np.random.default_rng(zlib.crc32(kind.encode()) & 0xFFFFFFFF)
    noise = rng.random((size, size))
    if kind == "walnut":
        grain = 0.96 + 0.08 * np.sin(y * 0.85 + 1.2 * np.sin(x * 0.32))
        shade = grain * (0.98 + 0.04 * noise)
    elif kind == "birch":
        grain = 1.0 + 0.05 * np.sin(y * 0.7 + np.sin(x * 0.4))
        shade = grain * (0.99 + 0.03 * noise)
    elif kind == "cabinet":
        # A darker stile around a bright panel: the front reads as a cupboard, brighter than plaster.
        margin = size * 0.14
        inset = (x > margin) & (x < size - margin) & (y > margin) & (y < size - margin)
        shade = np.where(inset, 1.06 + 0.02 * noise, 0.78)
    elif kind == "tile":
        grout = ((x % 32) < 1) | ((y % 32) < 1)
        shade = np.where(grout, 0.72, 1.04 + 0.03 * noise)
    elif kind == "paint":
        seam = np.abs(y - size / 2) < 1.5
        shade = np.where(seam, 0.82, 1.0 + 0.04 * noise)
    elif kind == "plaster":
        # Darker than the counter panel, so a frontal white carcass does not continue the wall.
        shade = 0.92 + 0.04 * noise
    elif kind == "plank":
        seam = (y % 21) < 1
        grain = 0.98 + 0.06 * np.sin(x * 0.45)
        shade = np.where(seam, 0.90, grain * (1.0 + 0.03 * noise))
    else:
        raise ValueError(f"unknown finish {kind!r}")
    rgb = np.clip(base * shade[..., None] * 255.0, 0, 255).astype(np.uint8)
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(size))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))

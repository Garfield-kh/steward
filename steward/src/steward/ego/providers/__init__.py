"""Optional perception frontends for Phase 1.

- Geometric depth masks (no model): the ``phase1-sam`` segmenter fills regions SAM 3.1 left
  unlabelled with them (:meth:`steward.ego.segmenter.Segmenter.fill`)
- NVS / SLARM / NovelAnything clients for gated depth / novel-view assist

Requires ``requests`` when calling remote APIs (``pip install requests`` or
``pip install 'steward[providers]'``).
"""

from steward.ego.providers.geometric_masks import masks_from_depth, replace_frame_mask

__all__ = [
    "masks_from_depth",
    "replace_frame_mask",
]

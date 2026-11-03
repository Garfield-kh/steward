"""Optional perception frontends for Phase 1.

- Geometric depth masks (offline / no API)
- SAM3 / SAM3.1 PCS masks (``https://linxlab-research.com/sam3``)
- geo∪SAM3 merge (best local Phase-1 recall)
- NVS / SLARM / NovelAnything clients for gated depth / novel-view assist

Requires ``requests`` when calling remote APIs (``pip install requests`` or
``pip install 'steward[providers]'``).
"""

from steward.ego.providers.fused_rgbd import env_mask_source, prepare_rgbd_frame
from steward.ego.providers.geometric_masks import masks_from_depth, replace_frame_mask

__all__ = [
    "masks_from_depth",
    "replace_frame_mask",
    "prepare_rgbd_frame",
    "env_mask_source",
]

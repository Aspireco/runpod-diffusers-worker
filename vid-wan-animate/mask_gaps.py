"""Fill the frames where SAM2 lost the person (2026-10-19).

Wan2.2's preprocessing tracks the person through the driving video with SAM2, then takes each frame's mask
bounding box with np.nonzero(mask).min() (wan/modules/animate/preprocess/utils.py get_mask_boxes). A frame where
the track is lost -- a muzzle flash, smoke, a rifle across the body (Scarface's shootout, John's swap) -- has an
empty mask, and the whole job dies with "zero-size array to reduction operation minimum". Here an empty frame
borrows the mask of its nearest frame that has one (the earlier one on a tie), which is what the person most
likely still covers; a video with no person at all fails with a sentence that says so.
"""
from __future__ import annotations

import numpy as np


def fill_empty_masks(masks: list, log=print) -> list:
    if not masks:
        return masks
    found = [i for i, m in enumerate(masks) if np.asarray(m).any()]
    if not found:
        raise ValueError("no person was found in any frame of the driving video; use a shot where the person "
                         "is clearly visible")
    out = list(masks)
    filled = 0
    for i, m in enumerate(masks):
        if np.asarray(m).any():
            continue
        nearest = min(found, key=lambda j: (abs(j - i), j > i))
        out[i] = masks[nearest]
        filled += 1
    if filled:
        log(f"[prep] the person's mask was lost on {filled} of {len(masks)} frames; filled from the nearest frame")
    return out

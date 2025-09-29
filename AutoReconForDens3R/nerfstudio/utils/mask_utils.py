
from typing import Dict, Any

import torch
from torchtyping import TensorType


def alpha_composite(rgb0: torch.Tensor, rgb1: torch.Tensor,
                    alpha0: torch.Tensor, alpha1: torch.Tensor):
    """Alpha composite two RGB images."""
    alpha = alpha0 + alpha1 * (1 - alpha0)
    rgb = (rgb0 * alpha0 + rgb1 * alpha1 * (1 - alpha0)) / alpha
    rgba = torch.cat([rgb, alpha], -1)
    return rgba


def overlay_mask_on_image(mask_rgb: TensorType["h", "w", 3],
                          img_rgb: TensorType["h", "w", 3],
                          mask_prob: TensorType["h", "w", 1]):
    mask_alpha = mask_prob
    img_alpha = torch.ones_like(mask_alpha)
    return alpha_composite(mask_rgb, img_rgb, mask_alpha, img_alpha)


def compute_mask_indices(collated_batch: Dict[str, Any]) -> Dict[str, Any]:
    """precompute zero and non-zero mask indices for fast sampling from masked & non-masked regions"""
    if "mask" not in collated_batch:
        return collated_batch
    if not isinstance(collated_batch["mask"], torch.Tensor):
        raise NotImplementedError  # e.g., handling ``BasicImages``
    
    mask = collated_batch["mask"][..., 0]  # (n_frames, h, w)
    assert mask.ndim == 3
    nonzero_indices = torch.nonzero(mask, as_tuple=False).to(torch.int16)
    zero_indices = torch.nonzero(~mask, as_tuple=False).to(torch.int16)
    collated_batch.update({
        "mask_nonzero_indices": nonzero_indices,
        "mask_zero_indices": zero_indices
    })
    return collated_batch

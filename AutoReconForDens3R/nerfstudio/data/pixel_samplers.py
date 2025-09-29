# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code for sampling pixels.
"""

import random
from typing import Dict, Optional

import numpy as np
import torch
from einops import rearrange
from torchtyping import TensorType
from typing_extensions import Literal

from nerfstudio.utils.images import BasicImages


def _unbatched_interpolate_2d(
    inputs: TensorType["b", "c", "h", "w"],
    indices: TensorType["n_inds", "n_dims"],  # <b, h, w>
    mode: Literal["nearest", "linear"] = "linear",
    padding_mode: Literal["zeros", "border", "reflection"] = "border",
    align_corners: Optional[bool] = None,
) -> TensorType["n_inds", "c"]:
    """
    TODO: Implement CUDA & CPP versions.
    """
    if mode != "linear":
        raise NotImplementedError
    if padding_mode != "border":
        raise NotImplementedError

    if align_corners is None and mode == "linear":
        align_corners = False

    # mode = "linear" | padding_mode = "border"
    B, C, H, W = inputs.shape
    N_INDS, N_DIMS = indices.shape

    # compute indices for interpolation: (n_rays, 4, 3) 3: <n, h, w>, 4: <tl, tr, bl, br>
    ib, iy, ix = indices[..., 0].long(), indices[..., 1], indices[..., 2]  # (n_inds, )

    if align_corners:
        # -1 and +1 get sent to the centers of the corner pixels
        # -1 --> 0
        # +1 --> (size - 1)
        # scale_factor = (size - 1) / 2
        ix = ((ix + 1) / 2) * (W - 1)
        iy = ((iy + 1) / 2) * (H - 1)
    else:
        # -1 and +1 get sent to the image edges
        # -1 --> -0.5
        # +1 --> (size - 1) + 0.5 == size - 0.5
        # scale_factor = size / 2
        ix = ((ix + 1) / 2) * W - 0.5
        iy = ((iy + 1) / 2) * H - 0.5

    with torch.no_grad():
        ix_nw, iy_nw = torch.floor(ix), torch.floor(iy)  # (n_inds, )
        ix_ne, iy_ne = ix_nw + 1, iy_nw
        ix_sw, iy_sw = ix_nw, iy_nw + 1
        ix_se, iy_se = ix_nw + 1, iy_nw + 1

    # compute interpolation weights: (n_rays, 4)
    nw = (ix_se - ix) * (iy_se - iy)  # (n_inds, )
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    # handle padding (padding_mode == border)
    with torch.no_grad():  # TODO: optimize
        torch.clamp(ix_nw, 0, W - 1, out=ix_nw)
        torch.clamp(iy_nw, 0, H - 1, out=iy_nw)

        torch.clamp(ix_ne, 0, W - 1, out=ix_ne)
        torch.clamp(iy_ne, 0, H - 1, out=iy_ne)

        torch.clamp(ix_sw, 0, W - 1, out=ix_sw)
        torch.clamp(iy_sw, 0, H - 1, out=iy_sw)

        torch.clamp(ix_se, 0, W - 1, out=ix_se)
        torch.clamp(iy_se, 0, H - 1, out=iy_se)

    # indexing inputs: (n_rays, 4, c)
    inputs = rearrange(inputs, "b c h w -> (b h w) c")
    nw_val = torch.gather(
        inputs, 0, (ib * H * W + iy_nw * W + ix_nw).long().view(N_INDS, 1).repeat(1, C)
    )  # (n_inds, c)
    ne_val = torch.gather(inputs, 0, (ib * H * W + iy_ne * W + ix_ne).long().view(N_INDS, 1).repeat(1, C))
    sw_val = torch.gather(inputs, 0, (ib * H * W + iy_sw * W + ix_sw).long().view(N_INDS, 1).repeat(1, C))
    se_val = torch.gather(inputs, 0, (ib * H * W + iy_se * W + ix_se).long().view(N_INDS, 1).repeat(1, C))

    # perform interpolation: (n_rays, c)
    out_val = nw_val * nw[:, None] + ne_val * ne[:, None] + sw_val * sw[:, None] + se_val * se[:, None]
    return out_val


def unbatched_interpolate(
    inputs: torch.Tensor,  # TensorType["b", "c", "spatial_dims": ...],
    indices: TensorType["n_inds", "n_dims"],  # <b, h, w> / <b, d, h, w>
    mode: Literal["nearest", "linear"] = "linear",
    padding_mode: Literal["zeros", "border", "reflection"] = "border",
    align_corners: Optional[bool] = None,
) -> TensorType["n_inds", "c"]:
    """Perform unbatched interpolation of inputs at indices.
    This is useful when interpolate multi-view images with randomly sampled rays (images, spatial coordinates)

    n_idims = len(spatial_dims) + 1 (1 is the b dim)

    NOTE: indices dims order is different from grid_sample (<h, w> instead of <w, h>)
    """
    kwargs = {"mode": mode, "padding_mode": padding_mode, "align_corners": align_corners}
    if inputs.ndim == 4:
        assert indices.shape[-1] == 3
        return _unbatched_interpolate_2d(inputs, indices, **kwargs)
    else:
        raise NotImplementedError


def unbatched_interpolate_input(
    inputs: TensorType["n", "c", "h", "w"],
    indices: TensorType["n_rays", "n_dims"],  # <img_idx, height_idx, width_idx>
    image_height: int,
    image_width: int,
    mode: Literal["linear", "nearest"] = "nearest",
) -> TensorType["n_rays", "c"]:
    """Interpolate input with ray sample indices."""
    if mode != "linear":
        raise NotImplementedError("Only linear interpolation is supported for now.")

    # normalize indices (align_corners=True)
    indices = indices.to(torch.float32)
    indices[:, 1:] = indices[:, 1:] / torch.tensor([image_height - 1, image_width - 1], device=indices.device)
    indices[:, 1:] = indices[:, 1:] * 2 - 1  # in range [-1, 1]

    # interpolate
    interped_inputs = unbatched_interpolate(
        inputs, indices.to(inputs.device), mode=mode, align_corners=True
    )  # (n_rays, c)
    return interped_inputs


def collate_image_dataset_batch(
    batch: Dict[str, torch.Tensor],
    num_rays_per_batch: int,
    keep_full_image: bool = False,
    step: Optional[int] = None,
    mask_sample_start: Optional[int] = None,
    mask_sample_ratio: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """
    Operates on a batch of images and samples pixels to use for generating rays.
    Returns a collated batch which is input to the Graph.
    It will sample only within the valid 'mask' if it's specified.

    Args:
        batch: batch of images to sample from
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
        mask_sample_ratio: sample this ratio of pixels within masks, if masks are available.
            if None, sample all pixels within masks.
    TODO:
        - option-1: implement new PixelSampler subclasses for new cases, instead of changing the original one
        - option-2: extend the current PixelSampler to support more cases, and apply to cases through registration
    """
    device = batch["image"].device  # NOTE: batch["image"] is on cpu, other tensors are on gpu
    num_images, image_height, image_width, _ = batch["image"].shape

    # only sample within the mask, if the mask is in the batch
    sample_in_mask = mask_sample_ratio is not None and (mask_sample_start is None or mask_sample_start <= step)
    if "mask" in batch and sample_in_mask:
        if mask_sample_ratio is None:
            n_rays_in_mask, n_rays_out_mask = num_rays_per_batch, 0
        else:
            n_rays_in_mask = int(np.floor(num_rays_per_batch * mask_sample_ratio))
            n_rays_out_mask = num_rays_per_batch - n_rays_in_mask

        _all_indices = []
        if n_rays_in_mask > 0:
            nonzero_indices = batch["mask_nonzero_indices"]
            chosen_indices = torch.randint(0, len(nonzero_indices), (n_rays_in_mask,), device=nonzero_indices.device)
            # chosen_indices = random.sample(range(len(nonzero_indices)), k=n_rays_in_mask)
            _all_indices.append(nonzero_indices[chosen_indices])
        if n_rays_out_mask > 0:
            zero_indices = batch["mask_zero_indices"]
            chosen_indices = torch.randint(0, len(zero_indices), (n_rays_out_mask,), device=zero_indices.device)
            # chosen_indices = random.sample(range(len(zero_indices)), k=n_rays_out_mask)
            _all_indices.append(zero_indices[chosen_indices])
        indices = torch.cat(_all_indices, dim=0).to(device=device, dtype=torch.long)  # (n_ray, 3), <idx, height, width>
    else:
        indices = torch.floor(
            torch.rand((num_rays_per_batch, 3), device=device)
            * torch.tensor([num_images, image_height, image_width], device=device)
        ).long()  # (n_ray, 3), <idx, height, width>

    c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))

    collated_batch = {
        key: value[c, y, x]
        for key, value in batch.items()
        # TODO: support configuration
        # TODO: pre-built a set!
        if key
        not in (
            "image_idx",
            "src_imgs",
            "src_idxs",
            "sparse_sfm_points",
            "feature",
            "gt_features_pca",
            "feature_pca",
            "feature_pca_min",
            "feature_pca_max",
            "mask_nonzero_indices",
            "mask_zero_indices",
            "ptcd_data",
        )
        and value is not None
    }

    assert collated_batch["image"].shape == (num_rays_per_batch, 3), collated_batch["image"].shape

    if "sparse_sfm_points" in batch:
        collated_batch["sparse_sfm_points"] = batch["sparse_sfm_points"].images[c[0]]

    # handle data requiring interpolation instead of discrete indexing (e.g., feature maps with different res as images)
    if "feature" in batch:
        collated_batch["feature"] = unbatched_interpolate_input(
            batch["feature"], indices, image_height, image_width, mode="linear"
        )
    # if "semantics" in batch:
    #     collated_batch["semantics"] = unbatched_interpolate_input(batch["semantics"], indices,
    #                                                               image_height, image_width, mode="nearest")

    # Needed to correct the random indices to their actual camera idx locations.

    indices[:, 0] = batch["image_idx"][c]
    collated_batch["indices"] = indices  # with the abs camera indices

    if keep_full_image:
        collated_batch["full_image"] = batch["image"]

    return collated_batch


def collate_image_dataset_batch_list(batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
    """
    Does the same as collate_image_dataset_batch, except it will operate over a list of images / masks inside
    a list.

    We will use this with the intent of DEPRECIATING it as soon as we find a viable alternative.
    The intention will be to replace this with a more efficient implementation that doesn't require a for loop, but
    since pytorch's ragged tensors are still in beta (this would allow for some vectorization), this will do

    Args:
        batch: batch of images to sample from
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """

    device = batch["image"][0].device
    num_images = len(batch["image"])

    # only sample within the mask, if the mask is in the batch
    all_indices = []
    all_images = []
    all_fg_masks = []

    # sample indices
    if "mask" in batch:
        num_rays_in_batch = num_rays_per_batch // num_images
        for i in range(num_images):
            if i == num_images - 1:
                num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch
            # nonzero_indices = torch.nonzero(batch["mask"][i][..., 0], as_tuple=False)
            nonzero_indices = batch["mask"][i]

            chosen_indices = random.sample(range(len(nonzero_indices)), k=num_rays_in_batch)
            indices = nonzero_indices[chosen_indices]
            indices = torch.cat([torch.full((num_rays_in_batch, 1), i, device=device), indices], dim=-1)
            all_indices.append(indices)
            all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])

            if "fg_mask" in batch:
                all_fg_masks.append(batch["fg_mask"][i][indices[:, 1], indices[:, 2]])
    else:
        num_rays_in_batch = num_rays_per_batch // num_images
        for i in range(num_images):
            image_height, image_width, _ = batch["image"][i].shape
            if i == num_images - 1:
                num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch
            indices = torch.floor(
                torch.rand((num_rays_in_batch, 3), device=device)
                * torch.tensor([1, image_height, image_width], device=device)
            ).long()
            indices[:, 0] = i
            all_indices.append(indices)
            all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])

            if "fg_mask" in batch:
                all_fg_masks.append(batch["fg_mask"][i][indices[:, 1], indices[:, 2]])

    indices = torch.cat(all_indices, dim=0)  # (n, n_rays_per_img, 3), <idx, height, width>

    c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))

    # TODO: handle data requiring interpolation instead of simple indexing (e.g., feature maps with different res as images)
    __import__("ipdb").set_trace()

    collated_batch = {
        key: value[c, y, x]
        for key, value in batch.items()
        if key != "image_idx"
        and key != "image"
        and key != "mask"
        and key != "fg_mask"
        and key != "sparse_pts"
        and value is not None
    }

    collated_batch["image"] = torch.cat(all_images, dim=0)
    if len(all_fg_masks) > 0:
        collated_batch["fg_mask"] = torch.cat(all_fg_masks, dim=0)

    if "sparse_pts" in batch:
        rand_idx = random.randint(0, num_images - 1)
        collated_batch["sparse_pts"] = batch["sparse_pts"][rand_idx]

    assert collated_batch["image"].shape == (num_rays_per_batch, 3), collated_batch["image"].shape

    # Needed to correct the random indices to their actual camera idx locations.
    indices[:, 0] = batch["image_idx"][c]
    collated_batch["indices"] = indices  # with the abs camera indices

    if keep_full_image:
        collated_batch["full_image"] = batch["image"]

    return collated_batch


class PixelSampler:  # pylint: disable=too-few-public-methods
    """Samples 'pixel_batch's from 'image_batch's.

    Args:
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
        mask_sample_start: from this iteration, sample mask_sample_ratio samples within the mask
        mask_sample_ratio: sample this ratio of pixels within masks, if masks are available
    """

    def __init__(
        self,
        num_rays_per_batch: int,
        keep_full_image: bool = False,
        mask_sample_start: Optional[float] = None,
        mask_sample_ratio: Optional[float] = None,
    ) -> None:
        self.num_rays_per_batch = num_rays_per_batch
        self.keep_full_image = keep_full_image
        self.mask_sample_start = mask_sample_start
        self.mask_sample_ratio = mask_sample_ratio
        if self.mask_sample_ratio is not None:
            assert 0 <= self.mask_sample_ratio <= 1

    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = num_rays_per_batch

    def sample(self, image_batch: Dict, step: Optional[int] = None):
        """Sample an image batch and return a pixel batch.

        Args:
            image_batch: batch of images to sample from
            step: current training step, used to determine if we should sample within masks
        """
        if self.mask_sample_ratio is not None and not isinstance(image_batch["image"], torch.Tensor):
            raise NotImplementedError

        if isinstance(image_batch["image"], list):
            image_batch = dict(image_batch.items())  # copy the dictioary so we don't modify the original
            pixel_batch = collate_image_dataset_batch_list(
                image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image
            )
        elif isinstance(image_batch["image"], BasicImages):
            image_batch = dict(image_batch.items())  # copy the dictioary so we don't modify the original
            image_batch["image"] = image_batch["image"].images
            if "mask" in image_batch:
                image_batch["mask"] = image_batch["mask"].images

            # TODO clean up
            # parser real inputs from the BasicImages structure
            if "fg_mask" in image_batch:
                image_batch["fg_mask"] = image_batch["fg_mask"].images
            if "sparse_pts" in image_batch:
                image_batch["sparse_pts"] = image_batch["sparse_pts"].images

            pixel_batch = collate_image_dataset_batch_list(
                image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image
            )
        elif isinstance(image_batch["image"], torch.Tensor):
            pixel_batch = collate_image_dataset_batch(
                image_batch,
                self.num_rays_per_batch,
                keep_full_image=self.keep_full_image,
                step=step,
                mask_sample_start=self.mask_sample_start,
                mask_sample_ratio=self.mask_sample_ratio,
            )
        else:
            raise ValueError("image_batch['image'] must be a list or torch.Tensor")
        return pixel_batch


def collate_image_dataset_batch_equirectangular(batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
    """
    Operates on a batch of equirectangular images and samples pixels to use for
    generating rays. Rays will be generated uniformly on the sphere.
    Returns a collated batch which is input to the Graph.
    It will sample only within the valid 'mask' if it's specified.

    Args:
        batch: batch of images to sample from
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """
    # TODO(kevinddchen): make more DRY
    device = batch["image"].device
    num_images, image_height, image_width, _ = batch["image"].shape

    # only sample within the mask, if the mask is in the batch
    if "mask" in batch:
        # TODO(kevinddchen): implement this
        raise NotImplementedError("Masking not implemented for equirectangular images.")

    # We sample theta uniformly in [0, 2*pi]
    # We sample phi in [0, pi] according to the PDF f(phi) = sin(phi) / 2.
    # This is done by inverse transform sampling.
    # http://corysimon.github.io/articles/uniformdistn-on-sphere/
    num_images_rand = torch.rand(num_rays_per_batch, device=device)
    phi_rand = torch.acos(1 - 2 * torch.rand(num_rays_per_batch, device=device)) / torch.pi
    theta_rand = torch.rand(num_rays_per_batch, device=device)
    indices = torch.floor(
        torch.stack((num_images_rand, phi_rand, theta_rand), dim=-1)
        * torch.tensor([num_images, image_height, image_width], device=device)
    ).long()

    c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
    collated_batch = {key: value[c, y, x] for key, value in batch.items() if key != "image_idx" and value is not None}

    assert collated_batch["image"].shape == (num_rays_per_batch, 3), collated_batch["image"].shape

    # Needed to correct the random indices to their actual camera idx locations.
    indices[:, 0] = batch["image_idx"][c]
    collated_batch["indices"] = indices  # with the abs camera indices

    if keep_full_image:
        collated_batch["full_image"] = batch["image"]

    return collated_batch


class EquirectangularPixelSampler(PixelSampler):  # pylint: disable=too-few-public-methods
    """Samples 'pixel_batch's from 'image_batch's. Assumes images are
    equirectangular and the sampling is done uniformly on the sphere.

    Args:
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """

    # overrides base method
    def sample(self, image_batch: Dict):
        pixel_batch = collate_image_dataset_batch_equirectangular(
            image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image
        )
        return pixel_batch

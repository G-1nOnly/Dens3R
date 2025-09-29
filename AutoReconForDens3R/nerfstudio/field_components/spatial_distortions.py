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

"""Space distortions."""

from typing import Optional, Union

import torch
from functorch import jacrev, vmap
from torch import nn
from torchtyping import TensorType

from nerfstudio.utils.math import Gaussians


class SpatialDistortion(nn.Module):
    """Apply spatial distortions"""

    def forward(
        self, positions: Union[TensorType["bs":..., 3], Gaussians]
    ) -> Union[TensorType["bs":..., 3], Gaussians]:
        """
        Args:
            positions: Sample to distort

        Returns:
            Union: distorted sample
        """


class SceneContraction(SpatialDistortion):
    """Contract unbounded space using the contraction was proposed in MipNeRF-360.
        We use the following contraction equation:

        .. math::

            f(x) = \\begin{cases}
                x & ||x|| \\leq 1 \\\\
                (2 - \\frac{1}{||x||})(\\frac{x}{||x||}) & ||x|| > 1
            \\end{cases}

        If the order is not specified, we use the Frobenius norm, this will contract the space to a sphere of
        radius 1. If the order is L_inf (order=float("inf")), we will contract the space to a cube of side length 2.
        If using voxel based encodings such as the Hash encoder, we recommend using the L_inf norm.

        Args:
            order: Order of the norm. Default to the Frobenius norm. Must be set to None for Gaussians.
            scale_factor: Scale positions by this factor before applying contraction, default to 1.0.
                This is useful when the [-1, 1] region (no contraction) is too small and only covers a small portion of the scene.
    """

    def __init__(
        self,
        order: Optional[Union[float, int]] = None,
        scale_factor: float = 1.0
    ) -> None:
        super().__init__()
        self.order = order
        self.scale_factor = scale_factor

    def forward(self, positions):
        def contract(x):
            x = x * self.scale_factor
            
            mag = torch.linalg.norm(x, ord=self.order, dim=-1)
            mask = mag >= 1
            x_new = x.clone()
            x_new[mask] = (2 - (1 / mag[mask][..., None])) * (x[mask] / mag[mask][..., None])

            return x_new

        if isinstance(positions, Gaussians):
            means = contract(positions.mean.clone())

            if self.scale_factor != 1.0:
                raise NotImplementedError('Scaling of gaussian covariances is not implemented.')
            
            # FIXME: covariances are not handled for now!
            contract = lambda x: (2 - (1 / torch.linalg.norm(x, ord=self.order, dim=-1, keepdim=True))) * (
                x / torch.linalg.norm(x, ord=self.order, dim=-1, keepdim=True)
            )
            jc_means = vmap(jacrev(contract))(positions.mean.view(-1, positions.mean.shape[-1]))
            jc_means = jc_means.view(list(positions.mean.shape) + [positions.mean.shape[-1]])

            # Only update covariances on positions outside the unit sphere
            mag = positions.mean.norm(dim=-1)
            mask = mag >= 1
            cov = positions.cov.clone()
            cov[mask] = jc_means[mask] @ positions.cov[mask] @ torch.transpose(jc_means[mask], -2, -1)

            return Gaussians(mean=means, cov=cov)

        return contract(positions)


class ForegroundAwareSceneContraction(SpatialDistortion):
    """Contract unbounded space using a modified version of the contraction proposed in MipNeRF-360.
    
    1. Keep the "||x|| <= 1 / x in aabb" region unchanged, i.e., to [-1, 1] / [aabb[0], aabb[1]], which should be modeled by the foreground field.
    2. Linearly contract the 1 < ||x|| <= alpha / x not in aabb but x <= alpha region to [[-1, -1/alpha], [1/alpha, 1]] (alpha>=1), which should be modeled with a separate background field.
    3. Non-linearly contract the ||x|| > alpha region to [[-2, -1], [1, 2]], which should be modeled with the background field.
    
    # FIXME: covariances are not handled for now!
    """
    def __init__(
        self,
        order: Optional[Union[float, int]] = None,
        aabb: Optional[torch.Tensor] = None,  # (2, 3)
        alpha: float = 5.0,
        scale_factor: float = 1.0
    ) -> None:
        super().__init__()
        self.order = order
        self.alpha = alpha
        self.scale_factor = scale_factor
        self.aabb = nn.Parameter(aabb, requires_grad=False) if aabb is not None else None
        if aabb is not None:
            assert aabb.abs().max() <= 1.0, f"Invalid aabb: aabb={aabb}"
            assert order == float("inf"), f"Must use inf norm with aabb: order={order}"
        assert self.scale_factor == 1.0
        assert alpha >= 1.0, f"Invalid hparam setting: alpha={alpha}"

    def forward(self, positions):
        def _contract(x):
            mag = torch.linalg.norm(x, ord=self.order, dim=-1)
            mask = mag >= 1
            x_new = x.clone()
            x_new[mask] = (2 - (1 / mag[mask][..., None])) * (x[mask] / mag[mask][..., None])

            return x_new

        def contract(x):
            x = x * self.scale_factor
            if self.aabb is None:
                mag = torch.linalg.norm(x, ord=self.order, dim=-1)
                mask = mag >= 1
            else:
                in_aabb_mask = ((x >= self.aabb[0]) & (x <= self.aabb[1])).all(-1)  # whether including = ?
                mask = ~in_aabb_mask
            x_new = x.clone()
            x_new[mask] = x[mask] / self.alpha
            return _contract(x_new)

        if isinstance(positions, Gaussians):
            means = contract(positions.mean.clone())
            # FIXME: covariances are not handled for now!
            if self.scale_factor != 1.0:
                raise NotImplementedError('Scaling of gaussian covariances is not implemented.')
            contract = lambda x: (2 - (1 / torch.linalg.norm(x, ord=self.order, dim=-1, keepdim=True))) * (
                x / torch.linalg.norm(x, ord=self.order, dim=-1, keepdim=True)
            )
            jc_means = vmap(jacrev(contract))(positions.mean.view(-1, positions.mean.shape[-1]))
            jc_means = jc_means.view(list(positions.mean.shape) + [positions.mean.shape[-1]])

            # Only update covariances on positions outside the unit sphere
            mag = positions.mean.norm(dim=-1)
            mask = mag >= 1
            cov = positions.cov.clone()
            cov[mask] = jc_means[mask] @ positions.cov[mask] @ torch.transpose(jc_means[mask], -2, -1)

            return Gaussians(mean=means, cov=cov)

        return contract(positions)

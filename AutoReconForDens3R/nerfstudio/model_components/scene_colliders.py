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
Scene Colliders
"""

from __future__ import annotations

import torch
from torch import nn
from torchtyping import TensorType

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils import profiler

class SceneCollider(nn.Module):
    """Module for setting near and far values for rays."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        super().__init__()

    def set_nears_and_fars(self, ray_bundle) -> RayBundle:
        """To be implemented."""
        raise NotImplementedError

    def forward(self, ray_bundle: RayBundle) -> RayBundle:
        """Sets the nears and fars if they are not set already."""
        if ray_bundle.nears is not None and ray_bundle.fars is not None:
            return ray_bundle
        return self.set_nears_and_fars(ray_bundle)


class AABBBoxCollider(SceneCollider):
    """Module for colliding rays with the scene box to compute near and far values.

    Args:
        scene_box: scene box to apply to dataset
    """

    def __init__(self, scene_box: SceneBox, near_plane: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.scene_box = scene_box
        self.near_plane = near_plane

    def _intersect_with_aabb(
        self, rays_o: TensorType["num_rays", 3], rays_d: TensorType["num_rays", 3], aabb: TensorType[2, 3]
    ):
        """Returns collection of valid rays within a specified near/far bounding box along with a mask
        specifying which rays are valid

        Args:
            rays_o: (num_rays, 3) ray origins
            rays_d: (num_rays, 3) ray directions
            aabb: (2, 3) This is [min point (x,y,z), max point (x,y,z)]
        """
        # avoid divide by zero
        dir_fraction = 1.0 / (rays_d + 1e-6)

        # x
        t1 = (aabb[0, 0] - rays_o[:, 0:1]) * dir_fraction[:, 0:1]
        t2 = (aabb[1, 0] - rays_o[:, 0:1]) * dir_fraction[:, 0:1]
        # y
        t3 = (aabb[0, 1] - rays_o[:, 1:2]) * dir_fraction[:, 1:2]
        t4 = (aabb[1, 1] - rays_o[:, 1:2]) * dir_fraction[:, 1:2]
        # z
        t5 = (aabb[0, 2] - rays_o[:, 2:3]) * dir_fraction[:, 2:3]
        t6 = (aabb[1, 2] - rays_o[:, 2:3]) * dir_fraction[:, 2:3]

        nears = torch.max(
            torch.cat([torch.minimum(t1, t2), torch.minimum(t3, t4), torch.minimum(t5, t6)], dim=1), dim=1
        ).values
        fars = torch.min(
            torch.cat([torch.maximum(t1, t2), torch.maximum(t3, t4), torch.maximum(t5, t6)], dim=1), dim=1
        ).values

        # clamp to near plane
        near_plane = self.near_plane if self.training else 0
        nears = torch.clamp(nears, min=near_plane)
        fars = torch.maximum(fars, nears + 1e-6)  # NOTE: this would cause numerical issue when sample along the ray

        return nears, fars

    def set_nears_and_fars(self, ray_bundle: RayBundle) -> RayBundle:
        """Intersects the rays with the scene box and updates the near and far values.
        Populates nears and fars fields and returns the ray_bundle.

        Args:
            ray_bundle: specified ray bundle to operate on
        """
        aabb = self.scene_box.aabb
        nears, fars = self._intersect_with_aabb(ray_bundle.origins, ray_bundle.directions, aabb)
        ray_bundle.nears = nears[..., None]
        ray_bundle.fars = fars[..., None]
        return ray_bundle


class NearFarCollider(SceneCollider):
    """Sets the nears and fars with fixed values.

    Args:
        near_plane: distance to near plane
        far_plane: distance to far plane
    """

    def __init__(self, near_plane: float, far_plane: float, **kwargs) -> None:
        self.near_plane = near_plane
        self.far_plane = far_plane
        super().__init__(**kwargs)

    def set_nears_and_fars(self, ray_bundle: RayBundle) -> RayBundle:
        ones = torch.ones_like(ray_bundle.origins[..., 0:1])
        near_plane = self.near_plane if self.training else 0
        ray_bundle.nears = ones * near_plane
        ray_bundle.fars = ones * self.far_plane
        return ray_bundle


class SphereCollider(SceneCollider):
    """Sets the nears and fars with ray-sphere intersection.
    """

    def __init__(self, radius: float = 1., near_plane: float = 0.1, far_plane: float = 1000.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.radius = radius
        self.near_plane = near_plane
        self.far_plane = far_plane

    def forward(self, ray_bundle: RayBundle) -> RayBundle:
        rr = self.radius ** 2
        ray_cam_dot = (ray_bundle.directions * ray_bundle.origins).sum(dim=-1, keepdims=True)  # negative
        no_intersect_mask = ray_cam_dot >= 0
        dd = ray_bundle.origins.norm(p=2, dim=-1, keepdim=True) ** 2 - ray_cam_dot ** 2
        no_intersect_mask |= (dd >= rr)
        half_len_squared = rr - dd
        
        intersections = torch.sqrt(half_len_squared) * dd.new_tensor([-1, 1]) - ray_cam_dot
        nears, fars = intersections[:, [0]], intersections[:, [1]]
        nears = torch.where(no_intersect_mask, torch.full_like(nears, self.near_plane), nears)
        fars = torch.where(no_intersect_mask, torch.full_like(fars, self.far_plane), fars)
        
        ray_bundle.nears = nears
        ray_bundle.fars = fars
        if isinstance(ray_bundle.metadata, dict):
            ray_bundle.metadata["no_intersect_mask"] = no_intersect_mask
        else:
            ray_bundle.metadata = {"no_intersect_mask": no_intersect_mask}
        
        return ray_bundle


class AABBBoxNearFarCollider(SceneCollider):
    """Use AABBBoxCollider for rays intersecting the scene box and NearFarCollider otherwise.
    
    NOTE: the distance nears and fars inferred from ray-aabb intersection can be near, which cannot be handled properly using float32.
    """
    
    def __init__(self, scene_box: SceneBox, near_plane: float = 0.1, far_plane: float = 1000.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.scene_box = scene_box
        self.near_plane = near_plane
        self.far_plane = far_plane
    
    def _intersect_with_aabb(
        self, rays_o: TensorType["num_rays", 3], rays_d: TensorType["num_rays", 3], aabb: TensorType[2, 3]
    ):
        """Compute nears and fars for all rays. """
        # avoid divide by zero
        dir_fraction = 1.0 / (rays_d + 1e-6)

        # x
        t1 = (aabb[0, 0] - rays_o[:, 0:1]) * dir_fraction[:, 0:1]
        t2 = (aabb[1, 0] - rays_o[:, 0:1]) * dir_fraction[:, 0:1]
        # y
        t3 = (aabb[0, 1] - rays_o[:, 1:2]) * dir_fraction[:, 1:2]
        t4 = (aabb[1, 1] - rays_o[:, 1:2]) * dir_fraction[:, 1:2]
        # z
        t5 = (aabb[0, 2] - rays_o[:, 2:3]) * dir_fraction[:, 2:3]
        t6 = (aabb[1, 2] - rays_o[:, 2:3]) * dir_fraction[:, 2:3]
        
        tx_min, tx_max = torch.minimum(t1, t2), torch.maximum(t1, t2)
        ty_min, ty_max = torch.minimum(t3, t4), torch.maximum(t3, t4)
        tz_min, tz_max = torch.minimum(t5, t6), torch.maximum(t5, t6)
        
        # no_intersect_mask = torch.logical_or(
        #     torch.logical_or(tx_min > ty_max, tx_min > tz_max),
        #     torch.logical_or(ty_min > tx_max, tz_min > tx_max)
        # )[..., 0]
        
        nears = torch.max(torch.cat([tx_min, ty_min, tz_min], dim=1), dim=1).values
        fars = torch.min(torch.cat([tx_max, ty_max, tz_max], dim=1), dim=1).values
        no_intersect_mask = torch.logical_or(nears - fars >= -1e-3, fars <= 0)  # use nears >= fars might cause the nears and fars being too close, which finally leads to negative weights due to numerical issues.
        bottom_intersect_mask = (~no_intersect_mask) & (fars == t4[..., 0])  # fars == t4 & t4 == ty_max
        nears = torch.where(no_intersect_mask, torch.full_like(nears, self.near_plane), nears)  # type: ignore
        fars = torch.where(no_intersect_mask, torch.full_like(fars, self.far_plane), fars)  # type: ignore
        return nears, fars, no_intersect_mask, bottom_intersect_mask
    
    def set_nears_and_fars(self, ray_bundle: RayBundle) -> RayBundle:
        """Intersects the rays with the scene box and updates the near and far values.
        Populates nears and fars fields and returns the ray_bundle.

        Args:
            ray_bundle: specified ray bundle to operate on
        """
        # TODO: use NearFarCollider if not intersection found
        aabb = self.scene_box.aabb
        nears, fars, no_intersect_mask, bottom_intersect_mask = self._intersect_with_aabb(
            ray_bundle.origins, ray_bundle.directions, aabb
        )
        ray_bundle.nears = nears[..., None]
        ray_bundle.fars = fars[..., None]
        if isinstance(ray_bundle.metadata, dict):
            ray_bundle.metadata["no_intersect_mask"] = no_intersect_mask[..., None]
            ray_bundle.metadata["bottom_intersect_mask"] = bottom_intersect_mask[..., None]
        else:
            ray_bundle.metadata = {
                "no_intersect_mask": no_intersect_mask[..., None],
                "bottom_intersect_mask": bottom_intersect_mask[..., None],
            }
        return ray_bundle

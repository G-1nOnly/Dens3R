from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Optional

import torch
from rich.console import Console

try:
    import faiss
    import faiss.contrib.torch_utils
except ImportError as err:
    pass
from torch import nn
from torch.nn import functional as F

CONSOLE = Console(width=120)


class BasicPointClouds(nn.Module):
    """A primitive struct for holding pointcloud related data (e.g., points, colors, features, etc.)

    The purpose of this is to have a special struct wrapping around a list so that the
    nerfstudio_collate fn and other parts of the code recognise this as a struct to leave alone
    instead of reshaping or concatenating into a single tensor (since this will likely be used
    for cases where we have images of different sizes and shapes).

    We assume that all images share the same pointcloud, thus only the first element of
    the list of pointclouds will be kept by the nerfstudio_collate.
    """

    # TOOD: make {fg,bg,plane}_pts non-optional

    def __init__(
        self,
        fg_pts: Optional[torch.Tensor] = None,
        bg_pts: Optional[torch.Tensor] = None,
        plane_pts: Optional[torch.Tensor] = None,
        fg_feats: Optional[torch.Tensor] = None,
        bg_feats: Optional[torch.Tensor] = None,
        plane_feats: Optional[torch.Tensor] = None,
        # data for ptcd regularization
        udf_plane_to_fg: Optional[torch.Tensor] = None,
        plane_sdf_deviation_thr_ptwise: Optional[torch.Tensor] = None,
        fg_sdf_deviation_thr_global: Optional[torch.Tensor] = None,
        fg_sdf_deviation_thr_ptwise: Optional[torch.Tensor] = None,
        # data for feature-field based segmentation
        fg_nn_index: Optional[faiss.GpuIndexFlatL2] = None,
        bg_nn_index: Optional[faiss.GpuIndexFlatL2] = None,
        plane_nn_index: Optional[faiss.GpuIndexFlatL2] = None,
    ):
        super().__init__()
        self.fg_pts = fg_pts
        self.bg_pts = bg_pts
        self.plane_pts = plane_pts
        self.fg_feats = fg_feats
        self.bg_feats = bg_feats
        self.plane_feats = plane_feats

        self.udf_plane_to_fg = udf_plane_to_fg
        self.fg_sdf_deviation_thr_global = fg_sdf_deviation_thr_global
        self.fg_sdf_deviation_thr_ptwise = fg_sdf_deviation_thr_ptwise
        self.plane_sdf_deviation_thr_ptwise = plane_sdf_deviation_thr_ptwise

        self.fg_nn_index = fg_nn_index
        self.bg_nn_index = bg_nn_index
        self.plane_nn_index = plane_nn_index

        self._register_all_tensors_as_buffers()

    def build_nn_search_index(self, separate_bg_plane: bool = False):
        """Build faiss gpu index for fast nn search"""
        for f in ["fg_feats", "bg_feats", "plane_feats"]:
            assert getattr(self, f, None) is not None, f"BasicPointClouds.{f} is None"
        nc_feat = self.fg_feats.shape[-1]

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = True

        # build fg inex
        self.fg_nn_index = faiss.GpuIndexFlatL2(res, nc_feat, cfg)
        self.fg_nn_index.add(F.normalize(self.fg_feats, p=2, dim=-1))
        CONSOLE.print(f"Using GpuIndexFlatL2 for fg NN search. (#fg: {len(self.fg_feats)})")

        # build bg index
        if not separate_bg_plane:
            self.bg_nn_index = faiss.GpuIndexFlatL2(res, nc_feat, cfg)
            self.bg_nn_index.add(F.normalize(torch.cat([self.bg_feats, self.plane_feats], 0), p=2, dim=-1))
            CONSOLE.print(
                f"Using GpuIndexFlatL2 for (bg, plane) NN search."
                f"(#bg: {len(self.bg_feats)} | #plane: {len(self.plane_pts)}))"
            )
            # TODO: build a mapper to distinguish bg and plane pts
            # TODO: tune the number of fg, bg and plane pts
        else:
            self.bg_nn_index = faiss.GpuIndexFlatL2(res, nc_feat, cfg)
            self.bg_nn_index.add(F.normalize(self.bg_feats, p=2, dim=-1))
            CONSOLE.print(f"Using GpuIndexFlatL2 for bg NN search. (#bg: {len(self.bg_feats)})")
            self.plane_nn_index = faiss.GpuIndexFlatL2(res, nc_feat, cfg)
            self.plane_nn_index.add(F.normalize(self.plane_feats, p=2, dim=-1))
            CONSOLE.print(f"Using GpuIndexFlatL2 for bg NN search. (#bg: {len(self.plane_feats)})")

    def sample_fg_points(self, n_pts: int, replace: bool = True):
        if len(self.fg_pts) <= n_pts:
            return self.fg_pts, self.fg_sdf_deviation_thr_ptwise

        rand_inds = (
            torch.randint(0, len(self.fg_pts), (n_pts,), device=self.device)
            if replace
            else torch.randperm(len(self.fg_pts), device=self.device)[:n_pts]
        )
        fg_pts = self.fg_pts[rand_inds]
        fg_sdf_thrs = self.fg_sdf_deviation_thr_ptwise[rand_inds]
        return fg_pts, fg_sdf_thrs

    def sample_plane_points(self, n_pts: int, replace: bool = True, rand_sample_half: bool = False):
        if rand_sample_half:
            raise NotImplementedError

        if len(self.plane_pts) <= n_pts:
            return self.plane_pts, self.udf_plane_to_fg, self.plane_sdf_deviation_thr_ptwise

        rand_inds = (
            torch.randint(0, len(self.plane_pts), (n_pts,), device=self.device)
            if replace
            else torch.randperm(len(self.plane_pts), device=self.device)[:n_pts]
        )
        plane_pts = self.plane_pts[rand_inds]
        udf_plane_to_fg = self.udf_plane_to_fg[rand_inds]
        plane_sdf_thrs = self.plane_sdf_deviation_thr_ptwise[rand_inds]

        return plane_pts, udf_plane_to_fg, plane_sdf_thrs

    @property
    def device(self):
        return self.fg_pts.device

    def _register_all_tensors_as_buffers(self):
        """register all Tensor attributes as buffers to automatically move them to the target device"""
        _attr_names = list(vars(self).keys())
        for _name in _attr_names:
            _attr = getattr(self, _name)
            if not isinstance(_attr, torch.Tensor):
                continue

            delattr(self, _name)
            self.register_buffer(_name, _attr, persistent=False)

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, ItemsView, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import pycolmap
import torch
from jaxtyping import Float
from sklearn.decomposition import PCA
from torch import Tensor

from auto_decomp.utils import tqdm as tqdm_utils
from auto_decomp.utils import viz_3d
from auto_decomp.utils.aggregation import (
    multiview_feature_aggregation,
    multiview_feature_extraction,
)
from auto_decomp.utils.misc import EarlyTermination
from auto_decomp.utils.viz_3d import PlotlySaveMode

from .convert import open3d_ptcd_from_numpy


def build_color_ptcd_from_colmap(
    rec: pycolmap.Reconstruction, image_dir: Path, max_reproj_error: float, min_track_length: int, device: str
) -> Tuple[Tensor, Tensor]:
    imgs = dict()

    for img_id in tqdm_utils.tqdm(rec.reg_image_ids(), desc="Loading registered images"):
        img_name = rec.images[img_id].name
        img_path = image_dir / img_name
        img_rgb = cv2.imread(str(img_path))[..., ::-1]
        imgs[img_id] = torch.tensor(img_rgb.copy(), dtype=torch.float32, device=device).permute(2, 0, 1)

    p3Ds_filtered, colors_filtered = multiview_feature_extraction(
        rec,
        {"rgb": imgs},
        interpolate="bilinear",
        align_corners=False,
        device=device,
        max_reproj_error=max_reproj_error,
        min_track_length=min_track_length,
    )
    colors_filtered = multiview_feature_aggregation(p3Ds_filtered, colors_filtered, mode="mean")["rgb"]

    # p3Ds_filtered, colors_filtered = p3Ds_filtered.cpu(), colors_filtered.cpu()
    colors_filtered = colors_filtered.round().clamp(0, 256).to(torch.uint8)
    return p3Ds_filtered, colors_filtered


def build_feature_ptcd(
    rec: pycolmap.Reconstruction,
    feat_dir: Path,
    max_reproj_error: float,
    min_track_length: int,
    device: str,
    feature_dtype: Optional[str] = None,
    load_all_features: bool = False,  # there might be multiple feature sources
    main_feature_key: str = "features",
) -> Tuple[Dict[str, Any], torch.dtype]:
    # load dino features
    all_mv_feats = defaultdict(dict)  # {feat_name: {img_id: Tensor(C, H, W)}}
    mv_cls_feats = defaultdict(dict)  # {feat_name: {imd_id: Tensor(C,)}
    for img_id in tqdm_utils.tqdm(rec.reg_image_ids(), desc="Loading DINO features"):
        img_name = rec.images[img_id].name
        feat_path = (feat_dir / img_name).with_suffix(".npz")
        if not feat_path.exists():
            raise EarlyTermination(f"Cannot find the required feature file: {str(feat_path)}")

        try:
            vit_feats = np.load(feat_path)
        except ValueError as ve:
            raise EarlyTermination(f"Invalid npz file: {str(feat_path)} ({ve})")

        if feature_dtype is None:
            feature_dtype = next(iter(vit_feats.values())).dtype
        elif isinstance(feature_dtype, str):
            feature_dtype = getattr(torch, feature_dtype)

        if load_all_features:
            for feat_name, feat in vit_feats.items():
                # feat: (h, w, c) / (c, )
                _mv_feats = torch.tensor(feat, dtype=torch.float32, device=device)  # (h, w, c) / (c, )
                if _mv_feats.ndim == 3:
                    _mv_feats = _mv_feats.permute(2, 0, 1)
                    all_mv_feats[feat_name][img_id] = _mv_feats  # type: ignore
                else:
                    assert "cls_" in feat_name
                    mv_cls_feats[feat_name][img_id] = _mv_feats  # type: ignore
        else:  # only load the main_feature_key
            feat = torch.tensor(vit_feats[main_feature_key], dtype=torch.float32, device=device).permute(2, 0, 1)
            all_mv_feats[main_feature_key][img_id] = feat  # type: ignore

    # sanity check
    for feat_name, _mv_feats in all_mv_feats.items():
        if len(_mv_feats) != len(rec.reg_image_ids()):
            raise EarlyTermination(
                f"Missing features for some frames ({len(_mv_feats)}/{len(rec.reg_image_ids())})" f" ({str(feat_dir)})"
            )

    # interpolate & aggregate mv-features
    p3Ds_filtered, feats_filtered = multiview_feature_extraction(
        rec,
        all_mv_feats,
        interpolate="nearest",
        align_corners=False,
        device=device,
        max_reproj_error=max_reproj_error,
        min_track_length=min_track_length,
    )
    feats_filtered = multiview_feature_aggregation(p3Ds_filtered, feats_filtered, mode="mean")

    cls_feats = (
        {k: torch.stack(list(mv_feats.values()), 0).mean(0) for k, mv_feats in mv_cls_feats.items()}  # type: ignore
        if len(mv_cls_feats) > 0
        else {}
    )

    features = {
        "point_feats": feats_filtered,  # Dict[str, Tensor]: {feature_name: (L, C)}
        "global_feats": cls_feats,  # Dict[str, Tensor]: {'cls_{facet}': (C)}
    }  # main feature key: "features"
    return features, feature_dtype


class NeuralPointCloud:
    MAIN_FEATURE_KEY = "features"

    def __init__(
        self,
        p3Ds: Float[Tensor, "n_p 3"],
        colors: Float[Tensor, "n_p n_c"],
        pointwise_features: Dict[str, Float[Tensor, "n_p n_f"]],  # features from multiple sources
        imagewise_features: Dict[str, Float[Tensor, "n_i n_f"]],  # global image-wise features
        device: Optional[str] = None,
        rec: Optional[pycolmap.Reconstruction] = None,
        vis_dir: Optional[Path] = None,
        plotly_save_mode: PlotlySaveMode = PlotlySaveMode.html,
        feature_dtype: Optional[torch.dtype] = torch.float16,
    ):
        self.rec = rec
        self.device = device if device is not None else p3Ds.device
        self.vis_dir = vis_dir
        self.plotly_save_mode = plotly_save_mode
        self.feature_dtype = feature_dtype

        # point-wise attrs
        self.p3Ds = p3Ds
        self.colors = colors
        self.pointwise_features = pointwise_features

        # global attrs
        self.imagewise_features = imagewise_features

    @classmethod
    def from_pycolmap_rec(
        cls,
        rec: pycolmap.Reconstruction,
        image_dir: Path,
        feat_dir: Path,
        max_reproj_error: float,
        min_track_length: int,
        device: str,
        feature_dtype: Optional[str] = None,
        load_all_features: bool = False,
        **kwargs,
    ):
        p3Ds, colors = build_color_ptcd_from_colmap(rec, image_dir, max_reproj_error, min_track_length, device)
        features, feature_dtype = build_feature_ptcd(
            rec,
            feat_dir,
            max_reproj_error,
            min_track_length,
            device,
            feature_dtype=feature_dtype,
            load_all_features=load_all_features,
            main_feature_key=cls.MAIN_FEATURE_KEY,
        )  # type: ignore
        torch.cuda.empty_cache()

        return cls(
            p3Ds,
            colors,
            features["point_feats"],
            features["global_feats"],
            device=device,
            rec=rec,
            feature_dtype=feature_dtype,
            **kwargs,
        )

    def statistical_outlier_removal(self, n_neighbors, std_ratio):
        o3d_pcd = self.to_open3d_ptcd()
        _, stat_inlier_inds = o3d_pcd.remove_statistical_outlier(nb_neighbors=n_neighbors, std_ratio=std_ratio)
        self.update_ptcd_with_indices(stat_inlier_inds)

    def to_open3d_ptcd(self):  # points only, no attributes
        return open3d_ptcd_from_numpy(self.p3Ds.cpu().numpy())

    def update_ptcd_with_indices(self, inds):
        inds = torch.tensor(inds, dtype=torch.long, device=self.device)
        self.p3Ds = self.p3Ds[inds]
        self.colors = self.colors[inds]
        self.pointwise_features = {k: v[inds] for k, v in self.pointwise_features.items()}

    def update_ptcd_with_mask(self):
        raise NotImplementedError

    def visualize_ptcd(self, base_name):
        """Visualize the pointcloud with rgb and feature pca colors."""
        self._draw_color_ptcd(base_name)
        self._draw_feature_ptcd(base_name)

    def _draw_color_ptcd(self, base_name):
        fig, fig_name = viz_3d.init_figure(), f"{base_name}_rgb-ptcd"
        p3Ds = self.p3Ds.cpu().numpy()
        colors = self.colors.cpu().numpy()
        viz_3d.plot_points(fig, p3Ds, color=colors, name=fig_name)
        if self.rec is not None:
            viz_3d.plot_cameras(fig, self.rec, color="lavender", legendgroup=fig_name, size=1.0)
        viz_3d.save_fig(fig, self.vis_dir, fig_name, mode=self.plotly_save_mode)

    def _draw_feature_ptcd(self, base_name):
        fig, fig_name = viz_3d.init_figure(), f"{base_name}_feat-ptcd"
        p3Ds = self.p3Ds.cpu().numpy()
        feats = self.pointwise_features[self.MAIN_FEATURE_KEY].cpu().numpy()

        feats_pca = PCA(n_components=3).fit_transform(feats)
        components = feats_pca[:, -3:]
        _min, _max = components.min(axis=0), components.max(axis=0)
        feats_rgb = ((components - _min) / (_max - _min) * 255).round().astype(np.uint8)

        plotly_color = [f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})" for rgb in feats_rgb]  # FIXME
        viz_3d.plot_points(fig, p3Ds, color=plotly_color, name=fig_name)
        if self.rec is not None:
            viz_3d.plot_cameras(fig, self.rec, color="lavender", legendgroup=fig_name, size=1.0)
        viz_3d.save_fig(fig, self.vis_dir, fig_name, mode=self.plotly_save_mode)

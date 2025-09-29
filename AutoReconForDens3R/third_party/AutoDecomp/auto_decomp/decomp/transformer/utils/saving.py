import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from hydra_zen import store
from jaxtyping import Float
from loguru import logger
from natsort import natsorted

from auto_decomp.utils import viz_3d
from auto_decomp.utils.geometry import box3d


@dataclass
class SfMTransformerSaveConfig:
    enabled: bool = False  # only for testing
    save_features: bool = False  # save per-pt features for fine-decomposition
    save_dirname: str = "auto-deocomp_sfm-transformer"
    sfm_dirname: str = "triangulate_loftr-1920000_sequential_np-10"
    use_cache: bool = True
    """infer sfm_dirname from cache"""
    cache_name: str = ".cache.json"
    bbox_normalize_mode: str = "side"
    """'diag' (for NeRF++) / 'side' (for MipNeRF-360 Scene Contraction w/ inf-norm)"""
    bbox_diag_len: Optional[float] = 1.9
    """rescale the object bbox s.t., its diagonal length equals to bbox_diag_len"""
    bbox_side_len: Optional[float] = 1.9
    """rescale the object bbox s.t., its maximum side length equals to bbox_diag_len"""


saving_store = store(group="decomp/transformer/saving")
saving_store(SfMTransformerSaveConfig, name="train")
saving_store(SfMTransformerSaveConfig, name="test_autorecon", enabled=True, save_features=False)
saving_store(SfMTransformerSaveConfig, name="test_autorecon_cvpr", enabled=True, save_features=False, bbox_side_len=2.0)
saving_store(SfMTransformerSaveConfig, name="test_autorecon++", enabled=True, save_features=True)
store.add_to_hydra_store()


class SaveResults:
    def __init__(self, cfg: SfMTransformerSaveConfig):
        self.cfg = cfg
        self._sanity_check()

    def _sanity_check(self):
        if self.cfg.bbox_diag_len is not None and self.cfg.bbox_diag_len > 2.0:
            raise ValueError(
                f"Invalid normalization (bbox_diag_len={self.cfg.bbox_diag_len}),"
                "we assume the foreground object is normalized into a unit sphere."
            )
        if self.cfg.bbox_side_len is not None and self.cfg.bbox_side_len > 2.0:
            raise ValueError(
                f"Invalid normalization (bbox_side_len={self.cfg.bbox_side_len}),"
                "we assume the foreground object is normalized into a unit cube."
            )

    def save_annotations(self, batch: Dict[str, Any], all_decomp_results: Dict[str, Dict[str, Any]]):
        """Save annotations for reconstruction & decomposition,
        including camera poses, decomposed points clouds, etc.
        """
        if not self.cfg.enabled:
            return

        bs = len(batch["xyz"])
        if bs != 1:
            raise NotImplementedError

        for b_id in range(bs):
            inst_name = batch["inst_name"][b_id]
            decomp_results = all_decomp_results[inst_name]
            inst_dir = Path(str(batch["inst_dir"][b_id]))
            if self.cfg.use_cache:
                cache_path = inst_dir / self.cfg.cache_name
                with open(cache_path, "r") as f:
                    cache = json.load(f)
                sfm_dirname = Path(cache["sfm_dir"]).parent.stem
                logger.info(f"Parsing sfm_dirname from cache: {sfm_dirname}")
            else:
                sfm_dirname = self.cfg.sfm_dirname
            assert (inst_dir / sfm_dirname).exists()
            save_dir = inst_dir / sfm_dirname / self.cfg.save_dirname
            save_dir.mkdir(exist_ok=True)
            norm_len = self.cfg.bbox_diag_len if self.cfg.bbox_normalize_mode == "diag" else self.cfg.bbox_side_len
            save_name_suffix = f"cameras_norm-obj-{self.cfg.bbox_normalize_mode}-{norm_len}"

            translation_oo2on, scale_oo2on = self._compute_transform_new_object_frame(decomp_results["aabb_min_max"])
            object_anno = self._build_object_anno(translation_oo2on, scale_oo2on, decomp_results)
            pose_anno = self._build_pose_anno(  # build camera poses (IDR format)
                translation_oo2on,
                scale_oo2on,
                batch["normalize_min"][b_id].cpu().numpy(),
                batch["normalize_scale"][b_id].cpu().numpy(),
                decomp_results["T44_w2o"],
                dict(natsorted(batch["poses"].items(), key=lambda item: item[0])),  # FIXME: only support bs=1 for now
            )

            # visualize
            self.visualize_object_anno(object_anno, save_dir / "vis_decomposition")

            # save annotations
            np.savez(save_dir / f"cameras_{save_name_suffix}.npz", **pose_anno)
            np.savez(save_dir / f"objects_{save_name_suffix}.npz", **object_anno)
            logger.info(f"Annotations saved: {save_dir}")

    def _compute_transform_new_object_frame(
        self, aabb_min_max: Float[np.ndarray, "4 4"]
    ) -> Tuple[Float[np.ndarray, "3"], float]:
        """Compute transformation: object-centric frame (plane-aligned-pca) -> centered & rescaled object-centric frame

        Retruns:
            translation_oo2on: old object-centric frame -> new object-centric frame
                # NOTE: not a euclidean transformation
        """
        object_center = (aabb_min_max[1] + aabb_min_max[0]) / 2
        object_extent = aabb_min_max[1] - aabb_min_max[0]

        if self.cfg.bbox_normalize_mode == "diag":
            aabb_diag_len = np.linalg.norm(object_extent)
            scale_oo2on = self.cfg.bbox_diag_len / aabb_diag_len  # type: ignore
        else:
            assert self.cfg.bbox_normalize_mode == "side"
            scale_oo2on = self.cfg.bbox_side_len / np.max(object_extent)
        translation_oo2on = -object_center * scale_oo2on

        return translation_oo2on, scale_oo2on

    def _build_object_anno(
        self, translation_oo2on: Float[np.ndarray, "3"], scale_oo2on: float, decomp_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Transform decomposition results into the new object-centric frame."""
        fg_pts, bg_pts, plane_pts = map(
            lambda k: scale_oo2on * decomp_results[k] + translation_oo2on,
            ["fg_pts", "bg_pts", "plane_pts"],
        )
        udf_plane_to_fg = decomp_results["udf_plane_to_fg"] * scale_oo2on
        plane_params = decomp_results.get("plane_params", None)
        if plane_params is not None:  # FIXME: debug
            plane_params[-1] = plane_params[-1] * scale_oo2on - translation_oo2on[1]
        aabb_min_max = decomp_results["aabb_min_max"]
        object_extent = (aabb_min_max[1] - aabb_min_max[0]) * scale_oo2on

        # build decomposition results
        # NOTE: different from camera poses, which decompose pose to (w->c & o->w), we save o->c directly here
        # TODO: factor out the normalization matrix
        object_scale_mat = object_scale_mat_acc = np.diag([*(object_extent / 2), 1]).astype(np.float32)
        object_anno = {
            "scale_mat_0": object_scale_mat,
            "scale_mat_accurate_0": object_scale_mat_acc,
            "fg_pts": fg_pts,
            "bg_pts": bg_pts,
            "plane_pts": plane_pts,
            "udf_plane_to_fg": udf_plane_to_fg,
            "plane_params": plane_params,
        }
        if self.cfg.save_features:
            object_anno.update({f"{k}_feats": decomp_results[f"aux_{k}"]["feat"] for k in ["fg", "bg", "plane"]})
        return object_anno

    def _build_pose_anno(
        self,
        translation_oo2on: Float[np.ndarray, "3"],
        scale_oo2on: float,
        normalize_min: Float[np.ndarray, "3"],
        normalize_scale: Float[np.ndarray, "1"],
        T44_wn2oo: Float[np.ndarray, "4 4"],
        camera_poses: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        NOTE: we normalize the pointcloud to (-1, 1) for Transformer inference
              with: x' = (x - min) * scale * 2 - 1, so we need to normalize the camera poses accordingly.
        TODO: normalize the camera poses at the same time during data loading.
        """
        # build transformation (IDR normalization matrix)
        # (world (w) -> normalized world (wn) -> plane-aligned-pca world (oo) -> object-centric world (on))
        T44_w2wn, T44_oo2on = np.eye(4), np.eye(4)
        T44_w2wn[:3, :3] *= 2 * normalize_scale
        T44_w2wn[:3, -1] = -(2 * normalize_scale * normalize_min + 1)
        T44_oo2on[:3, -1] = translation_oo2on
        T44_oo2on[:3, :3] *= scale_oo2on
        T44_w2on = T44_oo2on @ T44_wn2oo @ T44_w2wn  # NOTE: not euclidean matrices
        T44_on2w = np.linalg.inv(T44_w2on)

        # build pose annotation
        pose_anno = {}
        for f_id, (img_name, cam_pose) in enumerate(camera_poses.items()):
            K33, T44_w2c = cam_pose["K33"][0].cpu().numpy(), cam_pose["T44_w2c"][0].cpu().numpy()
            K44 = np.eye(4)
            K44[:3, :3] = K33
            pose_anno[f"image_name_{f_id}"] = img_name
            pose_anno[f"scale_mat_{f_id}"] = T44_on2w
            pose_anno[f"scale_mat_inv_{f_id}"] = T44_w2on
            pose_anno[f"world_mat_{f_id}"] = K44 @ T44_w2c  # KRT

        return pose_anno

    def visualize_object_anno(self, object_anno: Dict[str, Any], save_path: Path):
        """Visualize object annotation."""
        fig = viz_3d.init_figure()
        # 1. visualize fg, bg, plane segmentation
        fg_pts, bg_pts, plane_pts = map(object_anno.get, ["fg_pts", "bg_pts", "plane_pts"])
        viz_3d.plot_points(fig, fg_pts, color="#ffb6c1", ps=2, name="fg")  # lpink
        viz_3d.plot_points(fig, bg_pts, color="#ffff9e", ps=2, name="bg")  # lyellow
        viz_3d.plot_points(fig, plane_pts, color="#90ee90", ps=2, name="plane")  # lgreen

        # TODO: visualize the fitted plane

        # 2. visualize bbox
        obj_scale_mat = object_anno["scale_mat_0"].astype(np.float32)
        obj_bbox_min = np.array([-1.0, -1.0, -1.0, 1.0])
        obj_bbox_max = np.array([1.0, 1.0, 1.0, 1.0])
        obj_bbox_min = obj_scale_mat @ obj_bbox_min[:, None]
        obj_bbox_max = obj_scale_mat @ obj_bbox_max[:, None]
        obj_aabb = np.stack([obj_bbox_min[:3, 0], obj_bbox_max[:3, 0]], 0)
        obj_bbox_verts = box3d.min_max_to_vertices(obj_aabb)

        viz_3d.plot_cube(
            fig,
            vertices=obj_bbox_verts,
            color="lightskyblue",
            opacity=0.3,
            name="fg aabb",
            show_legend=True,
        )

        # 3. save result (html & json)
        viz_3d.save_fig(fig, save_path.parent, save_path.stem, mode="both")
        logger.info(f"Object anno visualization saved: {save_path}")

    def save_decomp_visualization(self):
        """Save decomposition results for visualization."""
        pass

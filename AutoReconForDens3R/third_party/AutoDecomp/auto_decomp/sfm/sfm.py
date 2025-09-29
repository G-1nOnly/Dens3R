import copy
import json
import shutil
import subprocess
from dataclasses import dataclass, make_dataclass
from pathlib import Path
from typing import Optional

import hydra
import natsort
import numpy as np
import pycolmap
import ray
from hloc import (
    extract_features,
    match_features,
    pairs_from_exhaustive,
    pairs_from_poses,
    reconstruction,
    triangulation,
)
from hydra.core.config_store import ConfigStore
from hydra_zen import store
from loftr.tools import hloc_match_features as loftr_match_features
from loguru import logger
from ray.util.queue import Queue
from rich.traceback import install
from typing_extensions import Literal

from auto_decomp.sfm import pairs_from_sequential
from auto_decomp.utils import viz_3d

install(show_locals=False)


@dataclass
class SfMConfig:
    # basic configs
    force_rerun: bool = False
    data_root: Path = Path("data/")
    inst_rel_dir: Optional[str] = None
    """the relative path to the instance directory"""
    image_dirname: str = "images"
    inst_list_path: Optional[Path] = None
    """path to the file containing a list of instances"""
    n_images: Optional[int] = None
    """evenly subsample n_images for reconstruction"""
    make_plot: bool = True
    delete_sfm_dir: bool = True
    """only keep the manhattan-aligned sfm results"""
    cache_name: str = ".cache.json"
    """save cache for further use"""

    # parallel configs
    n_feature_workers: int = 1
    n_recon_workers: int = 1

    # colmap configs
    sfm_mode: str = "sfm"
    share_intrinsics: bool = True
    matching_mode: str = "exhaustive"  # TODO: make a separate config group for different matching modes
    n_matching_neighbors: int = 10
    """number of matching pairs to consider for sequential matcher / viewpoint-angle based matcher
    for sequential matcher, overlap = ceil(n_matching_neighbors / 2)
    """
    loop_detection_frequency: int = 1
    loop_detection_num_images: int = 15

    # triangulation configs
    reference_sfm_reldir: str = "sfm"
    """the reference sfm directory for triangulation"""
    use_cache: bool = False
    """read ref_sfm_dir from cache for triangulation"""

    # feature extractor & matcher
    extractor: str = "superpoint_aachen"
    matcher: str = "superglue"
    resize_max_area: int = 1920000
    """resize images to have a maximum area of this number, for LoFTR matching only"""
    
    dens3r_model_ckpt: str = "None"

sfm_store = store(group="sparse_recon")
sfm_store(SfMConfig, name="base")
sfm_store(
    SfMConfig,
    sfm_mode="sfm",
    matching_mode="sequential",
    n_matching_neighbors=10,
    loop_detection_frequency=1,
    loop_detection_num_images=15,
    name="sequential",
)
trig_store = store(group="triangulation")
trig_store(
    SfMConfig,
    sfm_mode="triangulate",
    matching_mode="sequential",
    use_cache=True,
    extractor="loftr",
    matcher="loftr",
    resize_max_area=1920000,
    name="sequential",
)
trig_store(
    SfMConfig,
    sfm_mode="triangulate",
    matching_mode="sequential",
    use_cache=True,
    extractor="loftr",
    matcher="loftr",
    resize_max_area=720000,
    name="sequential_low-res",
)
trig_store(
    SfMConfig,
    sfm_mode="triangulate",
    reference_sfm_reldir="sfm_from_idr",
    matching_mode="pairs_from_poses",
    use_cache=False,
    extractor="loftr",
    matcher="loftr",
    resize_max_area=720000,
    name="pairs_from_poses_low-res",
)
store.add_to_hydra_store()


SparseReconTask = make_dataclass(
    "SparseReconTask",
    [
        "job_type",
        "job_id",
        "inst_id",
        "futures",
        "args",
        "inst_root",
        "output_dir",
        "sfm_dir",
        "manhattan_sfm_dir",
        "image_dir",
        "pairs_path",
        "features_path",
        "matches_path",
        "image_list",
        "ref_sfm_dir",
    ],
)


TerminateTask = make_dataclass(
    "TerminateTask",
    [
        ("job_type", str, "exit"),
    ],
)


@ray.remote(num_cpus=1)  # pyright: ignore
class SparseReconActor:
    def __init__(self, task_queue, return_queue, share_intrinsics=True, make_plot=True, delete_sfm_dir=True):
        self.task_queue = task_queue
        self.return_queue = return_queue
        self.share_intrinsics = share_intrinsics
        self.make_plot = make_plot
        self.delete_sfm_dir = delete_sfm_dir

    def run(self):
        task_inst_id = [None]
        try:
            self._run(task_inst_id)
        except Exception as err:
            self.return_queue.put_nowait((task_inst_id[0], False, f"Unknown error: {err}"))
            raise err

    def _run(self, task_inst_id):
        while True:
            task = self.task_queue.get()  # SparseReconTask
            task_inst_id[0] = task.inst_id

            if task.job_type == "exit":
                return

            if task.futures is not None:
                logger.info("Waiting for feature extraction task done.")
                _ = ray.get(task.futures)  # wait untill precending tasks done

            if task.job_type == "sparse_recon":
                sparse_recon_succeed = self.sparse_recon(task)
                if not sparse_recon_succeed:
                    self.return_queue.put_nowait((task.inst_id, False, "sparse reconstruction failed!"))
                    continue
                self.manhattan_alignment(task)
            elif task.job_type == "sparse_recon_postprocess":
                self.manhattan_alignment(task)
            elif task.job_type == "triangulation":
                triangulation_succeed = self.triangulate(task)
                if not triangulation_succeed:
                    self.return_queue.put_nowait((task.inst_id, False, "triangulation failed!"))
                    continue
                self.manhattan_alignment(task)
            else:
                raise ValueError(f"Unknown job: {task.job_type}")
            self.return_queue.put_nowait((task.inst_id, True))
            logger.info(f"[job_id:{task.job_id} | inst_id:{task.inst_id}] sparse reconstruction saved!")

    def sparse_recon(self, task):
        rec = reconstruction.main(
            task.sfm_dir,
            task.image_dir,
            task.pairs_path,
            task.features_path,
            task.matches_path,
            camera_mode=pycolmap.CameraMode.SINGLE if self.share_intrinsics else pycolmap.CameraMode.AUTO,
            image_list=task.image_list,
        )
        if rec is not None:
            if self.make_plot:
                plot_reconstruction(rec, task.output_dir, show_axes=True)
            return True
        else:
            logger.warning(f"[{task.inst_id}] Failed to reconstruct any model!")
            return False

    def triangulate(self, task):
        rec = triangulation.main(
            task.sfm_dir,
            task.ref_sfm_dir,
            task.image_dir,
            task.pairs_path,
            task.features_path,
            task.matches_path,
            skip_geometric_verification=False,
            min_match_score=None,
        )
        if rec is not None:
            if self.make_plot:
                plot_reconstruction(rec, task.output_dir, show_axes=True)
            return True
        else:
            logger.warning(f"[{task.inst_id}] Failed to triangulate any model!")
            return False

    def manhattan_alignment(self, task):
        # if (task.manhattan_sfm_dir / "points3D.bin").exists():
        #     logger.info(f"Instance {task.inst_root.stem} already manhattan-aligned!")
        #     return
        rec = manhattan_alignment(task.image_dir, task.sfm_dir, task.manhattan_sfm_dir)
        if self.make_plot:
            plot_reconstruction(rec, task.output_dir, show_axes=True, suffix="manhattan")
        if self.delete_sfm_dir:
            shutil.rmtree(task.sfm_dir)
            logger.info(f"sfm directory deleted! ({str(task.sfm_dir)})")


# TODO: merge fields into args
FeatureTask = make_dataclass(
    "FeatureTask",
    [
        "job_type",
        "job_id",
        "inst_id",
        "args",
        "inst_root",
        "image_dir",
        "sfm_dir_name",
        "image_list",
        "features_path",
        "matches_path",
        "pairs_path",
        "sparse_recon_task",
    ],
)


@ray.remote(num_cpus=1, num_gpus=1)  # pyright: ignore
def extract_and_match(task):  # FeatureTask
    feature_conf = extract_features.confs[task.args.extractor]
    matcher_conf = match_features.confs[task.args.matcher]
    extract_features.main(
        feature_conf, task.image_dir, image_list=task.image_list, feature_path=task.features_path
    )  # TODO: set smaller dataloader num_workers if args.n_feature_workers > 1
    match_features.main(matcher_conf, task.pairs_path, features=task.features_path, matches=task.matches_path)
    return


@ray.remote(num_cpus=1)  # pyright: ignore
class FeatureActor:
    def __init__(self, feature_queue: Queue, recon_queue: Queue):
        self.feature_queue = feature_queue
        self.recon_queue = recon_queue

    def run(self):
        while True:
            task = self.feature_queue.get()
            if task.job_type == "exit":
                return

            obj_refs = self._run_task(task)
            if obj_refs is not None and len(obj_refs) > 0:
                _ = ray.get(obj_refs)
            logger.info(f"[job_id:{task.job_id} | inst_id:{task.inst_id}] features & matches saved!")
            self.recon_queue.put_nowait(task.sparse_recon_task)

    def _run_task(self, task):
        if task.job_type == "extract_and_match":
            if task.args.matcher != "loftr":
                obj_refs = self._extract_and_match(task)
            else:
                obj_refs = self._run_loftr(task)
        else:
            raise ValueError(f"Unknown job: {task.job_type}")
        return obj_refs

    def _extract_and_match(self, task):  # FeatureTask
        logger.info(f"Matching [{task.inst_id}] w/ ({task.args.extractor}, {task.args.matcher})")
        if task.args.n_feature_workers > 1:
            logger.warning("n_feature_workers > 1 is not supported for HLoc internal matchers!")
        obj_ref = extract_and_match.remote(task)
        return [obj_ref]

    def _run_loftr(self, task):  # FeatureTask
        # TODO: support batch_size > 1
        loftr_args = loftr_match_features.get_default_args()
        loftr_args.root_dir = task.inst_root
        loftr_args.hloc_dirname = task.sfm_dir_name
        loftr_args.image_dirname = task.image_dir.relative_to(task.inst_root)  # task.image_dir.stem
        loftr_args.pairs_fn = task.pairs_path.name
        loftr_args.features_fn = task.features_path.name
        loftr_args.matches_fn = task.matches_path.name
        loftr_args.force_rerun = task.args.force_rerun
        loftr_args.max_area = task.args.resize_max_area
        loftr_args.n_workers = task.args.n_feature_workers
        obj_ref = loftr_match_features.process_instance(loftr_args, task.image_list, block=False)
        logger.info(f"Matching [{task.inst_id}] w/ LoFTR...")

        return obj_ref


def read_write_cache(
    cache_path: Path,
    current_sfm_dir: Path,
    sfm_mode: Literal["sfm", "triangulate", "sfm_then_triangulate"],
    ref_sfm_dir: Optional[Path] = None,
    use_cache: bool = False,
):
    if not cache_path.exists():
        cache = {}
    else:
        with open(cache_path, "r") as f:
            cache = json.load(f)

    ref_sfm_dir = ref_sfm_dir  # read from cache
    if sfm_mode == "sfm":
        assert not use_cache
        cache["sfm_dir_prev"] = ""
        cache["sfm_dir"] = str(current_sfm_dir)
    else:
        if use_cache:
            assert cache.get("sfm_dir", "") != "" and cache["sfm_dir_prev"] == "", "cache is not valid!"
            ref_sfm_dir = Path(cache["sfm_dir"])
            logger.info(f"Using cached sfm_dire as renference: {ref_sfm_dir}")
        else:
            assert ref_sfm_dir is not None
        cache["sfm_dir_prev"] = str(ref_sfm_dir)
        cache["sfm_dir"] = str(current_sfm_dir)

    with open(cache_path, "w") as f:
        cache = json.dump(cache, f)
    logger.info(f"Cache updated: {cache_path}")
    return ref_sfm_dir


def evenly_sample_images(images, n_samples):
    """evenly sample n_samples from images and keep the start & last elements."""
    n_total = len(images)
    if n_samples is None or n_samples >= n_total:
        return images

    idx = np.round(np.linspace(0, n_total - 1, n_samples)).astype(int)
    images = [images[i] for i in idx]
    logger.info(f"Images subsampled: {len(images)} / {n_total}")
    return images


def parse_images_from_reference_model(sfm_ref_dir):
    rec = pycolmap.Reconstruction(sfm_ref_dir)

    img_names = [rec.images[img_id].name for img_id in rec.reg_image_ids()]
    return img_names


def plot_reconstruction(rec, outputs_dir, show_axes=False, suffix=None):
    fig = viz_3d.init_figure(show_axes=show_axes)
    viz_3d.plot_reconstruction(
        fig, rec, points_color="rgba(255,0,0,0.5)", points_name="vis_sfm_points", cameras_name="vis_sfm_cameras"
    )
    suffix = "" if suffix is None else f"_{suffix}"

    fig_path = outputs_dir / f"vis_sfm{suffix}"
    viz_3d.save_fig(fig, fig_path.parent, fig_path.stem, mode="both")
    logger.info(f"Reconstruction plot saved: {fig_path}")


def manhattan_alignment(image_dir, sfm_dir, output_dir):
    output_dir.mkdir(exist_ok=True)
    cmd = [
        "colmap",
        "model_orientation_aligner",
        "--image_path",
        str(image_dir),
        "--input_path",
        str(sfm_dir),
        "--output_path",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    logger.info(f"manhattan-aligned sfm model saved: {str(output_dir)}")
    rec = pycolmap.Reconstruction(output_dir)
    return rec


def reconstruct_instance(
    args, inst_rel_dir, feature_queue, sparse_recon_task_queue, sparse_recon_return_queue, task_id=None
):
    inst_root = args.data_root / inst_rel_dir
    images = inst_root / args.image_dirname
    cache_path = inst_root / args.cache_name

    _out_prefix = args.sfm_mode
    if args.extractor == "superpoint_aachen" and args.matcher == "superglue":
        sfm_dir_name = f"{_out_prefix}_spp-spg"
    elif args.extractor == "sift":  # NOTE: the HLoc sift impl is slow
        sfm_dir_name = f"{_out_prefix}_sift"
        assert args.matcher == "NN-ratio"
    elif args.extractor == "loftr":
        sfm_dir_name = f"{_out_prefix}_loftr-{args.resize_max_area}"
    else:
        raise NotImplementedError()

    _match_str = args.matching_mode.replace("_", "-")
    if args.matching_mode in ["sequential", "pairs_from_poses"]:
        _match_str = f"{_match_str}_np-{args.n_matching_neighbors}"
    sfm_dir_name = f"{sfm_dir_name}_{_match_str}"
    if args.n_images is not None:
        sfm_dir_name = f"{sfm_dir_name}_nimgs-{args.n_images}"

    outputs = inst_root / sfm_dir_name
    outputs.mkdir(exist_ok=True)
    logger.info(f"Reconstruction directory: {sfm_dir_name}")
    sfm_dir = outputs / "sfm"
    manhattan_sfm_dir = outputs / "manhattan"
    ref_sfm_dir = None if args.sfm_mode == "sfm" else inst_root / args.reference_sfm_reldir
    ref_sfm_dir = read_write_cache(
        cache_path, manhattan_sfm_dir, args.sfm_mode, ref_sfm_dir=ref_sfm_dir, use_cache=args.use_cache
    )

    features = outputs / f"features_{args.extractor}.h5"
    matches = outputs / f"matches_{args.matcher}.h5"

    if args.sfm_mode != "triangulate":
        references = natsort.natsorted([p.relative_to(images).as_posix() for p in images.iterdir()], key=str)
        references = evenly_sample_images(references, args.n_images)
        logger.info(f"[{inst_rel_dir}] #mapping_images: {len(references)}")
    else:  # triangulate
        references = parse_images_from_reference_model(ref_sfm_dir)
        if args.n_images is not None:
            logger.warning("Donot support image sampling in triangulation mode.")
        logger.info(f"[{inst_rel_dir}] Found {len(references)} images in the reference sfm model")

    # TODO: support sfm_mode == "sfm_then_triangulate"
    sparse_recon_task = SparseReconTask(
        job_type="sparse_recon" if args.sfm_mode != "triangulate" else "triangulation",
        job_id=task_id,
        inst_id=inst_rel_dir,
        args=args,
        inst_root=inst_root,
        output_dir=outputs,
        sfm_dir=sfm_dir,
        manhattan_sfm_dir=manhattan_sfm_dir,
        image_dir=images,
        pairs_path=None,
        features_path=features,
        matches_path=matches,
        image_list=references,
        futures=None,
        ref_sfm_dir=ref_sfm_dir,
    )
    feature_task = FeatureTask(
        job_type="extract_and_match",
        job_id=task_id,
        inst_id=inst_rel_dir,
        args=args,
        sfm_dir_name=sfm_dir_name,
        inst_root=inst_root,
        image_dir=images,
        image_list=references,
        features_path=features,
        matches_path=matches,
        pairs_path=None,
        sparse_recon_task=None,
    )

    if not args.force_rerun and (manhattan_sfm_dir / "points3D.bin").exists():
        logger.info(f"Instance {inst_rel_dir} already reconstructed & aligned, Skipped!")
        sparse_recon_return_queue.put_nowait((inst_rel_dir, True))
    elif (
        not args.force_rerun and (sfm_dir / "points3D.bin").exists()
    ):  # sparse-recon already done, only run postprocessing
        logger.info(f"Instance {inst_rel_dir} already reconstructed, run alignment only!")
        sparse_recon_task.job_type = "sparse_recon_postprocess"
        sparse_recon_task_queue.put_nowait(sparse_recon_task)
        feature_ref = None
    else:  # extract-features & (sparse-recon / triangulation)
        # generate matches
        if args.matching_mode == "sequential":
            sfm_pairs = outputs / f"pairs-sfm_seq-loop-det_n{args.n_matching_neighbors}.txt"
            if ref_sfm_dir is not None and (ref_sfm_dir.parent / sfm_pairs.name).exists():
                shutil.copy(ref_sfm_dir.parent / sfm_pairs.name, sfm_pairs)
                logger.info("Using existing pairs in the reference model!")
            else:
                pairs_from_sequential.main(
                    sfm_pairs,
                    image_root_dir=images,
                    image_list=references,
                    overlap=int(np.ceil(args.n_matching_neighbors / 2)),
                    quadratic_overlap=True,
                    loop=True,
                    loop_detection=True,
                    loop_detection_frequency=args.loop_detection_frequency,
                    loop_detection_num_images=args.loop_detection_num_images,
                )
        elif args.matching_mode == "pairs_from_poses":
            sfm_pairs = outputs / f"pairs-sfm_existing-poses_n{args.n_matching_neighbors}.txt"
            # TODO: set rotation_threshold (default=30°)
            pairs_from_poses.main(ref_sfm_dir, sfm_pairs, args.n_matching_neighbors)
        elif args.matching_mode == "exhaustive":
            sfm_pairs = outputs / "pairs-sfm_exhaustive.txt"
            pairs_from_exhaustive.main(sfm_pairs, image_list=references)
        elif args.matching_mode == "vocab_tree":
            raise NotImplementedError
        elif args.matching_mode == "database":
            raise NotImplementedError  # TODO: use existing pairs in the database, w/ at least k matches after verification
        else:
            raise ValueError
        feature_task.pairs_path = sfm_pairs
        sparse_recon_task.pairs_path = sfm_pairs
        feature_task.sparse_recon_task = sparse_recon_task
        feature_queue.put_nowait(feature_task)


def reconstruct_instance_wrapper(cli_args, *args, **kwargs):
    """Handle sfm-then-triangulate mode"""
    if cli_args.sfm_mode != "sfm_then_triangulate":
        reconstruct_instance(copy.deepcopy(cli_args), *args, **kwargs)
    else:
        # sparse-recon
        sfm_args = copy.deepcopy(cli_args)
        sfm_args.sfm_mode = "sfm"
        reconstruct_instance(sfm_args, *args, **{**kwargs, "use_cache": False})

        # semi-dense triangulation
        trig_args = copy.deepcopy(cli_args)
        trig_args.sfm_mode = "triangulate"
        trig_args.n_images = None
        trig_args.extractor, trig_args.matcher = "loftr", "loftr"

        reconstruct_instance(trig_args, *args, **{**kwargs, "use_cache": True})


def main(sfm: SfMConfig):
    # TODO: refactor to use a class
    args = sfm

    ray.init(
        num_cpus=args.n_feature_workers * 3 + args.n_recon_workers,
        num_gpus=args.n_feature_workers,
        # object_store_memory=int(1e11),  # FIXME
        include_dashboard=False,
        # runtime_env={"env_vars": {"PYTHONPATH": f"{os.environ.get('PYTHONPATH', '')}:LOFTR_PATH"}},
    )
    feature_task_queue = Queue()
    sparse_recon_task_queue = Queue()
    sparse_recon_return_queue = Queue()

    # initialize reconstruction actors
    recon_actors = [
        SparseReconActor.remote(
            sparse_recon_task_queue,
            sparse_recon_return_queue,
            share_intrinsics=args.share_intrinsics,
            make_plot=args.make_plot,
            delete_sfm_dir=args.delete_sfm_dir,
        )
        for _ in range(args.n_recon_workers)
    ]
    for actor in recon_actors:
        actor.run.remote()

    # initialize feature extraction actors
    n_feature_actors = 1 if args.matcher == "loftr" else args.n_feature_workers
    feature_actors = [FeatureActor.remote(feature_task_queue, sparse_recon_task_queue) for _ in range(n_feature_actors)]
    for actor in feature_actors:
        actor.run.remote()

    n_inst = None
    assert np.array([i is None for i in [args.inst_rel_dir, args.inst_list_path]]).sum() == 1
    _queue_args = (feature_task_queue, sparse_recon_task_queue, sparse_recon_return_queue)
    if args.inst_rel_dir is not None:
        n_inst = 1
        reconstruct_instance(copy.deepcopy(args), args.inst_rel_dir, *_queue_args, task_id=0)
    elif args.inst_list_path is not None:
        with open(args.inst_list_path, "r") as f:
            inst_rel_dirs = f.read().strip().split()
        n_inst = len(inst_rel_dirs)

        for task_id, inst_rel_dir in enumerate(inst_rel_dirs):
            reconstruct_instance(copy.deepcopy(args), inst_rel_dir, *_queue_args, task_id=task_id)
    else:
        raise ValueError('Either "inst_rel_dir" or "inst_list_path" should be provided!')

    # wait untill all tasks are done
    failed_tasks = []
    while True:
        state = sparse_recon_return_queue.get()
        n_inst -= 1
        if not state[1]:
            failed_tasks.append((state[0], state[2]))  # (inst_rel_dir, msg)
        if n_inst == 0:
            break
    if len(failed_tasks) > 0:
        _msgs = [f"[{inst_rel_dir}] {msg}" for inst_rel_dir, msg in failed_tasks]
        logger.warning(f"{len(failed_tasks)} tasks failed!\n" "\n".join(_msgs))
    ray.shutdown()
    
def dens3r_recon(sfm: SfMConfig):
    import os
    import sys
    sys.path.append(str(Path(os.path.abspath(__file__)).resolve().parents[5]))
    from infer.dens3r_recon import Dens3RReconstruction
    from infer.infer_normal_pts3d import infer_imgs
    
    inst_root = sfm.data_root / sfm.inst_rel_dir
    images_path = inst_root / sfm.image_dirname
    references = natsort.natsorted([p.relative_to(images_path).as_posix() for p in images_path.iterdir()], key=str)
    references = evenly_sample_images(references, sfm.n_images)
    logger.info(f"[{sfm.inst_rel_dir}] #mapping_images: {len(references)}")
    
    # copy references images to a new directory
    ref_image_dir = os.path.join(str(inst_root), "images_ref")
    os.makedirs(ref_image_dir, exist_ok=True)
    for imagename in references:
        shutil.copy(os.path.join(str(images_path), imagename), 
                    os.path.join(str(ref_image_dir), imagename))
    
    sfm_dir_name = f"dens3r_{sfm.n_images}_{sfm.matching_mode}"
    # run dens3r recon
    Dens3RReconstruction(ckpt_path = sfm.dens3r_model_ckpt, 
                         img_dir = ref_image_dir, 
                         sfm_dir = os.path.join(str(inst_root), sfm_dir_name),
                         matching_mode = sfm.matching_mode, 
                         window_size = 30, 
                         size = 512, 
                         triangulate_from_pose = None,
                         use_mapper = "glomap",
                         camera_model = "SIMPLE_RADIAL",
                         conf_thr = 1.001, 
                         pixel_tol = 5, 
                         subsample = 8)
    
    save_normal_dir = f"dens3r_normal"
    infer_imgs(model_path = sfm.dens3r_model_ckpt,
               img_dir = ref_image_dir,
               save_dir = os.path.join(str(inst_root), save_normal_dir),
               size=1024, 
               device="cuda",
               seq_len=None, 
               pair_mode="self", 
               save_depth=False, 
               save_normal=True, 
               save_pcd=False)
    
    cache_path = inst_root / sfm.cache_name
    manhattan_sfm_dir = Path(os.path.join(str(inst_root), sfm_dir_name)) / "manhattan"
    ref_sfm_dir = read_write_cache(
        cache_path, manhattan_sfm_dir, sfm.sfm_mode, ref_sfm_dir=sfm_dir_name, use_cache=sfm.use_cache
    )
    import torch
    torch.cuda.empty_cache()
    
    return sfm_dir_name

    
def postprocess_dens3r(sfm: SfMConfig, sfm_dir_name: str):
    import os
    ws_dir = os.path.join(str(sfm.data_root), sfm.inst_rel_dir, sfm_dir_name)
    if os.path.exists(os.path.join(ws_dir, "hloc_sfm/models/0/images.bin")):
        recon_colmap_path = os.path.join(ws_dir, "hloc_sfm/models/0")
    elif os.path.exists(os.path.join(ws_dir, "hloc_sfm/0/images.bin")):
        recon_colmap_path = os.path.join(ws_dir, "hloc_sfm/0")
    else:
        recon_colmap_path = os.path.join(ws_dir, "hloc_sfm")
    rec = manhattan_alignment(Path(os.path.join(str(sfm.data_root), sfm.inst_rel_dir, sfm.image_dirname)), 
                              Path(recon_colmap_path), 
                              Path(os.path.join(ws_dir, "manhattan")))
    if sfm.make_plot:
        plot_reconstruction(rec, Path(ws_dir), show_axes=True, suffix="manhattan")
    if sfm.delete_sfm_dir:
        shutil.rmtree(recon_colmap_path)
        logger.info(f"sfm directory deleted! ({recon_colmap_path})")


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="config", node=SfMConfig)
    hydra.main(config_name="config", version_base=None)(main)()

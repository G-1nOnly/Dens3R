import argparse
import copy
import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import List, Optional

import cv2
import hydra
import numpy as np
import pycolmap
import ray
import torch
from hydra.core.config_store import ConfigStore
from hydra_zen import store
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


@dataclass
class DINOExtractionConfig:
    task: str = "feature_extraction"
    """options: 'pca', 'feature_extraction'"""  # TODO: use enum
    facet: List[str] = field(default_factory=lambda: ["key"])
    """options: 'key', 'value', 'query', 'token'"""
    n_workers: int = 1
    """number of workers for parallely feature extraction of multiple instances"""
    data_root: Path = Path("data")
    image_dirname: Path = Path("images")
    inst_rel_dir: Optional[str] = None
    """relative directory to the specific instance"""
    subsets_relpath: Optional[str] = None
    """path of the file listing all inst_rel_dir to be processed"""
    max_area: int = 640 * 480  # takes about 24GB VRAM
    registered_only: bool = True
    """only extract features for registered images"""
    force_extract: bool = False
    fp16: bool = True
    """store features in fp16"""
    sfm_relpath: Optional[str] = None
    cache_name: str = ".cache.json"
    use_cache: bool = True
    """read the sfm_relpath from cache"""

    def __post_init__(self):
        self.save_fn_stem = "dino_feats"
        self.facet = " ".join(self.facet)  # type: ignore

        if self.registered_only:
            if self.task == "pca":
                raise NotImplementedError()
            assert self.sfm_relpath is not None or self.use_cache

        if self.fp16:
            logger.info("Saving features w/ fp16")


feature_store = store(group="dino_feature")
feature_store(DINOExtractionConfig, name="base")
feature_store(DINOExtractionConfig, max_area=480 * 360, name="low-res")
store.add_to_hydra_store()


def resize_max_area(h, w, max_area):
    area = h * w
    scale = np.sqrt(max_area / area)
    new_h, new_w = h * scale, w * scale
    new_h, new_w = map(lambda x: int(np.floor(x)), [new_h, new_w])
    return new_h, new_w


def get_sfm_dir(args, inst_rel_dir):
    assert np.sum([args.sfm_relpath is not None, args.use_cache]) == 1
    sfm_dir = None
    if args.use_cache:
        with open(args.data_root / inst_rel_dir / args.cache_name, "r") as f:
            cache = json.load(f)
        sfm_dir = Path(cache["sfm_dir"])
    else:
        sfm_dir = args.data_root / inst_rel_dir / args.sfm_relpath
        if not sfm_dir.exists():
            raise FileNotFoundError(f"sfm_dir not found! ({sfm_dir})")
    return sfm_dir


def _save_feat_dirname_to_cache(args, inst_rel_dir, feat_dirname):
    if args.task != "feature_extraction":
        return

    inst_dir = args.data_root / inst_rel_dir
    with open(inst_dir / args.cache_name, "r") as f:
        cache = json.load(f)
    cache["feat_dirname"] = feat_dirname
    with open(inst_dir / args.cache_name, "w") as f:
        json.dump(cache, f)


def extract_single(args, inst_rel_dir):
    task = args.task
    max_area = args.max_area
    cat = args.data_root.stem
    image_dir = args.data_root / inst_rel_dir / args.image_dirname
    if not args.registered_only:
        image_names = [p.stem for p in image_dir.iterdir()]
        n_images = len(image_names)
    else:  # only extract features for registered images
        sfm_dir = get_sfm_dir(args, inst_rel_dir)
        if not sfm_dir.exists():
            # TODO: add to a pending list, and wait until {sfm_dir} is created. (reconstruction done)
            raise FileNotFoundError(f"sfm_dir not found! ({sfm_dir})")
        rec = pycolmap.Reconstruction(sfm_dir)
        image_paths = [image_dir / rec.images[img_id].name for img_id in rec.reg_image_ids()]
        image_names = [p.stem for p in image_paths]
        n_images = len(image_names)
        logger.info(f"Extracting features for registered {n_images} images by SfM: {sfm_dir}")
        image_paths_str = "\n".join(map(str, image_paths))
        tmp_filename = f"{inst_rel_dir.replace('/', '-')}_reg-img-paths_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        tmp_image_list_path = args.data_root / inst_rel_dir / tmp_filename
        with open(tmp_image_list_path, "w") as f:
            f.write(image_paths_str)

    output_dirname = f"{args.save_fn_stem}_{max_area}" if task == "feature_extraction" else f"dino_pca_{max_area}"
    output_path = args.data_root / inst_rel_dir / output_dirname
    _save_feat_dirname_to_cache(args, inst_rel_dir, output_dirname)

    _extracted = (
        output_path.exists()
        and n_images <= len(list(output_path.iterdir()))
        and set(image_names).issubset(set([p.stem for p in output_path.iterdir()]))
    )

    if _extracted:
        if not args.force_extract:
            logger.info(f"Instance {cat}/{inst_rel_dir} already processed!")
            return
        else:
            shutil.rmtree(output_path)
            logger.info(f"Instance {cat}/{inst_rel_dir} deleted & rerun")

    # compute dynamic load_size
    image_path = next(iter(image_dir.iterdir()))
    h, w, _ = cv2.imread(str(image_path)).shape
    new_h, new_w = resize_max_area(h, w, max_area)
    load_size = min(new_h, new_w)

    if task == "feature_extraction":
        cmd = f"python {Path(__file__).parent}/vit_extractor.py \
                --image_path {str(image_dir) if not args.registered_only else str(tmp_image_list_path)} \
                --output_path {str(output_path)} \
                --load_size {load_size} \
                --facet {args.facet} \
                --unflatten --save_npz"
        cmd = cmd.split()
        if args.fp16:
            cmd.append("--fp16")
    else:
        raise ValueError()

    subprocess.run(cmd)


@ray.remote(num_cpus=1, num_gpus=1)  # pyright: ignore
def ray_wrapper(*args, worker=None, task_id=None, **kwargs):
    assert not (worker is None or task_id is None)
    torch.cuda.empty_cache()
    # gpu_id = ray.get_gpu_ids()[0]
    # inst_rel_dir = kwargs['inst_rel_dir']

    try:
        worker(*args, **kwargs)
    except Exception as ex:
        return False, ex
    return True, None


def main(dino_extraction: DictConfig):
    args: DINOExtractionConfig = OmegaConf.to_object(dino_extraction)  # run __post_init__ for validation

    assert np.array([i is not None for i in [args.inst_rel_dir, args.subsets_relpath]], dtype=int).sum() == 1
    if args.inst_rel_dir is not None:
        if args.n_workers > 1:
            raise ValueError("n_workers > 1 is not supported for single instance extraction")
        extract_single(args, args.inst_rel_dir)
    elif args.subsets_relpath is not None:
        subsets_path = args.data_root / args.subsets_relpath
        ray.init(num_cpus=1 * args.n_workers, num_gpus=args.n_workers)
        with open(subsets_path, "r") as f:
            subsets = json.load(f)
        cat_insts = list(chain(*subsets.values()))
        n_inst = len(cat_insts)
        logger.info(f"{n_inst} sequences to be extracted!")
        obj_refs = []
        for task_id, cat_inst in tqdm(enumerate(cat_insts), "Creating ray tasks"):
            cat, inst_rel_dir = cat_inst.split("/")
            cat_dir = args.data_root / cat
            _args = copy.copy(args)
            _args.data_root = cat_dir
            obj_refs.append(
                ray_wrapper.remote(_args, inst_rel_dir=inst_rel_dir, worker=extract_single, task_id=task_id)
            )
        task_rets = ray.get(obj_refs)
        failed_errs = [str(ex) for success, ex in task_rets if not success]
        failed_errs_str = "\n".join(failed_errs)
        logger.warning(f"#failed_tasks={len(failed_errs)}:\n{failed_errs_str}")


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="config", node=DINOExtractionConfig)
    hydra.main(config_name="config", version_base=None)(main)()

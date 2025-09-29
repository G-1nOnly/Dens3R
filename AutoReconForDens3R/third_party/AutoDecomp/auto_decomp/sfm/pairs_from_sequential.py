import argparse
import collections.abc as collections
import copy
import shutil
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import natsort
import numpy as np
import torch
from hloc import extract_features, pairs_from_retrieval
from hloc.utils.io import list_h5_names
from hloc.utils.parsers import parse_image_list
from loguru import logger

from auto_decomp.utils.cli import str2bool


def read_image_pairs(path):
    with open(path, "r") as f:
        pairs = [p.split() for p in f.read().rstrip("\n").split("\n")]
    return pairs


def pairs_from_file(image_list: List[Path], pairs_path: Path) -> List[Tuple[Path, Path]]:
    converted_pairs = []  # absolute paths
    name2path = {p.name: p for p in image_list}
    pairs = read_image_pairs(pairs_path)
    for pair in pairs:
        img0_name, img1_name = map(lambda x: Path(x).name, pair)
        if img0_name in name2path and img1_name in name2path:
            converted_pairs.append((name2path[img0_name], name2path[img1_name]))
        else:
            logger.warning(f"Pairs does not exist in image_list: {img0_name}-{img1_name}")
    return converted_pairs


def build_loop_detection_pairs(
    *,
    pairs_path: Path,
    image_list: List[Path],  # absolute image paths
    num_images: int,  # number of images to be matched for loop-detection, should be significantly larger than sequential pairs
    frequency: int,  # run loop-detection for every frequency images
    method: Literal["colmap", "retrieval"] = "retrieval",
    retrieval_method: Literal["netvlad"] = "netvlad",
) -> List[Tuple[Path, Path]]:
    if method == "colmap":
        # ref: https://github.com/colmap/pycolmap/blob/743a4ac305183f96d2a4cfce7c7f6418b31b8598/pipeline/match_features.cc#L177
        raise NotImplementedError("loop detection with colmap is not implemented yet.")

    if pairs_path.exists():
        return pairs_from_file(image_list, pairs_path)

    retrival_conf = extract_features.confs[retrieval_method]
    temp_dir = pairs_path.parent / "temp_retrival"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    image_dir = image_list[0].parent
    retrival_feat_path = extract_features.main(
        retrival_conf, image_list[0].parent, temp_dir, image_list=[p.relative_to(image_dir) for p in image_list]
    )
    torch.cuda.empty_cache()

    loop_check_images = [Path(img_path).name for img_path in image_list[::frequency]]
    pairs_from_retrieval.main(
        retrival_feat_path, pairs_path, num_matched=min(num_images, len(image_list)), query_list=loop_check_images
    )
    shutil.rmtree(temp_dir)
    return pairs_from_file(image_list, pairs_path)


def main(
    output: Path,
    image_root_dir: Path,
    image_list: Optional[Union[Path, List[str]]] = None,  # relative to image_root_dir
    features: Optional[Path] = None,
    overlap: int = 5,
    quadratic_overlap: bool = True,
    loop: bool = True,  # treat the image_list as a sequential loop
    loop_detection: bool = True,
    loop_detection_method: Literal["colmap", "retrieval"] = "retrieval",
    loop_detection_frequency: int = 10,
    loop_detection_num_images: int = 20,
):
    if image_list is not None:
        if isinstance(image_list, (str, Path)):  # parse from file
            names_q = parse_image_list(image_list)
        elif isinstance(image_list, collections.Iterable):
            names_q = natsort.natsorted(list(map(str, image_list)))
        else:
            raise ValueError(f"Unknown type for image list: {image_list}")
    elif features is not None:
        names_q = list_h5_names(features)
        names_q = list(sorted(list(names_q)))
    else:
        raise ValueError("Provide either a list of images or a feature file.")
    image_list = copy.deepcopy(names_q)

    # build sequential pairs
    idxs = np.arange(0, len(names_q))
    dists = np.abs(idxs[None, :] - idxs[:, None]).astype(np.float32)  # (N, N)
    if loop:
        dists_loop = np.abs(idxs[None, :] - (idxs[:, None] - len(names_q))).astype(np.float32)  # (N, N)
        dists = np.minimum(dists, dists_loop)
    log2_dists = np.log2(dists, out=np.zeros_like(dists), where=(dists != 0))
    mask_overlap = (dists > 0) & (dists <= overlap)
    mask_quadratic_overlap = (
        (log2_dists > 0) & (log2_dists <= overlap) & (log2_dists == log2_dists.astype(int))
        if quadratic_overlap
        else np.zeros_like(mask_overlap, dtype=bool)
    )
    idxs0, idxs1 = np.nonzero(mask_overlap | mask_quadratic_overlap)

    names_ref = [names_q[i] for i in idxs0]
    names_q = [names_q[j] for j in idxs1]

    pairs = []
    for n0, n1 in zip(names_ref, names_q):
        pairs.append(tuple(sorted((n0, n1))))
    pairs = set(pairs)
    n_sequential_pairs, n_loop_det_pairs = len(pairs), 0

    # build loop-detection pairs
    if loop_detection:
        loop_det_pairs_path = output.parent / f"pairs_loop-det_netvlad-{loop_detection_num_images}.txt"
        image_list_absolute = [image_root_dir / n for n in image_list]
        loop_det_pairs = build_loop_detection_pairs(
            pairs_path=loop_det_pairs_path,
            image_list=image_list_absolute,
            num_images=loop_detection_num_images,
            frequency=loop_detection_frequency,
            method=loop_detection_method,
            retrieval_method="netvlad",
        )
        loop_det_pairs = set(
            [tuple(sorted(map(lambda p: str(p.relative_to(image_root_dir)), (p0, p1)))) for p0, p1 in loop_det_pairs]
        )
        pairs |= loop_det_pairs
        n_loop_det_pairs = len(loop_det_pairs)
    pairs = list(pairs)

    logger.info(f"Found {len(pairs)} pairs (n_seq={n_sequential_pairs} | n_loop={n_loop_det_pairs}).")
    with open(output, "w") as f:
        f.write("\n".join(" ".join([i, j]) for i, j in pairs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path, help="path to save the pairs")
    parser.add_argument("--image_root_dir", required=True, type=Path)
    parser.add_argument("--image_list", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--overlap", type=int, help="number of overlapping image pairs")
    parser.add_argument(
        "--quadratic_overlap", action="store_true", help="whether to match images against their quadratic neighbors"
    )
    parser.add_argument(
        "--loop",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="treat the image sequence as a loop when considering overlapping pairs",
    )
    parser.add_argument("--loop_detection", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--loop_detection_method", choices=["colmap", "retrieval"], default="retrieval")
    parser.add_argument(
        "--loop_detection_frequency", type=int, default=5, help="perform loop detection every frequency images."
    )
    parser.add_argument(
        "--loop_detection_num_images",
        type=int,
        default=20,
        help="The number of images to retrieve in loop detection. This number should be "
        "significantly bigger than the sequential matching overlap.",
    )

    args = parser.parse_args()
    main(**args.__dict__)

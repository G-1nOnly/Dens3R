import argparse
import sys
from pathlib import Path

import cv2
import h5py
import imagesize
import numpy as np
import ray
import torch
from hloc.match_features import find_unique_new_pairs
from hloc.utils.parsers import names_to_pair
from loguru import logger
from PIL import Image
from ray.experimental import tqdm_ray
from ray.util.queue import Queue
from torch.utils.data import DataLoader, Dataset, DistributedSampler

LOFTR_PATH = str(Path(__file__).parents[2])

from loftr.config.default import get_cfg_defaults
from loftr.loftr import LoFTR, default_cfg
from loftr.utils.misc import lower_config


def list_h5_names(path):
    names = []
    with h5py.File(str(path), "r") as fd:

        def visit_fn(_, obj):
            if isinstance(obj, h5py.Dataset):
                names.append(obj.parent.name.strip("/"))

        fd.visititems(visit_fn)
    return list(set(names))


class MatchDataset(Dataset):
    def __init__(
        self,
        root_dir,
        hloc_dirname="hloc_outputs",
        image_dirname="images",
        pairs_fn="pairs-sfm.txt",
        resize_max_area=1920000,
        df=8,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.hloc_dir = self.root_dir / hloc_dirname
        self.image_dir = self.root_dir / image_dirname
        self.image_pairs = self._parse_image_pairs(self.hloc_dir / pairs_fn)
        self.resize_max_area = resize_max_area
        self.df = df

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        name0, name1 = self.image_pairs[idx]
        img0_path, img1_path = map(lambda x: self.image_dir / x, self.image_pairs[idx])
        img0, img1 = map(lambda x: cv2.imread(str(x), cv2.IMREAD_GRAYSCALE), [img0_path, img1_path])
        img0, img1 = map(lambda x: self._preprocess_image(x), [img0, img1])
        return {"image0": img0, "image1": img1, "pair_name": names_to_pair(name0, name1)}

    def _parse_image_pairs(self, pairs_path):
        with open(pairs_path, "r") as f:
            pairs = f.read().rstrip("\n").split("\n")
            pairs = [pair.split() for pair in pairs]
        # TODO: filter duplicate pairs
        return pairs

    def _preprocess_image(self, img: np.array):
        h, w = img.shape
        new_h, new_w = compute_resize_max_area(h, w, max_area=self.resize_max_area, df=self.df)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        img = torch.from_numpy(img).float()[None] / 255  # (1, h, w)
        return img


@ray.remote(num_cpus=1)
class MatchSaver(object):
    def __init__(self, save_dir, save_fn, n_pairs, queue):
        self.save_path = Path(save_dir) / save_fn
        self.queue = queue
        if self.save_path.exists():
            self.save_path.unlink()
        self.f_matches = h5py.File(self.save_path, "a")
        self.max_n_pairs = n_pairs

    def run(self):
        _cntr = 0
        required_keys = set(["matches0", "matching_scores0"])
        pair_name_cache = set()
        pbar = tqdm_ray.tqdm(desc="Extracting LoFTR matches:", total=self.max_n_pairs)
        while _cntr < self.max_n_pairs:
            data = self.queue.get(block=True)
            pair_name = data.pop("pair_name")
            if pair_name in pair_name_cache:  # DistributedSampler might duplicate data
                continue
            pair_name_cache.add(pair_name)
            assert set(data.keys()) == required_keys
            if pair_name in self.f_matches:
                del self.f_matches[pair_name]
            grp = self.f_matches.create_group(pair_name)
            grp.create_dataset("matches0", data=data["matches0"])
            grp.create_dataset("matching_scores0", data=data["matching_scores0"])
            pbar.update()
            _cntr += 1
        self.f_matches.close()


@ray.remote(num_cpus=1, num_gpus=1)
class MatchExtrator(object):
    def __init__(self, args, rank, world_size, queue, coarse_hws_cached):
        dataset = build_dataset(args)
        self.loader = build_dataloader(args, dataset, rank, world_size)
        self.queue = queue
        self.coarse_hws_cached = coarse_hws_cached
        # init matcher
        if args.config is not None:
            config = get_cfg_defaults()
            config.merge_from_file(args.config)
            config = lower_config(config)["loftr"]
        else:
            config = default_cfg
        matcher = LoFTR(config=config)
        matcher.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["state_dict"])
        self.matcher = matcher.eval().cuda()

    def run(self):
        for data in self.loader:
            data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
            pair_name = data["pair_name"][0]
            raw_coarse_matches, coarse_hws0, coarse_hws1 = extract_coarse_matches(self.matcher, data)
            name0, name1 = pair_name.split("/")
            assert coarse_hws0 == self.coarse_hws_cached[name0] and coarse_hws1 == self.coarse_hws_cached[name1]
            hloc_coarse_matches = build_coarse_matches_hloc(raw_coarse_matches, coarse_hws0)
            hloc_coarse_matches["pair_name"] = pair_name
            self.queue.put_nowait(hloc_coarse_matches)
            del data, raw_coarse_matches
            torch.cuda.empty_cache()


def extract_coarse_matches(matcher, data):
    """
    Args:
        data (Dict): {'image0': Tensor, 'image1': Tensor}
    Returns:
        matches (Dict): {
            'mconfs' (np.ndarray): (n, ),
            'mids0' (np.ndarray): (n, ),
            'mids1' (np.ndarray): (n, )
        }
    """
    matcher(data, coarse_only=True)
    assert (data["b_ids"] == 0).all(), "BatchSize > 1 not supported yet!"
    coarse_hws0, coarse_hws1 = tuple(data["hw0_c"]), tuple(data["hw1_c"])
    mids0, mids1 = data["m_iids"], data["m_jids"]
    mconfs = data["mconf"]  # (n, )
    mconfs, mids0, mids1 = map(lambda x: x.cpu().numpy(), [mconfs, mids0, mids1])
    return {"mconfs": mconfs, "mids0": mids0, "mids1": mids1}, coarse_hws0, coarse_hws1


def build_coarse_matches_hloc(result, coarse_hws0):
    # mconfs, c_xys0, c_xys1 = map(result.get, ['mconfs', 'c_xys0', 'c_xys1'])
    mconfs, mids0, mids1 = map(result.get, ["mconfs", "mids0", "mids1"])

    # transform to hloc matches format
    n_kpts = coarse_hws0[0] * coarse_hws0[1]
    dtype = np.int16 if n_kpts < 32767 else np.int32
    matches0 = np.full((n_kpts,), -1, dtype=dtype)
    matching_scores0 = np.zeros((n_kpts,), dtype=dtype)
    matches0[mids0] = mids1.astype(dtype)
    matching_scores0[mids0] = mconfs.astype(np.float16)

    matches = {"matches0": matches0, "matching_scores0": matching_scores0}
    return matches


def check_consistent_image_sizes(image_dir, image_list):
    sizes = np.array([imagesize.get(image_dir / p) for p in image_list])
    all_equal = (sizes[0, 0] == sizes[:, 0]).all() and (sizes[0, 1] == sizes[:, 1]).all()
    return all_equal


def compute_resize_max_area(h, w, max_area=1920000, df=8):
    if h * w <= max_area:
        tgt_shape = (int(h // df * df), int(w // df * df))
    else:
        ratio = np.sqrt(max_area / h / w)
        h *= ratio
        w *= ratio
        tgt_shape = (int(h // df * df), int(w // df * df))
    return tgt_shape


def _build_pseudo_features(image_path, max_area=1920000, df=8, coarse_scale=8):
    img = Image.open(image_path)
    h, w = img.height, img.width
    new_h, new_w = compute_resize_max_area(h, w, max_area=max_area, df=df)
    coarse_h, coarse_w = new_h // coarse_scale, new_w // coarse_scale
    Xs, Ys = np.mgrid[:coarse_w, :coarse_h]
    coarse_XYs = np.stack([Xs, Ys], axis=-1)  # (c_w, c_h, 2)
    coarse2ori_scale = np.array([w / coarse_w, h / coarse_h])
    ori_XYs = coarse_XYs * coarse2ori_scale + coarse2ori_scale / 2 - 0.5  # hloc save keypoints in dpix
    ori_XYs = ori_XYs.transpose(1, 0, 2).reshape(-1, 2)  # ((c_h, c_w), 2)
    return ori_XYs, (coarse_h, coarse_w), coarse2ori_scale


def build_pseudo_features(
    image_dir,
    image_list,
    save_dir,
    save_fn="features.h5",
    max_area=1920000,
    df=8,
    coarse_scale=8,
    detection_noise=1.0,
    as_half=True,
):
    save_path = Path(save_dir) / save_fn
    consistent_image_size = check_consistent_image_sizes(image_dir, image_list)
    coarse_hws = dict()  # caches for coarse dense grid sizes (for later sanity check)

    if not consistent_image_size:
        raise NotImplementedError()
    else:  # generate once and reuse
        pseudo_kpts, (kpts_h, kpts_w), kpts_scale = _build_pseudo_features(
            image_dir / image_list[0], max_area=max_area, df=df, coarse_scale=coarse_scale
        )  # (n, 2)
        uncertainty = detection_noise * kpts_scale.mean()
        features = {
            "keypoints": pseudo_kpts if not as_half else pseudo_kpts.astype(np.float16),
            "scors": np.empty((0,), dtype=np.float16),
            "descriptors": np.empty((0,), dtype=np.float16),
        }
        with h5py.File(str(save_path), "a") as fd:
            for name in image_list:
                try:
                    if name in fd:
                        del fd[name]
                    grp = fd.create_group(name)
                    for k, v in features.items():
                        grp.create_dataset(k, data=v)
                    grp["keypoints"].attrs["uncertainty"] = uncertainty
                except OSError as error:
                    if "No space left on device" in error.args[0]:
                        logger.error(
                            "Out of disk space: storing features on disk can take "
                            "significant space, did you enable the as_half flag?"
                        )
                        del grp, fd[name]
                    raise error

        coarse_hws.update({name: (kpts_h, kpts_w) for name in image_list})
    logger.info(f"Finished generaing pseudo features. ({len(image_list)} in total.)")
    return save_path, coarse_hws


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=Path, default="/raid/yuang/Data/CO3D_V2/chair/380_45217_90441")
    parser.add_argument("--hloc_dirname", type=str, default="hloc_loftr_outputs")
    parser.add_argument("--image_dirname", type=str, default="images")
    parser.add_argument("--pairs_fn", type=str, default="pairs-sfm.txt")
    parser.add_argument("--features_fn", type=str, default="features.h5")
    parser.add_argument("--matches_fn", type=str, default="matches.h5")
    parser.add_argument("--force_rerun", action="store_true")

    parser.add_argument("--max_area", type=int, default=1920000)
    parser.add_argument("--df", type=int, default=8)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--n_loader_workers", type=int, default=1)
    # LoFTR-related fields
    parser.add_argument("--config", type=Path, default=f"{LOFTR_PATH}/configs/loftr/outdoor/buggy_pos_enc/loftr_ds.py")
    parser.add_argument("--checkpoint", type=Path, default=f"{LOFTR_PATH}/weights/outdoor_ds.ckpt")
    return parser


def get_default_args():
    return get_parser().parse_args([])


def build_dataset(args):
    dset = MatchDataset(
        args.root_dir,
        hloc_dirname=args.hloc_dirname,
        image_dirname=args.image_dirname,
        pairs_fn=args.pairs_fn,
        resize_max_area=args.max_area,
        df=args.df,
    )
    return dset


def build_dataloader(args, dataset, rank, world_size):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=0, drop_last=False)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.n_loader_workers, pin_memory=True, sampler=sampler
    )
    return loader


def check_existence(args, image_list):
    hloc_dir = args.root_dir / args.hloc_dirname
    features_path = hloc_dir / args.features_fn
    matches_path = hloc_dir / args.matches_fn
    if not (features_path.exists() and matches_path.exists()) or args.force_rerun:
        return False

    try:
        with h5py.File(features_path, "r") as ff, h5py.File(matches_path) as mf:
            pass
    except OSError as err:
        logger.warning(
            "At least one of the features & the matches file is corrupted, rerun matching...\n"
            f"{err}\n"
            f" ({str(features_path)},{str(matches_path)})"
        )
        features_path.unlink()
        matches_path.unlink()
        return False

    features_all_done, matches_all_done = False, False
    if features_path.exists():
        skip_feature_names = list_h5_names(features_path)
        if set(image_list).issubset(set(skip_feature_names)):
            features_all_done = True
    if matches_path.exists():
        pairs_path = hloc_dir / args.pairs_fn
        with open(pairs_path, "r") as f:
            pairs = f.read().rstrip("\n").split("\n")
        pairs = [tuple(pair.split()) for pair in pairs]
        pairs = find_unique_new_pairs(pairs_all=pairs, match_path=matches_path)
        if len(pairs) == 0:
            matches_all_done = True

    # TODO: return unmatched pairs and only match these pairs
    existed = features_all_done and matches_all_done
    if existed:
        logger.info(f"Features & matches already extracted: {str(hloc_dir)}, skipped!")
    return existed


def process_instance(args, image_list, block=True):  # process a single instance
    if check_existence(args, image_list):
        logger.info(f'Instance ({"/".join(args.root_dir.parts[-2:])}) already processed, skipping...')
        return

    args.root_stem = args.root_dir.stem
    coarse_scale = 8  # TODO: extract coarse_scale from LoFTR config
    feature_path, coarse_hws_cached = build_pseudo_features(
        args.root_dir / args.image_dirname,
        image_list,
        args.root_dir / args.hloc_dirname,
        args.features_fn,
        max_area=args.max_area,
        df=args.df,
        coarse_scale=coarse_scale,
        as_half=True,
    )

    if not ray.is_initialized():
        ray.init(num_cpus=args.n_workers + 1, num_gpus=args.n_workers, ignore_reinit_error=False)
    queue = Queue()
    _dataset = build_dataset(args)
    n_pairs = len(_dataset)
    match_saver = MatchSaver.remote(args.root_dir / args.hloc_dirname, args.matches_fn, n_pairs, queue)
    match_saver_ref = match_saver.run.remote()
    # -- init & run MatchActors
    match_extractors = [
        MatchExtrator.remote(args, rank, args.n_workers, queue, coarse_hws_cached) for rank in range(args.n_workers)
    ]
    match_extractor_refs = [a.run.remote() for a in match_extractors]
    obj_refs = [*match_extractor_refs, match_saver_ref]

    if block:
        ray.get(obj_refs)
        # ray.shutdown()
        logger.info(f"[{args.root_stem}] matches for {n_pairs} pairs extracted!")
    else:
        logger.info(f"[{args.root_stem}] LoFTRMatchWorker started ({n_pairs} pairs to be matched)")
        return obj_refs


def main():  # Testinng
    parser = get_parser()
    args = parser.parse_args()
    image_dir = args.root_dir / args.image_dirname
    images = [p.relative_to(image_dir).as_posix() for p in image_dir.iterdir()]
    images = images[:8]
    process_instance(args, images, block=True)


if __name__ == "__main__":
    main()

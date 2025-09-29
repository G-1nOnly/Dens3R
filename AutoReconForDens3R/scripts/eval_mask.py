"""Evaluate mask rendeings with IoU"""
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from typing_extensions import assert_type
from tqdm import tqdm
from einops import repeat

import cv2
import tyro
import numpy as np
from rich.console import Console
from rich.pretty import pprint
from nerfstudio.utils import bilateral_solver

CONSOLE = Console(width=120)


# --------------- preprocess utils  -------------- #
def evenly_sample_elems(elems, n_samples):
    """ evenly sample n_samples from elems and keep the start & last elements. """
    n_total = len(elems)
    if n_samples is None or n_samples >= n_total:
        return elems

    idx = np.round(np.linspace(0, n_total - 1, n_samples)).astype(int)
    elems = [elems[i] for i in idx]
    return elems


def read_mask(mask_path):
    gt_mask = cv2.imread(str(mask_path))
    if gt_mask is None:
        raise FileNotFoundError(mask_path)
    if gt_mask.ndim == 3:
        gt_mask = (gt_mask[..., 0] > 0).astype(np.uint8) * 255
    return gt_mask


def resize_mask(gt_mask, pred_mask, max_side_length=None):
    """Resize the mask to the same lower-res size"""
    h_gt, w_gt = gt_mask.shape[:2]
    h_pred, w_pred = pred_mask.shape[:2]
    
    if max_side_length is not None:  # return larger side to this size
        def _compute_size(mask):
            h, w = mask.shape[:2]
            s = max([h, w])
            scale = max_side_length / s
            h, w = int(h * scale), int(w * scale)
            return (w, h)
        
        gt_mask = cv2.resize(gt_mask, _compute_size(gt_mask), interpolation=cv2.INTER_NEAREST)
        pred_mask = cv2.resize(pred_mask, _compute_size(pred_mask), interpolation=cv2.INTER_NEAREST)
        return gt_mask, pred_mask
    
    if h_gt > h_pred:
        gt_mask = cv2.resize(gt_mask, (w_pred, h_pred), interpolation=cv2.INTER_NEAREST)
        CONSOLE.print(f'GT mask resized: {(h_gt, w_gt)} -> {(h_pred, w_pred)}')
    elif h_gt < h_pred:
        pred_mask = cv2.resize(pred_mask, (w_gt, h_gt), interpolation=cv2.INTER_NEAREST)
        CONSOLE.print(f'Pred mask resized: {(h_pred, w_pred)} -> {(h_gt, w_gt)}')
    else:
        assert w_gt == w_pred
    return gt_mask, pred_mask


# --------------- metric utils  -------------- #
def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    https://openaccess.thecvf.com/content/CVPR2021/papers/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.pdf
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1: h + 1, 1: w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def boundary_iou(gt, dt, dilation_ratio=0.02, vis_path=None, img_path=None):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    boundary_iou = intersection / union
    return boundary_iou


def mask_iou(gt, dt, vis_path=None, img_path=None):
    """
    Compute mask iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :return: boundary iou (float)
    """
    intersection = ((gt * dt) > 0).sum()
    union = ((gt + dt) > 0).sum()
    iou = intersection / union
    
    if vis_path is not None:
        assert img_path is not None
        _gt = np.zeros((*gt.shape, 3), dtype=np.uint8)
        _dt = _gt.copy()
        _gt[gt > 0] = [0, 0, 255]  # red
        _dt[dt > 0] = [0, 255, 0]  # green
        # _gt = repeat(gt, 'h w -> h w 3').astype(np.uint8) * 255
        # _dt = repeat(dt, 'h w -> h w 3').astype(np.uint8) * 255
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        img = cv2.addWeighted(img, 0.5, _gt, 0.5, 0)
        img = cv2.addWeighted(img, 0.5, _dt, 0.5, 0)
        cv2.imwrite(str(vis_path), img)
        
    return iou


# --------------- main  -------------- #
@dataclass
class EvalMask:
    pred_mask_dir: Path
    gt_mask_dir: Path
    n_eval_frames: Optional[int] = None
    binarization_threshold: float = 0.5
    bilateral_refine: bool = False
    bilateral_mask_confidence: float = 0.999
    max_side_length: Optional[int] = None  # since tokencut results have a max length of 980
    vis: bool = False
    gt_image_dir: Optional[Path] = None
    
    def main(self) -> None:
        gt_mask_paths = sorted(self.gt_mask_dir.iterdir())
        pred_mask_paths = sorted(self.pred_mask_dir.iterdir())
        
        if self.vis or self.bilateral_refine:
            assert self.gt_image_dir is not None
            self.gt_img_suffix = self.gt_image_dir.iterdir().__next__().suffix
        if self.vis:
            self.vis_dir = self.pred_mask_dir.parent / "eval_vis"
            self.vis_dir.mkdir(exist_ok=True, parents=True)
            CONSOLE.print(f'Mask vis dir: {self.vis_dir}')
        
        gt_filenames = [p.stem for p in gt_mask_paths]
        self.gt_mask_suffix = gt_mask_paths[0].suffix
        self.pred_mask_suffix = pred_mask_paths[0].suffix
        eval_filenames, eval_to_pred_fn_mapper = self.parse_filenames(
            gt_filenames, [p.stem for p in pred_mask_paths]
        )
        self.eval_to_pred_fn_mapper = eval_to_pred_fn_mapper
        
        # 1. use all frames w/ gt; 2. use evenly sampled frames
        if self.n_eval_frames is not None:
            eval_filenames = evenly_sample_elems(eval_filenames, self.n_eval_frames)
        
        # evalute
        all_metrics = {}
        for fn in tqdm(eval_filenames, desc="Evaluate mask metrics"):
            metrics = self.eval_frame(fn)
            all_metrics[fn] = metrics
        
        # gather results
        results = self.gather_results(all_metrics)
    
    def parse_filenames(self, gt_filenames, pred_filenames):
        if set(gt_filenames) <= set(pred_filenames):
            return gt_filenames, {fn: fn for fn in gt_filenames}
        else:  # for DTU, the gt filenames for eval & rendering might be different
            print(f'Not all gt masks have pred: {set(gt_filenames) - set(pred_filenames)}')
            if len(gt_filenames) != len(pred_filenames):
                raise RuntimeError(f'Number of gt & pred masks are different')
            gt_filenames = sorted(gt_filenames)
            pred_filenames = sorted(pred_filenames)
            # assume the filenames only contain numbers (TODO: grab number w/ re)
            assert all([int(gt_fn) == int(pred_fn) for gt_fn, pred_fn in zip(gt_filenames, pred_filenames)])
            gt_to_pred_fn = {gt_fn: pred_fn for gt_fn, pred_fn in zip(gt_filenames, pred_filenames)}
            return gt_filenames, gt_to_pred_fn
    
    def eval_frame(self, fn):
        gt_mask_path = (self.gt_mask_dir / fn).with_suffix(self.gt_mask_suffix)
        pred_fn = self.eval_to_pred_fn_mapper[fn]
        pred_mask_path = (self.pred_mask_dir / pred_fn).with_suffix(self.pred_mask_suffix)
        gt_mask, pred_mask = map(read_mask, [gt_mask_path, pred_mask_path])
        if self.gt_image_dir is not None:
            img_path = (self.gt_image_dir / f'{fn}').with_suffix(self.gt_img_suffix)
        
        if self.bilateral_refine:
            gt_img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            pred_mask = cv2.GaussianBlur(pred_mask, (31, 31), 0)
            pred_mask = ((pred_mask.astype(np.float32) / 255) > 0.5).astype(np.float32)
            pred_mask, _ = bilateral_solver.bilateral_refine(
                gt_img, pred_mask, soft_mask=False, confidence=self.bilateral_mask_confidence
            )
            pred_mask = pred_mask * 255
        
        gt_mask, pred_mask = resize_mask(gt_mask, pred_mask, max_side_length=self.max_side_length)
        gt_mask, pred_mask = map(lambda x: x.astype(float) / 255,
                                 [gt_mask, pred_mask])
        # binarization
        gt_mask, pred_mask = map(lambda x: (x > self.binarization_threshold).astype(np.uint8),
                                 [gt_mask, pred_mask])
        
        # compute metrics
        if self.vis:
            mask_vis_path = self.vis_dir / f'{fn}_mask_vis.png'
            boundary_vis_path = self.vis_dir / f'{fn}_boundary_vis.png'
        else:
            mask_vis_path, boundary_vis_path, img_path = None, None, None
        
        _mask_iou = mask_iou(gt_mask, pred_mask, vis_path=mask_vis_path, img_path=img_path)
        _boundary_iou = boundary_iou(gt_mask, pred_mask, vis_path=boundary_vis_path, img_path=img_path)
        return {'mask_iou': _mask_iou, 'boundary_iou': _boundary_iou}
    
    def gather_results(self, all_metrics):
        inst, cat = self.gt_mask_dir.parent.stem, self.gt_mask_dir.parents[1].stem
        cat_inst = f'{cat}/{inst} (n_frames={self.n_eval_frames})'
        
        frame_names = list(all_metrics.keys())
        all_mask_iou = [m['mask_iou'] for m in all_metrics.values()]
        all_boundary_iou = [m['boundary_iou'] for m in all_metrics.values()]
        mean_mask_iou = np.mean(all_mask_iou)
        mean_boundary_iou = np.mean(all_boundary_iou)
        min_mask_iou_idx, max_mask_iou_idx = np.argmin(all_mask_iou), np.argmax(all_mask_iou)
        min_boundary_iou_idx, max_boundary_iou_idx = np.argmin(all_boundary_iou), np.argmax(all_boundary_iou)
        
        results = {
            'cat_inst': cat_inst,
            'mean_mask_iou': mean_mask_iou,
            'mean_boundary_iou': mean_boundary_iou,
            'min_mask_iou_frame': f'{frame_names[min_mask_iou_idx]}({all_mask_iou[min_mask_iou_idx]:.3f})',
            'max_mask_iou_frame': f'{frame_names[max_mask_iou_idx]}({all_mask_iou[max_mask_iou_idx]:.3f})',
            'min_boundary_iou_frame': f'{frame_names[min_boundary_iou_idx]}({all_boundary_iou[min_boundary_iou_idx]:.3f})',
            'max_boundary_iou_frame': f'{frame_names[max_boundary_iou_idx]}({all_boundary_iou[max_boundary_iou_idx]:.3f})',
        }
        # print(tabulate(results, headers='keys', tablefmt='psql'))
        pprint(results)
        return results


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(tyro.conf.FlagConversionOff[EvalMask]).main()


if __name__ == "__main__":
    entrypoint()

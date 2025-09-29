import os
import cv2
import mat73
import torch
import argparse
import matplotlib
import numpy as np
from scipy import io
from tqdm import tqdm
from PIL import Image

import sys
sys.path.append(os.path.realpath(os.path.dirname(os.path.dirname(__file__))))

from dust3r.inference import inference
from mast3r.model import AsymmetricMASt3R
from dust3r.utils.image import load_images
from infer.infer_normal_pts3d import rescale_to_orig, rotate_recover


def normalized_depth_scale_and_shift(prediction, target, mask):
    """
    More info here: https://arxiv.org/pdf/2206.00665.pdf supplementary section A2 Depth Consistency Loss
    This function computes scale/shift required to normalizes predicted depth map,
    to allow for using normalized depth maps as input from monocular depth estimation networks.
    These networks are trained such that they predict normalized depth maps.

    Solves for scale/shift using a least squares approach with a closed form solution:
    Based on:
    https://github.com/autonomousvision/monosdf/blob/d9619e948bf3d85c6adec1a643f679e2e8e84d4b/code/model/loss.py#L7
    Args:
        prediction: predicted depth map
        target: ground truth depth map
        mask: mask of valid pixels
    Returns:
        scale and shift for depth prediction
    """
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    scale = torch.zeros_like(b_0)
    shift = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    scale[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    shift[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return scale, shift


class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg


class RunningAverageDict:
    """A dictionary of running averages."""
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if new_dict is None:
            return

        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        if self._dict is None:
            return None
        return {key: value.get_value() for key, value in self._dict.items()}


def compute_errors(gt, pred):
    """Compute metrics for 'pred' compared to 'gt'

    Args:
        gt (numpy.ndarray): Ground truth values
        pred (numpy.ndarray): Predicted values

        gt.shape should be equal to pred.shape

    Returns:
        dict: Dictionary containing the following metrics:
            'a1': Delta1 accuracy: Fraction of pixels that are within a scale factor of 1.25
            'a2': Delta2 accuracy: Fraction of pixels that are within a scale factor of 1.25^2
            'a3': Delta3 accuracy: Fraction of pixels that are within a scale factor of 1.25^3
            'abs_rel': Absolute relative error
            'rmse': Root mean squared error
            'log_10': Absolute log10 error
            'sq_rel': Squared relative error
            'rmse_log': Root mean squared error on the log scale
            'silog': Scale invariant log error
    """
    pred = pred.clip(min=1e-4)
    gt = gt.clip(min=1e-4)
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)


def compute_metrics(gt_depth, pred, save_vis=None, interpolate=False,
                    min_depth_eval=1e-3, max_depth_eval=20.0, align_dpeth=True):
    """Compute metrics of predicted depth maps. Applies cropping and masking as necessary or specified via arguments. Refer to compute_errors for more details on metrics.
    """
    if gt_depth.shape[-2:] != pred.shape[-2:] and interpolate:
        pred = torch.nn.functional.interpolate(
            pred, gt_depth.shape[-2:], mode='bilinear', align_corners=True)
    
    pred[pred < min_depth_eval] = min_depth_eval
    pred[pred > max_depth_eval] = max_depth_eval
    pred[np.isinf(pred)] = max_depth_eval
    pred[np.isnan(pred)] = min_depth_eval

    valid_mask = np.logical_and(
        gt_depth > min_depth_eval, gt_depth < max_depth_eval)
    cal_mask = valid_mask
    
    if align_dpeth:
        
        scale, shift = normalized_depth_scale_and_shift(prediction=torch.from_numpy(pred).unsqueeze(0), 
                                                        target=torch.from_numpy(gt_depth).unsqueeze(0), 
                                                        mask=torch.from_numpy(cal_mask).unsqueeze(0))
        pred = pred * scale.numpy() + shift.numpy()
        
    if save_vis is not None:
        concat_vis = colorize(np.concatenate([gt_depth, pred], axis=1), cmap="Spectral")
        Image.fromarray(concat_vis).save(save_vis)
        
    pred[pred < min_depth_eval] = min_depth_eval
    pred[pred > max_depth_eval] = max_depth_eval
    pred[np.isinf(pred)] = max_depth_eval
    pred[np.isnan(pred)] = min_depth_eval
        
    return compute_errors(gt_depth[valid_mask], pred[valid_mask]), pred


def colorize(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, 
             background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    value[invalid_mask] = np.nan
    # cmapper = matplotlib.cm.get_cmap(cmap)
    cmapper = matplotlib.colormaps[cmap]
    if value_transform:
        value = value_transform(value)
    value = cmapper(value, bytes=True)  # (nxmx4)

    img = value[...]
    img[invalid_mask] = background_color

    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img


def load_img_pairs(dataset_path=None, img_path_list=None, size=1024, pair_mode="self"):
    if img_path_list is None:
        assert dataset_path is not None
        filelist = os.listdir(dataset_path)
        img_list = [item for item in filelist if item.endswith(".png")]
        img_list.sort()
        img_path_list = [os.path.join(dataset_path, item) for item in filelist]
    images = load_images(img_path_list, size=size, square_ok=True)
    pairs = []
    for i in range(len(images)):
        if pair_mode == "self":
            pairs.append((images[i], images[i]))
        elif pair_mode == "near":
            if i == len(images) - 1:
                pairs.append((images[i], images[i-1]))
            else:
                pairs.append((images[i], images[i+1]))
    return pairs


def get_pred(model_path, img_list, save_path, device="cuda", size=1024, datatype="nyu"):
    model = AsymmetricMASt3R.from_pretrained(model_path).to(device)
    
    pairs = load_img_pairs(img_path_list=img_list, size=size)
    for i in tqdm(range(len(pairs)), desc=f"Infering result"):
        pair_slice = pairs[i:i+1]
        output = inference(pair_slice, model, device, batch_size=1, verbose=False)
        output = rotate_recover(output)
        view1, pred1 = output['view1'], output['pred1']
        # view2, pred2 = output['view2'], output['pred2']
        pred_pts3d = pred1["pts3d"][0, ...].detach().cpu().numpy()
        
        size_info = pair_slice[0][0]["size_info"]
        imgname = pair_slice[0][0]["model_id"]
        pred_pts3d = rescale_to_orig(pred_pts3d, size_info)
        pred_depth = pred_pts3d[..., -1]
        
        if datatype == "nyu":
            np.save(os.path.join(save_path, f"{imgname.split('.')[0]}_depth_pred.npy"), pred_depth)
        else:
            raise NotImplementedError


def eval_nyudv2(data_dir, save_dir, model_path, align_depth=True):
    # extract depth GT
    depth_gt_path = os.path.join(save_dir, "depth_gt")
    if not os.path.exists(depth_gt_path):
        os.makedirs(depth_gt_path)
        rgb_mat_path = os.path.join(data_dir, "data/images_uint8.mat")
        split_path = os.path.join(data_dir, "data/splits.mat")
        depth_mat_path = os.path.join(data_dir, "gt/depths.mat")
        mask_mat_path = os.path.join(data_dir, "gt/masks.mat")
        rgb_data = io.loadmat(rgb_mat_path)['images']
        test_idx = io.loadmat(split_path)['testNdxs']
        mask_data = mat73.loadmat(mask_mat_path)["masks"]
        depth_data = mat73.loadmat(depth_mat_path)["depths"]
        for idx in tqdm(test_idx, "extracting imgs"):
            idx = idx[0]-1
            rgb = rgb_data[..., idx]
            mask = mask_data[..., idx]
            valid_mask = (mask > 0)
            depth_gt = depth_data[..., idx]
        
            cut_w, cut_h = 12, 9
            rgb = rgb[cut_h:-cut_h, cut_w:-cut_w, ...]
            valid_mask = valid_mask[cut_h:-cut_h, cut_w:-cut_w, ...]
            depth_gt = depth_gt[cut_h:-cut_h, cut_w:-cut_w, ...]
            
            depth_gt[~valid_mask] = np.zeros_like(depth_gt)[~valid_mask]
            cv2.imwrite(os.path.join(depth_gt_path, f"{idx:05d}_rgb.png"), rgb[..., [2,1,0]])
            np.save(os.path.join(depth_gt_path, f"{idx:05d}_depth_gt.npy"), depth_gt)
    
    rgb_img_list = os.listdir(depth_gt_path)
    rgb_img_paths = [os.path.join(depth_gt_path, item) for item in rgb_img_list if item.endswith("_rgb.png")]
    pred_save_path = os.path.join(save_dir, "depth_pred")
    if not os.path.exists(pred_save_path):
        os.makedirs(pred_save_path)
        get_pred(model_path=model_path,
                 img_list=rgb_img_paths,
                 save_path=pred_save_path,
                 size=640, datatype="nyu")
    
    depth_gt_paths = [item.replace("_rgb.png", "_depth_gt.npy") for item in rgb_img_paths]
    depth_pred_paths = [f"{os.path.basename(item).split('_')[0]}_rgb_depth_pred.npy" for item in depth_gt_paths]
    depth_pred_paths = [os.path.join(pred_save_path, item) for item in depth_pred_paths]
    
    metrics = RunningAverageDict()
    log_info = []
    for i in range(len(depth_gt_paths)):
        depth_gt = np.load(depth_gt_paths[i])
        depth_pred = np.load(depth_pred_paths[i])
        errors, _ = compute_metrics(depth_gt, depth_pred, 
                                    save_vis=os.path.join(pred_save_path, depth_pred_paths[i].replace(".npy", "_vis_debug.png")), 
                                    align_dpeth=align_depth)
        metrics.update(errors)
        print(f"{rgb_img_paths[i]} abs_rel:{errors['abs_rel']} a1:{errors['a1']}")
        log_info.append(f"{rgb_img_paths[i]} abs_rel:{errors['abs_rel']} a1:{errors['a1']}")
    def r(m): return round(m, 3)
    metrics = {k: r(v) for k, v in metrics.get_value().items()}
    print(metrics)
    for key in metrics.keys():
        log_info.append(f'{key} = {metrics[key]}')
    with open(os.path.join(pred_save_path, "log.txt"), "w") as f:
        f.write("\n".join(log_info))


def parse_args():
    parser = argparse.ArgumentParser(description="Depth evaluation script")
    
    parser.add_argument("--model_path", required=True, help="path to model checkpoint")
    parser.add_argument("--data_dir", required=True, help="path to dataset folder")
    parser.add_argument("--save_dir", required=True, help="path to save folder")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    eval_nyudv2(args.data_dir, args.save_dir, args.model_path)


if __name__ == '__main__':
    main()
import os
import cv2
import torch
import mat73
import shutil
import argparse
import numpy as np
from scipy import io
from tqdm import tqdm

import sys
sys.path.append(os.path.realpath(os.path.dirname(os.path.dirname(__file__))))

from dust3r.inference import inference
from mast3r.model import AsymmetricMASt3R
from dust3r.utils.image import load_images, normal2color
from infer.infer_normal_pts3d import rescale_to_orig, rotate_recover


def dot(x, y):
    """dot product (along the last dim).

    Args:
        x (Union[Tensor, ndarray]): x, [..., C]
        y (Union[Tensor, ndarray]): y, [..., C]

    Returns:
        Union[Tensor, ndarray]: x dot y, [..., 1]
    """
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)

def length(x, eps=1e-20):
    """length of an array (along the last dim).

    Args:
        x (Union[Tensor, ndarray]): x, [..., C]
        eps (float, optional): eps. Defaults to 1e-20.

    Returns:
        Union[Tensor, ndarray]: length, [..., 1]
    """
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))

def safe_normalize(x, eps=1e-20):
    """normalize an array (along the last dim).
    Args:
        x (Union[Tensor, ndarray]): x, [..., C]
        eps (float, optional): eps. Defaults to 1e-20.
    Returns:
        Union[Tensor, ndarray]: normalized x, [..., C]
    """
    return x / length(x, eps)

def compute_metric(gt_path, pred_path, save_vis_dir=None, data_type=None):
    if gt_path.endswith(".png"):
        normal_gt = cv2.imread(gt_path)
        normal_gt = normal_gt / 255 * 2 - 1
    elif gt_path.endswith(".npy"):
        normal_gt = np.load(gt_path)

    if data_type == "DIODE":
        normal_gt[..., -1] *= -1
        normal_gt[..., 0] *= -1
    elif data_type == "IBIMS-1":
        normal_gt[..., 0] *= -1
    elif data_type == "NYUv2":
        normal_gt[..., -1] *= -1
    elif data_type == "Sintel":
        normal_gt = np.transpose(normal_gt[0, ...], (1, 2, 0))
        normal_gt[..., -1] *= -1
    elif data_type == "Scannet":
        normal_gt = normal_gt[..., [2,1,0]]
        normal_gt[..., 0] *= -1
        normal_gt[..., 1] *= -1
    else:
        raise NotImplementedError
    
    if pred_path.endswith(".png"):
        normal_pred_png = cv2.imread(pred_path)[..., [2,1,0]]
        normal_pred_png = cv2.resize(normal_pred_png, 
                                     (normal_gt.shape[1], normal_gt.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
        normal_pred = np.zeros_like(normal_gt)
        normal_pred[..., 0] = normal_pred_png[..., 0] / 255 * 2 - 1
        normal_pred[..., 1] = normal_pred_png[..., 1] / 255 * 2 - 1
        normal_pred[..., 2] = (normal_pred_png[..., 2] - 128) / 127 * -1
        normal_pred = safe_normalize(normal_pred)
    elif pred_path.endswith(".npy"):
        normal_pred = np.load(pred_path)
        
    if save_vis_dir is not None:
        os.makedirs(save_vis_dir, exist_ok=True)
        debug_save_path = os.path.join(save_vis_dir, os.path.basename(pred_path).replace(".npy", "_debug.png"))
        debug_img = np.hstack((normal2color(normal_gt), normal2color(normal_pred)))
        cv2.imwrite(debug_save_path, debug_img[..., [2,1,0]])

    normal_gt_norm = np.linalg.norm(normal_gt, axis=-1)
    fg_mask = (normal_gt_norm > 0.5) & (normal_gt_norm < 1.5)
    rmse = np.sqrt(((normal_pred - normal_gt) ** 2)[fg_mask].sum(axis=-1).mean())
    dot_product = (normal_pred * normal_gt).sum(axis=-1)
    dot_product = np.clip(dot_product, -1, 1)
    dot_product = dot_product[fg_mask]
    angle = np.arccos(dot_product) / np.pi * 180

    return angle, rmse

def compute_matric_list(img_list, normal_gt_list, gt_dir, save_path, data_type):
    angle_arr, rmse_arr = np.array([]), []
    log_info = []
    for i in range(len(img_list)):
        if data_type == "DIODE":
            gt_path = normal_gt_list[i]
            pred_path = os.path.join(save_path, normal_gt_list[i].split("/")[-2], normal_gt_list[i].split("/")[-1])
        elif data_type == "Scannet":
            gt_path = os.path.join(gt_dir, normal_gt_list[i])
            pred_path = os.path.join(save_path, normal_gt_list[i].replace("-normal.png", "-color_normal.npy"))
        else:
            gt_path = os.path.join(gt_dir, normal_gt_list[i])
            pred_path = os.path.join(save_path, normal_gt_list[i])
        
        angle, rmse = compute_metric(gt_path, pred_path, save_vis_dir=save_path, data_type=data_type)
        angle_arr = np.concatenate((angle_arr, angle))
        rmse_arr.append(rmse)
        print(f"{img_list[i]}: angle={np.mean(angle)}, rmse={rmse}")
        log_info.append(f"{img_list[i]}: angle={np.mean(angle)}, rmse={rmse}")
    pct_gt_5 = 100.0 * np.sum(angle_arr < 11.25, axis=0) / angle_arr.shape[0]
    pct_gt_10 = 100.0 * np.sum(angle_arr < 22.5, axis=0) / angle_arr.shape[0]
    pct_gt_30 = 100.0 * np.sum(angle_arr < 30, axis=0) / angle_arr.shape[0]
    print(f"Percentage of angle less than 11.25: {pct_gt_5:.3f}, 22.5: {pct_gt_10:.3f}, 30: {pct_gt_30:.3f}")
    print(f"Average angle: {np.mean(angle_arr):.3f}, Average rmse: {np.mean(rmse_arr):.3f}")
    print(f"Medium angle: {np.median(angle_arr):.3f}, Medium rmse: {np.median(rmse_arr):.3f}")
    log_info.append(f"Percentage of angle less than 11.25: {pct_gt_5:.3f}, 22.5: {pct_gt_10:.3f}, 30: {pct_gt_30:.3f}")
    log_info.append(f"Average angle: {np.mean(angle_arr):.3f}, Average rmse: {np.mean(rmse_arr):.3f}")
    log_info.append(f"Medium angle: {np.median(angle_arr):.3f}, Medium rmse: {np.median(rmse_arr):.3f}")
    with open(os.path.join(save_path, "log.txt"), "w") as f:
        f.write("\n".join(log_info))

def load_img_pairs(dataset_path=None, img_path_list=None, size=1024, pair_mode="self"):
    if img_path_list is None:
        assert dataset_path is not None
        filelist = os.listdir(dataset_path)
        img_list = [item for item in filelist if item.endswith(".png")]
        img_list.sort()
        img_path_list = [os.path.join(dataset_path, item) for item in filelist]
    images = load_images(img_path_list, size=size, square_ok=True)
    pairs = []
    for i in range(len(img_path_list)):
        if pair_mode == "self":
            pairs.append((images[i], images[i]))
        elif pair_mode == "near":
            if i == len(images) - 1:
                pairs.append((images[i], images[i-1]))
            else:
                pairs.append((images[i], images[i+1]))
        else:
            raise NotImplementedError
    return pairs

def get_pred(model_path, dataset_path, save_path, device="cuda", size=1024):
    model = AsymmetricMASt3R.from_pretrained(model_path).to(device)
    
    if isinstance(dataset_path, list):
        pairs = load_img_pairs(img_path_list=dataset_path, size=size)
    else:
        assert isinstance(dataset_path, str)
        pairs = load_img_pairs(dataset_path=dataset_path, size=size)
    
    for i in tqdm(range(len(pairs)), desc=f"Infering result"):
        pair_slice = pairs[i:i+1]
        output = inference(pair_slice, model, device, batch_size=1, verbose=False)
        output = rotate_recover(output)
        view1, pred1 = output['view1'], output['pred1']
        pred_normal = pred1["normal"][0, ...].detach().cpu().numpy()
        
        # left hand to right hand
        pred_normal[..., 0] *= -1
        
        size_info = pair_slice[0][0]["size_info"]
        imgname = pair_slice[0][0]["model_id"]
        pred_normal = rescale_to_orig(pred_normal, size_info)
        pred_normal_viz = normal2color(pred_normal)
        
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, imgname.replace(".png", "_normal.npy")), pred_normal)
        cv2.imwrite(os.path.join(save_path, imgname.replace(".png", "_pred_normal_viz.png")), pred_normal_viz[..., [2,1,0]])


def eval_nyudv2(data_dir, save_dir, model_path):
    '''
    :param data_dir: the root dir of NYUDv2
    :param save_dir: the root dir to save the result
    :param model_path: the path of the model(.pth file)
    '''
    rgb_mat_path = os.path.join(data_dir, "data/images_uint8.mat")
    split_path = os.path.join(data_dir, "data/splits.mat")
    normal_mat_path = os.path.join(data_dir, "gt/norm_gt_l.mat")
    mask_mat_path = os.path.join(data_dir, "gt/masks.mat")
    
    # first extract rgb and normal gt from mat files
    rgb_save_dir = os.path.join(save_dir, "rgb")
    normal_gt_save_dir = os.path.join(save_dir, "normal_gt")
    if not os.path.exists(rgb_save_dir) or not os.path.exists(normal_gt_save_dir):
        shutil.rmtree(rgb_save_dir)
        shutil.rmtree(normal_gt_save_dir)
        os.makedirs(rgb_save_dir)
        os.makedirs(normal_gt_save_dir)
        
        rgb_data = io.loadmat(rgb_mat_path)['images']
        test_idx = io.loadmat(split_path)['testNdxs']
        mask_data = mat73.loadmat(mask_mat_path)["masks"]
        normal_data = mat73.loadmat(normal_mat_path)["norm_gt_l"]
        
        for idx in tqdm(test_idx, "extracting imgs"):
            idx = idx[0]-1
            rgb = rgb_data[..., idx]
            mask = mask_data[..., idx]
            valid_mask = (mask > 0)[..., None].repeat(3, -1)
            normal_gt = normal_data[..., idx]
            
            cut_w, cut_h = 12, 9
            rgb = rgb[cut_h:-cut_h, cut_w:-cut_w, ...]
            valid_mask = valid_mask[cut_h:-cut_h, cut_w:-cut_w, ...]
            normal_gt = normal_gt[cut_h:-cut_h, cut_w:-cut_w, ...]
            
            normal_gt[~valid_mask] = np.zeros_like(normal_gt)[~valid_mask]
            cv2.imwrite(os.path.join(rgb_save_dir, f"{idx:05d}_rgb.png"), rgb[..., [2,1,0]])
            np.save(os.path.join(normal_gt_save_dir, f"{idx:05d}_rgb_normal.npy"), normal_gt)
            normal_map_color = normal2color(normal_gt)
            cv2.imwrite(os.path.join(normal_gt_save_dir, f"{idx:05d}_normal.png"), normal_map_color[..., [2,1,0]])
    
    # predict normal of every rgb image
    # since image size on NYUDv2 is 640, so we set size=640
    gt_dir = os.path.join(save_dir, "normal_gt")
    save_path = os.path.join(save_dir, "pred_normal")
    if not os.path.exists(save_path):
        get_pred(model_path=model_path,
                dataset_path=rgb_save_dir,
                save_path=save_path,
                size=640)
    
    # compute metric
    filelist = os.listdir(os.path.join(save_dir, "rgb"))
    img_list = sorted([item for item in filelist if item.endswith(".png")])
    normal_gt_list = [item.replace(".png", "_normal.npy") for item in img_list]
    
    compute_matric_list(img_list, normal_gt_list, gt_dir, save_path, data_type="NYUv2")


def parse_args():
    parser = argparse.ArgumentParser(description="Normal evaluation script")
    
    parser.add_argument("--model_path", required=True, help="path to model checkpoint")
    parser.add_argument("--data_dir", required=True, help="path to dataset folder")
    parser.add_argument("--save_dir", required=True, help="path to save folder")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    eval_nyudv2(args.data_dir, args.save_dir, args.model_path)


if __name__ == "__main__":
    main()
# Copyright (C) 2025-present Alibaba Group. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# infer images' normal and depth
# --------------------------------------------------------

import os
import cv2
import glob
import torch
import argparse
import numpy as np
import open3d as o3d
from tqdm import tqdm
from scipy.spatial.transform import Rotation

import sys
sys.path.append(os.path.realpath(os.path.dirname(os.path.dirname(__file__))))
from dust3r.inference import inference
from mast3r.model import AsymmetricMASt3R
from dust3r.utils.image import load_images, colorize, normal2color, inverse_normalize


def make_pairs(img_list, mode="sequence", seq_len=5):
    '''
    mg_list: list of obj, sorted
    seq_len=5: -2,-1,ref,1,2
    '''
    pairs = []
    if mode == "sequence":
        assert (seq_len % 2) == 1
        for i in range(len(img_list)):
            start = i - (seq_len-1)//2
            end = i + (seq_len-1)//2 + 1
            if start < 0:
                start = 0
                end = start + seq_len
            if end > len(img_list):
                start = len(img_list) - seq_len
                end = len(img_list)
            for j in range(start, end):
                if i == j:
                    continue
                pairs.append((img_list[i], img_list[j]))
    elif mode == "near":
        for i in range(len(img_list)):
            if i != len(img_list) - 1:
                pairs.append((img_list[i], img_list[i+1]))
            else:
                pairs.append((img_list[i], img_list[i-1]))
    elif mode == "self":
        for i in range(len(img_list)):
            pairs.append((img_list[i], img_list[i]))
    else:
        raise NotImplementedError
        
    return pairs

def rescale_to_orig(data, size_info):
    assert len(data.shape) == 3
    assert isinstance(data, np.ndarray)
    # img_oriWH -> img_rescaleWH -> img_cropWH
    # crop: img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh)), cx,cy=img_rescaleWH//2
    # img_oriW, img_oriH, img_rescaleW, img_rescaleH, cx-halfw, cy-halfh, cx+halfw, cy+halfh = size_info
    img_oriW, img_oriH, img_rescaleW, img_rescaleH, w_start, h_start, w_end, h_end = size_info
    # first recover to size before crop
    data = np.pad(data, ((h_start, img_rescaleH - h_end), (w_start, img_rescaleW - w_end), (0, 0)), mode='edge')
    # then recover to size before rescale
    data = cv2.resize(data, (img_oriW, img_oriH), interpolation=cv2.INTER_NEAREST)
    return data

def get_avg_normal(normal_list, delete_bad_normal=True):
    normal_array = np.array(normal_list)
    avg_normal = np.mean(normal_array, axis=0)
    
    if delete_bad_normal:
        bad_idx, max_err = 0, 0
        for i in range(normal_array.shape[0]):
            if np.abs(normal_array[i] - avg_normal).mean() > max_err:
                bad_idx, max_err = i, np.abs(normal_array[i] - avg_normal).mean()
        normal_list.pop(bad_idx)
        avg_normal = np.mean(np.array(normal_list), axis=0)
        
    # normalize
    avg_normal = avg_normal / (np.linalg.norm(avg_normal, axis=-1)[..., None])
    
    return avg_normal

def rotate_recover(output):
    for view_id, pred_id in [["view1", "pred1"], ["view2", "pred2"]]:
        if output[view_id]["rotate"][0]:
            output[view_id]["img"] = torch.rot90(output[view_id]["img"], k=-1, dims=(2, 3))
            output[view_id]["true_shape"] = output[view_id]["true_shape"][:,[1,0]]
            output[view_id]["size_info"] = output[view_id]["size_info"][[1,0,3,2,5,4,7,6]]
            output[view_id]["size_info_matching"] = output[view_id]["size_info_matching"][[1,0,3,2,5,4,7,6]]
            
            R = torch.from_numpy(Rotation.from_euler('z', 90 * 1, degrees=True).as_matrix()).float()
            if pred_id == "pred1":
                pts_rotate = torch.rot90(output[pred_id]["pts3d"], k=-1, dims=(1, 2))
                output[pred_id]["pts3d"] = (R[:3, :3] @ pts_rotate.reshape(-1, 3).T).T.reshape(pts_rotate.shape)
            else:
                pts_rotate = torch.rot90(output[pred_id]["pts3d_in_other_view"], k=-1, dims=(1, 2))
                output[pred_id]["pts3d_in_other_view"] = (R[:3, :3] @ pts_rotate.reshape(-1, 3).T).T.reshape(pts_rotate.shape)
                
            output[pred_id]["desc"] = torch.rot90(output[pred_id]["desc"], k=-1, dims=(1, 2))
            output[pred_id]["desc_conf"] = torch.rot90(output[pred_id]["desc_conf"], k=-1, dims=(1, 2))
            normal_rotate = torch.rot90(output[pred_id]["normal"], k=-1, dims=(1, 2))
            R = torch.from_numpy(Rotation.from_euler('z', 90 * -1, degrees=True).as_matrix()).float()
            output[pred_id]["normal"] = (R[:3, :3] @ normal_rotate.reshape(-1, 3).T).T.reshape(normal_rotate.shape)
        
    return output

def infer_imgs(model, img_dir, save_dir, 
               size=512, device="cuda", 
               seq_len=3, pair_mode="self", 
               save_normal=False, save_depth=False, save_pcd=False,
               ret_total_output=False):
    valid_img_ext = ["*.jpg", "*.png", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
    if isinstance(model, str):
        model = AsymmetricMASt3R.from_pretrained(model).to(device)
    # load images data
    img_list = []
    for img_ext in valid_img_ext:
        img_list = img_list + glob.glob(os.path.join(img_dir, img_ext))
    img_list = sorted(img_list)
    # decide final size if size is None
    ref_img = cv2.imread(img_list[0])
    if max(ref_img.shape) >= 1024:
        final_size = 1024
    else:
        final_size = max(ref_img.shape)
    if size is None:
        size = final_size
    images = load_images(img_list, size=size, square_ok=True)
    pairs = make_pairs(images, mode=pair_mode, seq_len=seq_len)
    
    ret_dict = {
        "rgbs": [],
        "normals": [],
        "depths": [],
        "pts": [],
        "pts_another": [],
        "outputs": []
    } # for online demo usage
    
    for i in tqdm(range(len(images)), desc="Infering result"):
        if pair_mode in ["sequence"]:
            pair_slice = pairs[i*(seq_len-1) : i*(seq_len-1) + (seq_len-1)]
            output = inference(pair_slice, model, device, batch_size=1, verbose=False)
            output = rotate_recover(output)
            view1, pred1 = output['view1'], output['pred1']
            view2, pred2 = output['view2'], output['pred2']
            # get avg normal of a sequence
            normal_list = []
            for j in range(seq_len-1):
                normal_list.append(pred1["normal"][j, ...].detach().cpu().numpy()) # H, W, 3
            avg_normal = get_avg_normal(normal_list)
        elif pair_mode in ["self", "near"]:
            pair_slice = pairs[i:i+1]
            output = inference(pair_slice, model, device, batch_size=1, verbose=False)
            output = rotate_recover(output)
            view1, pred1 = output['view1'], output['pred1']
            view2, pred2 = output['view2'], output['pred2']
            avg_normal = pred1["normal"][0, ...].detach().cpu().numpy()
        else:
            raise NotImplementedError
        rgb = inverse_normalize(view1["img"][0, ...]).detach().cpu().permute(1, 2, 0).numpy() # H, W, 3
        points_3d = pred1["pts3d"][0, ...].detach().cpu().numpy() # H, W, 3
        points_3d_another = pred2["pts3d_in_other_view"][0, ...].detach().cpu().numpy() # H, W, 3

        imgname = images[i]["model_id"]
        pcd_save_path = os.path.join(save_dir, f"{imgname[:-4]}_pointmap.ply")
        
        # right hand to left hand
        avg_normal[..., 0] = -avg_normal[..., 0]
        avg_normal_viz = normal2color(avg_normal)
        pre_depth = points_3d[..., -1]
        pcd = points_3d.reshape(-1, 3)
        
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pcd)
        pts_rgb = rgb.reshape(-1, 3)
        o3d_pcd.colors = o3d.utility.Vector3dVector(pts_rgb)
        
        ret_dict["normals"].append(avg_normal_viz[..., [2,1,0]])
        ret_dict["depths"].append(pre_depth)
        ret_dict["pts"].append(points_3d)
        ret_dict["pts_another"].append(points_3d_another)
        ret_dict["rgbs"].append(rgb)
        if ret_total_output: # not append to save memory
            ret_dict["outputs"].append(output)
        
        if save_normal:
            os.makedirs(save_dir, exist_ok=True)
            np.savez_compressed(pcd_save_path.replace("_pointmap.ply", "_normal.npz"), avg_normal)
            cv2.imwrite(pcd_save_path.replace("_pointmap.ply", "_normal_viz.png"), avg_normal_viz[..., [2,1,0]])
        if save_depth:
            os.makedirs(save_dir, exist_ok=True)
            pre_depth_viz = colorize(pre_depth, cmap="Spectral", vmin=None, vmax=None)
            cv2.imwrite(pcd_save_path.replace("_pointmap.ply", "_depth_viz.png"), pre_depth_viz)
        if save_pcd:
            os.makedirs(save_dir, exist_ok=True)
            o3d.io.write_point_cloud(pcd_save_path, o3d_pcd)
    
    return ret_dict

def parse_args():
    parser = argparse.ArgumentParser(description="Infer normal and depth from RGB images")
    
    parser.add_argument("--model_path", required=True, help="path to model checkpoint")
    parser.add_argument("--img_dir", required=True, help="path to images folder")
    parser.add_argument("--save_dir", required=True, help="path to save folder")
    parser.add_argument("--pair_mode", default="near", choices=["self", "near", "sequence"], help="mode to pair images")
    parser.add_argument("--size", default=512, type=int, help="image infer size")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    infer_imgs(model = args.model_path,
               img_dir = args.img_dir,
               save_dir = args.save_dir,
               size=args.size, 
               device="cuda",
               pair_mode="near", 
               save_depth=True, 
               save_normal=True, 
               save_pcd=True)
    
    
if __name__ == "__main__":
    main()
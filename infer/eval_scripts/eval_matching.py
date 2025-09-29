import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(os.path.realpath(os.path.dirname(os.path.dirname(__file__))))

from dust3r.inference import inference
from mast3r.model import AsymmetricMASt3R
from dust3r.utils.image import load_images
from mast3r.fast_nn import extract_correspondences_nonsym
from infer.infer_normal_pts3d import rotate_recover
from infer.eval_scripts.matching_metrics import compute_pose_errors_numpy, error_auc, vis_matching


def recover_coordinate(match_info, size_info):
    # img_oriWH -> img_rescaleWH -> img_cropWH
    # crop: img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh)), cx,cy=img_rescaleWH//2
    img_oriW, img_oriH, img_rescaleW, img_rescaleH, img_finalW, img_finalH, img_cropW, img_cropH = size_info.numpy()
    # crop to no_crop
    match_info[:, 0] = (match_info[:, 0] + (img_rescaleW//2 - img_cropW)) * (img_oriW / img_rescaleW)
    match_info[:, 1] = (match_info[:, 1] + (img_rescaleH//2 - img_cropH)) * (img_oriH / img_rescaleH)

    return match_info

def extract_match(output, device="cuda", conf_thr=1.001, pixel_tol=5, subsample=8):
    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()
    # find 2D-2D matches between the two images
    corres = extract_correspondences_nonsym(desc1, desc2, pred1['desc_conf'], pred2['desc_conf'],
                                            device=device, subsample=subsample, pixel_tol=pixel_tol)
    conf = corres[2]
    if conf_thr == "median":
        mask = conf >= conf.median()
    else:
        mask = conf >= conf_thr
    matches_im0 = corres[0][mask].cpu().numpy()
    matches_im1 = corres[1][mask].cpu().numpy()
    # ignore small border around the edge
    H0, W0 = view1['true_shape'][0]
    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
        matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)
    H1, W1 = view2['true_shape'][0]
    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
        matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]
    # recover origin coordinates in imgs
    matches_im0 = matches_im0.astype(np.float64)
    matches_im1 = matches_im1.astype(np.float64)
    
    ori_matches_im0 = recover_coordinate(matches_im0, view1['size_info_matching'])
    ori_matches_im1 = recover_coordinate(matches_im1, view2['size_info_matching'])
    
    return ori_matches_im0, ori_matches_im1

def load_img_dict(pair01_list, pair02_list, size=1024):
    unique_list = list(set(pair01_list + pair02_list))
    unique_list.sort()
    unique_images = load_images(unique_list, size=size, square_ok=True)
    unique_images_dict = {item: unique_images[i] for i, item in enumerate(unique_list)}
    
    return unique_images_dict

def load_img_pairs(pair01_list, pair02_list, size=1024):
    images01 = load_images(pair01_list, size=size, square_ok=True)
    images02 = load_images(pair02_list, size=size, square_ok=True)
    pairs = []
    assert len(images01) == len(images02)
    for i in range(len(images01)):
        pairs.append((images01[i], images02[i]))

    return pairs

def get_pred(model_path, pair01_list, pair02_list, save_path, 
             device="cuda", size=512, datatype="gim", conf_thr=1.001, pixel_tol=5, subsample=8):
    model = AsymmetricMASt3R.from_pretrained(model_path).to(device)
    
    unique_image_dict = load_img_dict(pair01_list, pair02_list, size=size)
    for i in tqdm(range(len(pair01_list)), desc=f"Infering result"):
        pair_slice = [(unique_image_dict[pair01_list[i]], unique_image_dict[pair02_list[i]])]
        output = inference(pair_slice, model, device, batch_size=1, verbose=False)
        output = rotate_recover(output)
        ori_matches_im0, ori_matches_im1 = extract_match(output, conf_thr=conf_thr, pixel_tol=pixel_tol, subsample=subsample)
        # save matches
        if datatype == "gim" or datatype == "megadepth":
            save_name = f"{os.path.basename(pair01_list[i])}_{os.path.basename(pair02_list[i])}.npz"
            np.savez_compressed(os.path.join(save_path, save_name), matches_im0=ori_matches_im0, matches_im1=ori_matches_im1)
        elif datatype == "scannet":
            scene_name = pair01_list[i].split("/")[-4]
            save_name = f"{scene_name}_{os.path.basename(pair01_list[i])}_{scene_name}_{os.path.basename(pair02_list[i])}.npz"
            np.savez_compressed(os.path.join(save_path, save_name), matches_im0=ori_matches_im0, matches_im1=ori_matches_im1)
        else:
            raise NotImplementedError(f"datatype {datatype} not implemented")
    
    del unique_image_dict
    
def read_gim_txt(path):
    with open(path, 'r') as f:
        data = f.readlines()
    head = data[0].split()
    img1name, img2name = head[0], head[1]
    unknown1, unkonwn2 = head[2], head[3]
    intrin1 = np.array(head[4:13]).astype(np.float32).reshape(3, 3)
    intrin2 = np.array(head[13:22]).astype(np.float32).reshape(3, 3)
    extrin = np.array(head[22:]).astype(np.float32).reshape(4, 4)
    return {
        "img1name": img1name,
        "img2name": img2name,
        "intrin1": intrin1,
        "intrin2": intrin2,
        "extrin": extrin
    }

def extract_gim_pairs(txt_list, sub="blendedmvs"):
    pairs = []
    for filename in txt_list:
        if sub in ["blendedmvs", "gl3d", "robotcarweather", "robotcarseason", "robotcarnight"]:
            prefix, img0, img1 = filename.replace(".txt", "").split("_")
            pairs.append([f"{prefix}_{img0}.png", f"{prefix}_{img1}.png"])
        elif sub in ["multifov", "eth3di", "eth3do", "kitti", "iclnuim", "gtasfm", "scenenet"]:
            prefix, img0, img1 = filename.replace(".txt", "").split("-")
            pairs.append([f"{prefix}-{img0}.png", f"{prefix}-{img1}.png"])
        else:
            raise NotImplementedError(f"sub {sub} not implemented")
    return pairs

def eval_gim(model_path, data_dir, pred_save_dir, sub="blendedmvs"):    
    data_prefix = os.path.join(data_dir, sub)
    all_data_list = os.listdir(data_prefix)
    all_data_list = all_data_list[:100]
    all_txt_list = [item for item in all_data_list if item.endswith(".txt")]
    all_txt_list = sorted(all_txt_list)
    pairs = extract_gim_pairs(all_txt_list, sub=sub)
    img1_list = [os.path.join(data_prefix, item[0]) for item in pairs]
    img2_list = [os.path.join(data_prefix, item[1]) for item in pairs]
    
    # predict
    if not os.path.exists(pred_save_dir):
        os.makedirs(pred_save_dir)
        get_pred(model_path, img1_list, img2_list, 
                    pred_save_dir, device="cuda", size=512, datatype="gim")
    # evaluate
    log_info = []
    thresholds=[5, 10, 20]
    auc_results = {
        str(item): [] for item in thresholds
    }
    for i in range(len(img1_list)):
        txt_path = os.path.join(data_prefix, all_txt_list[i])
        gt_info = read_gim_txt(txt_path)
        T_0to1 = gt_info["extrin"]
        K0, K1 = gt_info["intrin1"], gt_info["intrin2"]
        
        pred_path = os.path.join(pred_save_dir, f"{os.path.basename(img1_list[i])}_{os.path.basename(img2_list[i])}.npz")
        pred_info = np.load(pred_path)
        matches_im0, matches_im1 = pred_info["matches_im0"], pred_info["matches_im1"]
        if matches_im0.shape[0] == 0:
            print(f"{img1_list[i]} & {img2_list[i]}: no matches")
            log_info.append(f"{img1_list[i]} & {img2_list[i]}: no matches")
            for auc_i in thresholds:
                auc_results[str(auc_i)].append(0)
            continue
        data = dict(
            color0=cv2.imread(img1_list[i]),
            color1=cv2.imread(img2_list[i]),
            mkpts0_f=matches_im0,
            mkpts1_f=matches_im1,
            K0=K0, K1=K1, T_0to1=T_0to1,
        )
        compute_pose_errors_numpy(data, pixel_thr=0.5)
        error_info = error_auc([data["R_errs"], data["t_errs"]], thresholds=thresholds)
        print(f"{img1_list[i]} & {img2_list[i]}: {error_info}")
        log_info.append(f"{img1_list[i]} & {img2_list[i]}: {error_info}")
        for auc_i in thresholds:
            auc_results[str(auc_i)].append(error_info[f"auc@{auc_i}"])
        try:
            vis_debug = vis_matching(data['color0'], data['color1'], data['mkpts0_f'], data['mkpts1_f'])
            cv2.imwrite(os.path.join(pred_save_dir, f"{os.path.basename(img1_list[i])}_{os.path.basename(img2_list[i])}.jpg"), vis_debug)
        except:
            print(f"{img1_list[i]} & {img2_list[i]}: vis_debug error")
            log_info.append(f"{img1_list[i]} & {img2_list[i]}: vis_debug error")
    for auc_i in thresholds:
        print(f"total_auc: auc@{str(auc_i)}: {np.mean(auc_results[str(auc_i)])}")
        log_info.append(f"total_auc: auc@{str(auc_i)}: {np.mean(auc_results[str(auc_i)])}")
    with open(os.path.join(pred_save_dir, "log.txt"), "w") as f:
        f.write("\n".join(log_info))

def eval_scannet_1500(model_path, data_dir, pred_save_dir):
    info_path = os.path.join(data_dir, "scannet_test_pairs_with_gt.txt")
    if not os.path.exists(info_path):
        raise ValueError(f"info_path {info_path} does not exist")
    with open(info_path, "r") as f:
        gt_info = f.readlines()
    gt_info = [item.split() for item in gt_info]
    
    img1_list, img2_list = [], []
    for tmp_info in gt_info:
        scene_name = tmp_info[0].split("/")[1]
        frame_idx1 = int(tmp_info[0].split("/")[-1].replace("frame-", "").replace(".color.jpg", ""))
        frame_idx2 = int(tmp_info[1].split("/")[-1].replace("frame-", "").replace(".color.jpg", ""))
        img1_list.append(os.path.join(data_dir, scene_name, "sens/color", f"{frame_idx1}.jpg"))
        img2_list.append(os.path.join(data_dir, scene_name, "sens/color", f"{frame_idx2}.jpg"))
    
    if not os.path.exists(pred_save_dir):
        os.makedirs(pred_save_dir)
        get_pred(model_path, img1_list, img2_list, 
                 pred_save_dir, device="cuda", size=512, datatype="scannet", conf_thr=1.001, pixel_tol=5, subsample=8)
    # evaluate
    log_info = []
    thresholds=[5, 10, 20]
    auc_results = {
        str(item): [] for item in thresholds
    }
    for i in range(len(img1_list)):
        scene_name = gt_info[i][0].split("/")[1]
        pred_path = os.path.join(pred_save_dir, f"{scene_name}_{os.path.basename(img1_list[i])}_{scene_name}_{os.path.basename(img2_list[i])}.npz")
        pred_info = np.load(pred_path)
        matches_im0, matches_im1 = pred_info["matches_im0"], pred_info["matches_im1"]
        if matches_im0.shape[0] == 0:
            print(f"{img1_list[i]} & {img2_list[i]}: no matches")
            log_info.append(f"{img1_list[i]} & {img2_list[i]}: no matches")
            for auc_i in thresholds:
                auc_results[str(auc_i)].append(0)
            continue
        # compute epipolar distance
        assert len(gt_info[i]) == 38, 'Pair does not have ground truth info'
        K0 = np.array(gt_info[i][4:13]).astype(float).reshape(3, 3)
        K1 = np.array(gt_info[i][13:22]).astype(float).reshape(3, 3)
        T_0to1 = np.array(gt_info[i][22:]).astype(float).reshape(4, 4)
        
        data = dict(
            color0=cv2.imread(img1_list[i]),
            color1=cv2.imread(img2_list[i]),
            mkpts0_f=matches_im0,
            mkpts1_f=matches_im1,
            K0=K0, K1=K1, T_0to1=T_0to1,
        )
        compute_pose_errors_numpy(data, pixel_thr=0.5)
        error_info = error_auc([data["R_errs"], data["t_errs"]], thresholds=thresholds)
        print(f"{img1_list[i]} & {img2_list[i]}: {error_info}")
        log_info.append(f"{img1_list[i]} & {img2_list[i]}: {error_info}")

        for auc_i in thresholds:
            auc_results[str(auc_i)].append(error_info[f"auc@{auc_i}"])
    for auc_i in thresholds:
        print(f"total_auc: auc@{str(auc_i)}: {np.mean(auc_results[str(auc_i)])}")
        log_info.append(f"total_auc: auc@{str(auc_i)}: {np.mean(auc_results[str(auc_i)])}")
    with open(os.path.join(pred_save_dir, "log.txt"), "w") as f:
        f.write("\n".join(log_info))

def parse_args():
    parser = argparse.ArgumentParser(description="Matching evaluation script")
    
    parser.add_argument("--model_path", required=True, help="path to model checkpoint")
    parser.add_argument("--data_dir", required=True, help="path to dataset folder")
    parser.add_argument("--save_dir", required=True, help="path to save folder")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    for sub_set in ["blendedmvs", "kitti", "multifov", "robotcarweather", 
                    "robotcarseason", "robotcarnight", "scenenet", "eth3di", 
                    "eth3do", "gtasfm", "iclnuim", "gl3d"]:
        eval_gim(args.model_path, args.data_dir, args.save_dir, sub=sub_set)
        break


if __name__ == "__main__":
    main()

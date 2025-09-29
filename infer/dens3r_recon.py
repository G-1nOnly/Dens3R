# Copyright (C) 2025-present Alibaba Group. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# reconstruct 3D scene from images
# --------------------------------------------------------
import os
import pickle
import argparse
import datetime
import pycolmap
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.cluster.hierarchy import DisjointSet

import sys
sys.path.append(os.path.realpath(os.path.dirname(os.path.dirname(__file__))))
from mast3r.colmap_utils.database_utils import *
import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import extract_correspondences_nonsym
from dust3r.utils.image import load_images
from infer.infer_normal_pts3d import rotate_recover


def infer_pair_matches_batch(imgs_pairs, model, img_dir,
                             device="cuda", size=512, 
                             conf_thr=1.001, pixel_tol=5, subsample=8):
    # remove duplicate pairs
    imgs_pairs_filtered = []
    for pair in imgs_pairs:
        if [pair[0], pair[1]] not in imgs_pairs_filtered and [pair[1], pair[0]] not in imgs_pairs_filtered:
            imgs_pairs_filtered.append(pair)
    imgs_pairs = imgs_pairs_filtered
    # no mask
    imgname_set = set()
    for pair in imgs_pairs:
        for imgname in pair:
            imgname_set.add(imgname)
    imgname_list = sorted(list(imgname_set))
    imgs_data_list = load_images([os.path.join(img_dir, imgname) for imgname in imgname_list], 
                                 size=size, verbose=False)
    imgs_data_dict = {}
    for i in range(len(imgname_list)):
        imgs_data_dict[imgname_list[i]] = imgs_data_list[i]
        
    infer_match_list = []
    for pair in tqdm(imgs_pairs, desc="Dens3R Model Infer Pairs"):
        output = inference([tuple([imgs_data_dict[pair[0]], imgs_data_dict[pair[1]]])], 
                           model, device, batch_size=1, verbose=False)
        output = rotate_recover(output)

        # at this stage, you have the raw dust3r predictions
        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']
        desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()
        # find 2D-2D matches between the two images
        corres = extract_correspondences_nonsym(desc1, desc2, pred1['desc_conf'], pred2['desc_conf'],
                                                device=device, subsample=subsample, pixel_tol=pixel_tol)
        conf = corres[2]
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
        
        infer_match_list.append([ori_matches_im0, ori_matches_im1])
    return imgs_pairs, infer_match_list

def recover_coordinate(match_info, size_info):
    # img_oriWH -> img_rescaleWH -> img_cropWH
    # crop: img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh)), cx,cy=img_rescaleWH//2
    img_oriW, img_oriH, img_rescaleW, img_rescaleH, img_finalW, img_finalH, img_cropW, img_cropH = size_info.numpy()
    # crop to no_crop
    match_info[:, 0] = (match_info[:, 0] + (img_rescaleW//2 - img_cropW)) * (img_oriW / img_rescaleW)
    match_info[:, 1] = (match_info[:, 1] + (img_rescaleH//2 - img_cropH)) * (img_oriH / img_rescaleH)
    return match_info

def get_img_pairs(img_dir, mode="sequential", window_size=20):
    img_list = os.listdir(img_dir)
    img_list.sort()
    img_name_dict = {}
    img_pairs = []
    if mode == "sequential":
        if len(img_list) < window_size:
            raise ValueError(f"img_list length {len(img_list)} < window_size {window_size}")
        for i in range(len(img_list)):
            start = i - window_size//2
            if start < 0:
                start = 0
            end = start + window_size
            if end > len(img_list):
                end = len(img_list)
                start = end - window_size
            for j in range(start, end):
                if i == j:
                    continue
                img_pairs.append([img_list[i], img_list[j]])
            img_name_dict[img_list[i]] = len(img_name_dict) + 1
        
    elif mode == "exhaustive":
        for i in range(len(img_list)):
            for j in range(len(img_list)):
                if i == j:
                    continue
                img_pairs.append([img_list[i], img_list[j]])
            img_name_dict[img_list[i]] = len(img_name_dict) + 1
    else:
        raise NotImplementedError(f"mode {mode} not implemented!")
    
    return img_name_dict, img_pairs

def save_img_pairs_txt(img_pairs, sfm_dir):
    with open(os.path.join(sfm_dir, "pairs-sfm.txt"), 'w') as f:
        for pair in img_pairs:
            f.write(' '.join(pair))
            f.write("\n")
    
def colmap_recon(imgs_keypts_dict : dict, matches_dict : dict, img_name_dict : dict,
                 sfm_dir : str, image_dir : str, skip_geometric_verification : bool = True,
                 use_mapper : str = "colmap", camera_model : str = "SIMPLE_RADIAL"):
    sfm_dir = Path(sfm_dir)
    image_dir = Path(image_dir)
    database = sfm_dir / "database.db"
    img_pairs_path = sfm_dir / "pairs-sfm.txt"
    camera_mode = pycolmap.CameraMode.SINGLE
    
    create_empty_db(database)
    colmap_img_list = sorted(os.listdir(str(image_dir)))
    import_images(image_dir, database, camera_mode, 
                  colmap_img_list, options={'camera_model': camera_model})
    image_ids = get_image_ids(database) # image_ids == img_name_dict, TODO test
    import_keypts(database, imgs_keypts_dict, img_name_dict)
    import_mts(database, matches_dict, skip_geometric_verification=skip_geometric_verification)
    if not skip_geometric_verification:
        estimation_and_geometric_verification(database, img_pairs_path, verbose=False)
    
    if use_mapper == "colmap":
        reconstruction = run_reconstruction(
            sfm_dir/"hloc_sfm", database, image_dir, verbose=False, options=None)
    elif use_mapper == "glomap":
        reconstruction = glomap_run_mapper(glomap_bin="glomap",
                                           colmap_db_path=str(database),
                                           recon_path=str(sfm_dir/"hloc_sfm"),
                                           image_root_path=str(image_dir))
    else:
        raise NotImplementedError("use_mapper must in [colmap, glomap]")
    
    if reconstruction is not None:
        print(f'Reconstruction statistics:\n{reconstruction.summary()}'
            + f'\n\tnum_input_images = {len(image_ids)}')
        
def colmap_triangulation(imgs_keypts_dict : dict, matches_dict : dict, img_name_dict : dict,
                         sfm_dir : str, skip_geometric_verification : bool = True, img_dir : str = "images", 
                         init_model : str = "init_pose"):
    sfm_dir = Path(sfm_dir)
    database = sfm_dir / "database.db"
    image_dir = Path(img_dir)
    img_pairs_path = sfm_dir / "pairs-sfm.txt"
    camera_mode = pycolmap.CameraMode.AUTO
    # camera_mode = pycolmap.CameraMode.SINGLE
    # camera_mode = pycolmap.CameraMode.PER_FOLDER
    # camera_mode = pycolmap.CameraMode.PER_IMAGE
    reference_model = Path(init_model)
    
    reference = pycolmap.Reconstruction(reference_model)
    _ = create_db_from_model(reference, database)
    image_ids = get_image_ids(database) # image_ids == img_name_dict, TODO test
    colmap_img_list = sorted(os.listdir(str(image_dir)))
    import_images(image_dir, database, camera_mode, colmap_img_list)
    import_keypts(database, imgs_keypts_dict, img_name_dict)
    import_mts(database, matches_dict, skip_geometric_verification=skip_geometric_verification)
    if not skip_geometric_verification:
        estimation_and_geometric_verification(database, img_pairs_path, verbose=False)
    
    reconstruction = run_triangulation(sfm_dir/"hloc_sfm", database, image_dir, reference,
                                       verbose=False)
    if reconstruction is not None:
        print(f'Reconstruction statistics:\n{reconstruction.summary()}'
            + f'\n\tnum_input_images = {len(image_ids)}')

def mast3r_match_tracking(model : Any,
                          img_dir : str,
                          sfm_dir : str,
                          matching_mode : str = "sequential", # or exhaustive
                          window_size : int = 20,
                          device="cuda", size=512,
                          min_track_length: int = 3,
                          conf_thr=1.001, pixel_tol=5, subsample=8):
    if isinstance(model, str):
        model = AsymmetricMASt3R.from_pretrained(model).to(device)
    
    img_name_dict, img_pairs = get_img_pairs(img_dir, mode=matching_mode, window_size=window_size)
    img_name_dict_idx2name = {}
    for imgname, imgidx in img_name_dict.items():
        img_name_dict_idx2name[imgidx] = imgname
    save_img_pairs_txt(img_pairs, sfm_dir)
    
    imgs_keypts_dict = {imgname: {} for imgname in img_name_dict.keys()}
    
    imgs_pairs, infer_match_list = infer_pair_matches_batch(img_pairs, model, img_dir,
                                                            device=device, size=size, 
                                                            conf_thr=conf_thr, 
                                                            pixel_tol=pixel_tol, 
                                                            subsample=subsample)
    track_id_to_kpt_list = []
    to_merge = []
    matches_dict = {}
    for i in range(len(imgs_pairs)):
        img_pair = imgs_pairs[i]
        ori_matches_im0, ori_matches_im1 = infer_match_list[i]
        # get keypoints for each image
        matches_uv_pair_list = []
        
        for i_pts in range(ori_matches_im0.shape[0]):
            if tuple(ori_matches_im0[i_pts, :]) not in imgs_keypts_dict[img_pair[0]] and \
                tuple(ori_matches_im1[i_pts, :]) not in imgs_keypts_dict[img_pair[1]]:
                track_idx = len(track_id_to_kpt_list)
                track_id_to_kpt_list.append([(img_pair[0], tuple(ori_matches_im0[i_pts, :])), 
                                             (img_pair[1], tuple(ori_matches_im1[i_pts, :]))])
                imgs_keypts_dict[img_pair[0]][tuple(ori_matches_im0[i_pts, :])] = track_idx
                imgs_keypts_dict[img_pair[1]][tuple(ori_matches_im1[i_pts, :])] = track_idx
                
            elif tuple(ori_matches_im0[i_pts, :]) in imgs_keypts_dict[img_pair[0]] and \
                tuple(ori_matches_im1[i_pts, :]) not in imgs_keypts_dict[img_pair[1]]:
                track_idx = imgs_keypts_dict[img_pair[0]][tuple(ori_matches_im0[i_pts, :])]
                imgs_keypts_dict[img_pair[1]][tuple(ori_matches_im1[i_pts, :])] = track_idx
                track_id_to_kpt_list[track_idx].append((img_pair[1], tuple(ori_matches_im1[i_pts, :])))
                
            elif tuple(ori_matches_im0[i_pts, :]) not in imgs_keypts_dict[img_pair[0]] and \
                tuple(ori_matches_im1[i_pts, :]) in imgs_keypts_dict[img_pair[1]]:
                track_idx = imgs_keypts_dict[img_pair[1]][tuple(ori_matches_im1[i_pts, :])]
                imgs_keypts_dict[img_pair[0]][tuple(ori_matches_im0[i_pts, :])] = track_idx
                track_id_to_kpt_list[track_idx].append((img_pair[0], tuple(ori_matches_im0[i_pts, :])))
                
            elif tuple(ori_matches_im0[i_pts, :]) in imgs_keypts_dict[img_pair[0]] and \
                tuple(ori_matches_im1[i_pts, :]) in imgs_keypts_dict[img_pair[1]]:
                # both have tracks, merge them
                track_idx0 = imgs_keypts_dict[img_pair[0]][tuple(ori_matches_im0[i_pts, :])]
                track_idx1 = imgs_keypts_dict[img_pair[1]][tuple(ori_matches_im1[i_pts, :])]
                if track_idx0 != track_idx1:
                    # let's deal with them later
                    to_merge.append((track_idx0, track_idx1))
            else:
                raise RuntimeError("Impossible to run here")
            matches_uv_pair_list.append([tuple(ori_matches_im0[i_pts, :]), tuple(ori_matches_im1[i_pts, :])])
        # img idx: match keypts idx
        matches_dict[(img_name_dict[img_pair[0]], img_name_dict[img_pair[1]])] = matches_uv_pair_list
    
    # regroup merge targets
    print("merging tracks")
    unique = np.unique(to_merge)
    tree = DisjointSet(unique)
    for track_idx0, track_idx1 in tqdm(to_merge):
        tree.merge(track_idx0, track_idx1)

    subsets = tree.subsets()
    print("applying merge")
    for setvals in tqdm(subsets):
        new_trackid = len(track_id_to_kpt_list)
        kpt_list = []
        for track_idx in setvals:
            kpt_list.extend(track_id_to_kpt_list[track_idx])
            for imname, kpuv in track_id_to_kpt_list[track_idx]:
                imgs_keypts_dict[imname][kpuv] = new_trackid
        track_id_to_kpt_list.append(kpt_list)
        
    num_valid_tracks = sum(
        [1 for v in track_id_to_kpt_list if len(v) >= min_track_length])
    print(f"squashing keypoints {len(track_id_to_kpt_list)} to -> {num_valid_tracks} valid tracks")

    # re_orgnize imgs_keypts_dict, matches_dict
    final_imgs_keypts_dict = {}
    for imgname, keyptsuv_trackidx_dict in tqdm(imgs_keypts_dict.items(), desc="reorgnize keypts"):
        final_imgs_keypts_dict[imgname] = {} # tuple(u,v):track_id [start form 0]
        for keyptsuv, trackidx in keyptsuv_trackidx_dict.items():
            if len(track_id_to_kpt_list[trackidx]) < min_track_length:
                continue
            final_imgs_keypts_dict[imgname][keyptsuv] = len(final_imgs_keypts_dict[imgname])

    print("start reorgnize matching info from matching num", len(matches_dict), datetime.datetime.now())
    
    final_matches_dict = {} # (img0idx, img1idx): [[kypts0idx, keypts1idx], ...]
    for (img0idx, img1idx), matches_uv_list in tqdm(matches_dict.items(), desc="reorgnize matching"):
        img0name, img1name = img_name_dict_idx2name[img0idx], img_name_dict_idx2name[img1idx]
        keep_matching_list = []
        for uv_in_img0, uv_in_img1 in matches_uv_list:
            try:
                keep_matching_list.append([final_imgs_keypts_dict[img0name][uv_in_img0],
                                           final_imgs_keypts_dict[img1name][uv_in_img1]])
            except:
                pass
        final_matches_dict[(img0idx, img1idx)] = np.array(keep_matching_list)
    
    print("After matching num:", len(final_matches_dict), datetime.datetime.now())

    return final_imgs_keypts_dict, final_matches_dict, img_name_dict

def save_dict(dict_to_save, filepath):
    assert filepath.endswith(".pkl")
    f_save = open(filepath, 'wb')
    pickle.dump(dict_to_save, f_save)
    f_save.close()
 
def read_dict(filepath):
    assert filepath.endswith(".pkl")
    f_read = open(filepath, 'rb')
    dict_content = pickle.load(f_read)
    f_read.close()
    return dict_content

def manhattan_alignment(image_dir, sfm_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
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
    rec = pycolmap.Reconstruction(output_dir)
    return rec

def Dens3RReconstruction(path_or_model, img_dir, sfm_dir,
                         matching_mode="sequential", window_size=30, device="cuda", 
                         size=512, triangulate_from_pose=None, use_mapper="colmap",
                         camera_model="SIMPLE_RADIAL", conf_thr=1.001, pixel_tol=5, subsample=8):
    skip_geometric_verification = False
    os.makedirs(sfm_dir, exist_ok=True)
    
    if os.path.exists(os.path.join(sfm_dir, "dens3r_key_points.pkl")) and \
        os.path.exists(os.path.join(sfm_dir, "dens3r_matches.pkl")) and \
        os.path.exists(os.path.join(sfm_dir, "img_name_dict.pkl")):
        print("Reading matching info cache...")
        imgs_keypts_dict = read_dict(os.path.join(sfm_dir, "dens3r_key_points.pkl"))
        matches_dict = read_dict(os.path.join(sfm_dir, "dens3r_matches.pkl"))
        img_name_dict = read_dict(os.path.join(sfm_dir, "img_name_dict.pkl"))            
    else:
        imgs_keypts_dict, matches_dict, img_name_dict \
            = mast3r_match_tracking(path_or_model, img_dir, sfm_dir,
                                    matching_mode=matching_mode,
                                    window_size=window_size,
                                    device=device, size=size,
                                    min_track_length=3,
                                    conf_thr=conf_thr, pixel_tol=pixel_tol, subsample=subsample)
        print("Saving matching info...")
        save_dict(imgs_keypts_dict, os.path.join(sfm_dir, "dens3r_key_points.pkl"))
        save_dict(matches_dict, os.path.join(sfm_dir, "dens3r_matches.pkl"))
        save_dict(img_name_dict, os.path.join(sfm_dir, "img_name_dict.pkl"))
    
    if triangulate_from_pose is None:
        colmap_recon(imgs_keypts_dict, matches_dict, img_name_dict, sfm_dir, img_dir,
                     skip_geometric_verification=skip_geometric_verification, use_mapper=use_mapper, 
                     camera_model=camera_model)
    else:
        colmap_triangulation(imgs_keypts_dict, matches_dict, img_name_dict, sfm_dir,
                             skip_geometric_verification=skip_geometric_verification,
                             img_dir=img_dir, init_model=triangulate_from_pose)

def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruct pose and sparse pointclouds with Dens3R")
    
    parser.add_argument("--img_dir", required=True, help="path to images folder")
    parser.add_argument("--output_dir", required=True, help="path to sfm workspace folder")
    parser.add_argument("--model_path", required=True, help="path to model checkpoint")
    
    parser.add_argument("--triangulate_from_pose", default=None, 
                        help="colmap format pose file to triangulate from. If not None, only triangulate from this pose without pose estimation.")
    parser.add_argument("--matching_mode", choices=["exhaustive", "sequential"], default="exhaustive", 
                        help="matching mode")
    parser.add_argument("--window_size", type=int, default=20, 
                        help="window size for local matching, only works in local mode")
    parser.add_argument("--size", type=int, default=512,
                        help="infer resolution. suggested 512")
    parser.add_argument("--use_mapper", choices=["colmap", "glomap"], default="colmap", 
                        help="mapper to use")
    parser.add_argument("--camera_model", default="SIMPLE_RADIAL", 
                        help="camera model used in colmap reconstruction")
    parser.add_argument("--conf_thr", type=float, default=1.001, 
                        help="confidence threshold for matching")
    parser.add_argument("--pixel_tol", type=int, default=5, 
                        help="pixel tolerance for matching")
    parser.add_argument("--subsample", type=int, default=8, 
                        help="subsample keypoints. Smaller value produce more keypoints but slow down the reconstruction.")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    Dens3RReconstruction(path_or_model = args.model_path, 
                         img_dir = args.img_dir, 
                         sfm_dir = args.output_dir,
                         matching_mode = args.matching_mode, 
                         window_size = args.window_size, 
                         size = args.size, 
                         triangulate_from_pose = None,
                         use_mapper = args.use_mapper,
                         camera_model = args.camera_model,
                         conf_thr = args.conf_thr, 
                         pixel_tol = args.pixel_tol, 
                         subsample = args.subsample)


if __name__ == '__main__':
    main()
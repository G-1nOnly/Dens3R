# Copyright (C) 2025-present Alibaba Group. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# online gradio demo
# --------------------------------------------------------

import os
import gc
import pickle
import cv2
import torch
import shutil
import glob
import time
import trimesh
import argparse
import numpy as np
import gradio as gr
from datetime import datetime
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.realpath(os.path.dirname(os.path.dirname(__file__))))

from infer.infer_normal_pts3d import infer_imgs, rescale_to_orig
from infer.eval_scripts.matching_metrics import vis_matching
from infer.eval_scripts.eval_matching import extract_match
from infer.dens3r_recon import Dens3RReconstruction
from dust3r.inference import load_model
from dust3r.utils.image import colorize
from dust3r.post_process import estimate_focal_knowing_depth
from dust3r.utils.read_write_model import read_model, qvec2rotmat
from infer.demo_utils import predictions_to_glb, get_glb_from_recon_scene

plt.ion()
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12


def resize_image_max_size(image, image_size):
    h, w = image.shape[:2]
    scale = float(image_size) / max(h, w)
    if scale < 1:
        new_h, new_w =int(np.round(h * scale)), int(np.round(w * scale))
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized_image = image
    return resized_image

def get_two_view_recon(output, one_image_flag):
    extrinsics = []
    size_info = output["view1"]["size_info"].detach().cpu().numpy()
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    
    ori_pts3d_1_1 = pred1["pts3d"][0, ...].detach().cpu().numpy()
    pts3d_1_1 = rescale_to_orig(ori_pts3d_1_1, size_info)
    extrinsics.append(np.eye(4)[:3])
    if not one_image_flag:
        ori_pts3d_2_1 = pred2["pts3d_in_other_view"][0, ...].detach().cpu().numpy()
        pts3d_2_1 = rescale_to_orig(ori_pts3d_2_1, size_info)
        pts3d_1_1_torch = torch.from_numpy(pts3d_1_1)
        H, W = pts3d_1_1.shape[:2]
        center = torch.tensor([W/2, H/2])
        focal = estimate_focal_knowing_depth(pts3d_1_1_torch.unsqueeze(0), center.unsqueeze(0)).ravel()
        pixels = np.mgrid[:W, :H].T.astype(np.float32)
        K = torch.tensor([(focal, 0, center[0]), (0, focal, center[1]), (0, 0, 1)]).numpy()
        pixels = pixels.reshape(-1, 2)
        success, R, T, inliers = cv2.solvePnPRansac(pts3d_2_1.reshape(-1,3), 
                                                    pixels.reshape(-1,2),
                                                    K, None, iterationsCount=10,
                                                    reprojectionError=5, flags=cv2.SOLVEPNP_SQPNP)
        R = cv2.Rodrigues(R)[0]
        
        w2c = np.eye(4)
        w2c[:3,:3] = R
        w2c[:3, 3] = T.reshape(-1)
        extrinsics.append(w2c[:3])
        
        ori_matches_im0, ori_matches_im1 = extract_match(output, conf_thr=1.001, pixel_tol=5, subsample=8)
    else:
        ori_matches_im0, ori_matches_im1 = None, None

    return extrinsics, ori_matches_im0, ori_matches_im1

def get_two_view_prediction(filelist, out_dir):
    one_image_flag = True if len(filelist) == 1 else False
    # get mono_depth, mono_normal, viewspace_pts
    pair_mode = "self" if one_image_flag else "near"
    ret_dict = infer_imgs(model, img_dir=os.path.dirname(filelist[0]), 
                          save_dir=out_dir, pair_mode=pair_mode, size=512, 
                          save_normal=False, save_depth=False, save_pcd=True, ret_total_output=True)
    high_res_ret_dict = infer_imgs(model, img_dir=os.path.dirname(filelist[0]), 
                                   save_dir=out_dir, pair_mode=pair_mode, size=None, 
                                   save_normal=False, save_depth=False, save_pcd=False, ret_total_output=True)
    # get two-view extrinsics
    extrinsics, ori_matches_im0, ori_matches_im1 = get_two_view_recon(ret_dict["outputs"][0], one_image_flag)
    if not one_image_flag:
        matching_image = vis_matching(cv2.imread(filelist[0]), 
                                      cv2.imread(filelist[1]), 
                                      ori_matches_im0, ori_matches_im1,
                                      sample_matching_num=100)
    else:
        matching_image = None
    
    predictions = dict()
    if one_image_flag:
        predictions['world_points'] = ret_dict["pts"][0][None, ...]
        predictions['images'] = ret_dict["rgbs"][0][None, ...]
    else:
        predictions['world_points'] = np.stack([ret_dict["pts"][0], ret_dict["pts_another"][0]])
        predictions['images'] = np.stack(ret_dict["rgbs"])
    predictions['extrinsic'] = np.stack(extrinsics)
    predictions_path = os.path.join(out_dir, "predictions.npz")
    np.savez_compressed(predictions_path, **predictions)
    
    glbscene = predictions_to_glb(
            predictions,
            conf_thres=0,
            filter_by_frames='All',
            mask_black_bg=False,
            mask_white_bg=False,
            show_cam=True,
            prediction_mode="Predicted Pointmap"
        )
    glbfile = os.path.join(out_dir, "colmap_scene.glb")
    glbscene.export(file_obj=glbfile)

    return glbfile, ret_dict["rgbs"], high_res_ret_dict["depths"], high_res_ret_dict["normals"], matching_image

def get_sequence_prediction(filelist, out_dir):
    # get normal, depth, pts3d
    ret_dict = infer_imgs(model, img_dir=os.path.dirname(filelist[0]), save_dir=out_dir, 
                          pair_mode="near", size=512, save_normal=False, save_depth=False, save_pcd=True,
                          ret_total_output=True)
    high_res_ret_dict = infer_imgs(model, img_dir=os.path.dirname(filelist[0]), 
                                   save_dir=out_dir, pair_mode="near", size=None, 
                                   save_normal=False, save_depth=False, save_pcd=False, 
                                   ret_total_output=False)
    
    matching_images = []
    for i in range(len(filelist)):
        ori_matches_im0, ori_matches_im1 = extract_match(ret_dict["outputs"][i], conf_thr=1.001, pixel_tol=5, subsample=8)
        if i != len(filelist) - 1:
            file1, file2 = filelist[i], filelist[i+1]
        else:
            file1, file2 = filelist[i], filelist[i-1]
        matching_image = vis_matching(cv2.imread(file1), 
                                      cv2.imread(file2), 
                                      ori_matches_im0, ori_matches_im1,
                                      sample_matching_num=100)
        matching_images.append(matching_image)
    
    # recon sequence
    if len(filelist) > 30:
        pts_subsample = 8
    else:
        pts_subsample = 4
    Dens3RReconstruction(model, img_dir=os.path.dirname(filelist[0]), 
                         sfm_dir=out_dir, use_mapper="glomap", matching_mode="exhaustive", 
                         subsample=pts_subsample)
    
    colmap_intrinsics, colmap_extrinsics, colmap_points3D = \
        read_model(os.path.join(out_dir, "hloc_sfm/0"))

    def find_img_key_by_name(colmap_extrinsics, target_name):
        for idx, key in enumerate(colmap_extrinsics):
            if colmap_extrinsics[key].name == target_name:
                return key
        return None
    
    extrinsics = []
    valid_image_flag = []
    for idx, image_path in enumerate(filelist):
        file_name = os.path.basename(image_path)
        key = find_img_key_by_name(colmap_extrinsics, file_name)
        if key is None:
            valid_image_flag.append(False)
            continue
        else:
            valid_image_flag.append(True)
        extr = colmap_extrinsics[key]
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        w2c = np.eye(4)
        w2c[:3,:3] = R.T
        w2c[:3, 3] = T
        extrinsics.append(w2c[:3])
    
    valid_image_flag = np.array(valid_image_flag)
    predictions = dict()
    predictions['extrinsic'] = np.stack(extrinsics)
    points3d = []
    points_color = []
    for idx, (pt3d_id, pts3d) in enumerate(colmap_points3D.items()):
        points3d.append(pts3d.xyz)
        points_color.append(pts3d.rgb)
    predictions['world_points'] = np.stack(points3d)
    predictions['images'] = np.stack(ret_dict["rgbs"])
    predictions['pts_color'] = np.stack(points_color)
    
    outfile = get_glb_from_recon_scene(reconstruction_path=os.path.join(out_dir, "hloc_sfm/0"), 
                                       img_dir=os.path.join(out_dir, "images"), 
                                       cache_dir=out_dir, 
                                       outfile_name=os.path.join(out_dir, "scene_recon_True_All.glb"), 
                                       should_delete=False, transparent_cams=False, cam_size=0.8)
    
    valid_map = {}
    valid_idx = 0
    for idx in range(len(valid_image_flag)):
        if valid_image_flag[idx]:
            valid_map[idx] = valid_idx
            valid_idx = valid_idx + 1
        else:
            valid_map[idx] = 0
    predictions['valid_map'] = valid_map
    predictions_path = os.path.join(out_dir, "predictions.npz")
    np.savez_compressed(predictions_path, **predictions)
    
    return outfile, ret_dict["rgbs"], high_res_ret_dict["depths"], high_res_ret_dict["normals"], valid_image_flag, matching_images

def handle_uploads(input_video, input_images):
    """
    Create a new 'target_dir' + 'images' subfolder, and place user-uploaded
    images or extracted frames from video into it. Return (target_dir, image_paths).
    """
    max_size = 1024
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Create a unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"./log/input_{timestamp}"
    target_dir_images = os.path.join(target_dir, "images")

    # Clean up if somehow that folder already exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    os.makedirs(target_dir_images)

    image_paths = []

    # --- Handle images ---
    if input_images is not None:
        for file_data in input_images:
            if isinstance(file_data, dict) and "name" in file_data:
                file_path = file_data["name"]
            elif isinstance(file_data, tuple):
                file_path = file_data[0]
            else:
                file_path = file_data
            dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
            # shutil.copy(file_path, dst_path)
            frame = cv2.imread(file_path)
            frame = resize_image_max_size(frame, image_size=max_size)
            cv2.imwrite(dst_path, frame)
            image_paths.append(dst_path)

    # --- Handle video ---
    if input_video is not None:
        if isinstance(input_video, dict) and "name" in input_video:
            video_path = input_video["name"]
        else:
            video_path = input_video

        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 1)  # 1 frame/sec

        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            count += 1
            if count % frame_interval == 0:
                image_path = os.path.join(target_dir_images, f"{video_frame_num:06}.png")
                frame = resize_image_max_size(frame, image_size=max_size)
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                video_frame_num += 1

    # Sort final images for gallery
    image_paths = sorted(image_paths)

    end_time = time.time()
    print(f"Files copied to {target_dir_images}; took {end_time - start_time:.3f} seconds")
    return target_dir, image_paths

def update_gallery_on_upload(input_video, input_images):
    """
    Whenever user uploads or changes files, immediately handle them
    and show in the gallery. Return (target_dir, image_paths).
    If nothing is uploaded, returns "None" and empty list.
    """
    if not input_video and not input_images:
        return None, None, None, None
    target_dir, image_paths = handle_uploads(input_video, input_images)
    return None, target_dir, image_paths, "Upload complete. Click 'Reconstruct' to begin 3D processing."

def gradio_demo(
    target_dir,
    frame_filter="All"
):
    """
    Perform reconstruction using the already-created target_dir/images.
    """
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, None,  "No valid target directory found. Please upload first.", None

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Prepare frame_filter dropdown
    target_dir_images = os.path.join(target_dir, "images")
    image_extensions = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG", "*.bmp", "*.BMP", "*.gif", "*.GIF"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(target_dir_images, ext)))
        if len(image_files) > 0:
            break
    
    all_imgs = sorted(image_files)
    all_files = [f"{i}: {os.path.basename(filename)}" for i, filename in enumerate(all_imgs)]
    if len(all_imgs) == 1:
        recon_mode = "dens3r_single"
    elif len(all_imgs) == 2:
        recon_mode = "dens3r_pair"
    else:
        recon_mode = "dens3r_sequence"
    
    print("Model Infering...")
    with torch.no_grad():
        if recon_mode in ["dens3r_single", "dens3r_pair"]:
            glbfile, rgbimg, depth_list, normals, matching_image = \
                get_two_view_prediction(all_imgs, target_dir)
            frame_filter_choices = ["All"] + all_files
        else:
            glbfile, rgbimg, depth_list, normals, valid_image_flag, matching_images = \
                get_sequence_prediction(all_imgs, target_dir)
            frame_filter_choices = ["All"]
        
    gc.collect()
    torch.cuda.empty_cache()
    imgs = []
    
    if recon_mode == "dens3r_single":
        imgs.append((rgbimg[0]*255).astype(np.uint8))
        imgs.append(colorize(depth_list[0], vmin=0, vmax=depth_list[0].max(), cmap='Spectral')[..., :3])
        imgs.append(normals[0].astype(np.uint8)[..., [2,1,0]])
    elif recon_mode == "dens3r_pair":
        rgb_concat = (np.concatenate(rgbimg, axis=1) * 255).astype(np.uint8)
        depth_concat = np.concatenate([colorize(depth_list[0], vmin=0, vmax=depth_list[0].max(), cmap='Spectral')[..., :3], 
                                       colorize(depth_list[-1], vmin=0, vmax=depth_list[-1].max(), cmap='Spectral')[..., :3]], axis=1).astype(np.uint8)
        normal_concat = np.concatenate([normals[0], normals[-1]], axis=1).astype(np.uint8)
        imgs.append(rgb_concat)
        imgs.append(depth_concat)
        imgs.append(normal_concat[..., [2,1,0]])
        imgs.append(matching_image[..., [2,1,0]])
    else: # dens3r_sequence
        for i in range(len(rgbimg)):
            if not valid_image_flag[i]:
                continue
            imgs.append((rgbimg[i]*255).astype(np.uint8))
            ## skip none image, not build in colmap
            imgs.append(colorize(depth_list[i], vmin=0, vmax=depth_list[i].max(), cmap='Spectral')[..., :3])
            imgs.append(normals[i].astype(np.uint8)[..., [2,1,0]])
            imgs.append(matching_images[i][..., [2,1,0]])

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds (including IO)")
    log_msg = f"Reconstruction Success ({len(all_files)} frames). Waiting for visualization."

    return glbfile, imgs, log_msg, gr.Dropdown(choices=frame_filter_choices, value=frame_filter, interactive=True)

def clear_fields():
    """
    Clears the 3D viewer, the stored target_dir, and empties the gallery.
    """
    return None

def update_log():
    """
    Display a quick log message while waiting.
    """
    return "Loading and Reconstructing..."

def update_visualization(
    target_dir, frame_filter, 
    show_cam=True, cam_size=0.5, transparent_cams=False,
):
    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        return None, "No reconstruction available. Please click the Reconstruct button first."

    predictions_path = os.path.join(target_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        return None, f"No reconstruction available at {predictions_path}. Please run 'Reconstruct' first."

    if frame_filter != 'All':
        glbfile = os.path.join(target_dir, frame_filter.split(":")[1].strip() + ".ply")
        if os.path.exists(glbfile):
            return glbfile, "Updating Visualization"
    
    loaded = np.load(predictions_path, allow_pickle=True)
    predictions = {key: loaded[key] for key in loaded.keys()}

    frame_name = frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')
    glbfile = os.path.join(
        target_dir,
        f"scene_recon_{show_cam}_{frame_name}.glb",
    )

    if not os.path.exists(glbfile):
        glbscene = predictions_to_glb(
            predictions,
            conf_thres=0,
            filter_by_frames=frame_filter,
            show_cam=show_cam,
            prediction_mode="Predicted Pointmap"
        )
        glbscene.export(file_obj=glbfile)

    return glbfile, "Updating Visualization"

def main_demo(args):
    server_name, server_port = args.server_name, args.server_port
    
    theme = gr.themes.Default()
    theme.set(
        checkbox_label_background_fill_selected="*button_primary_background_fill",
        checkbox_label_text_color_selected="*button_primary_text_color",
    )
    css_html = """
    .custom-log * {
        font-style: italic;
        font-size: 22px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        font-weight: bold !important;
        color: transparent !important;
        text-align: center !important;
    }
    
    .example-log * {
        font-style: italic;
        font-size: 22px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent !important;
    }
    
    #my_radio .wrap {
        display: flex;
        flex-wrap: nowrap;
        justify-content: center;
        align-items: center;
    }

    #my_radio .wrap label {
        display: flex;
        width: 50%;
        justify-content: center;
        align-items: center;
        margin: 0;
        padding: 10px 0;
        box-sizing: border-box;
    }
    .progress-text { display: none !important; }
    
    .gradio-examples .wrap.svelte-1iyuev5 {
        gap: 10px;
    }
    .gradio-examples {
        min-width: 300px !important;
        height: auto !important;
        border-radius: 8px;
    }
    .gradio-examples video {
        height: 200px !important;
        width: auto !important;
        border-radius: 8px;
    }
    .gradio-examples .example-content {
        padding: 10px;
    }
    """
    with gr.Blocks(theme=theme, css=css_html, analytics_enabled=False) as demo:
        num_images = gr.Textbox(label="num_images", visible=False, value="None")

        target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")
        with gr.Row():
            with gr.Column(scale=2):
                input_video = gr.Video(label="Upload Video", interactive=True)
                input_images = gr.Files(file_count="multiple", label="Upload Images", interactive=True)
                # input_images = gr.Gallery(
                #     label="Upload Images",
                #     columns=4,
                #     height="300px",
                #     show_download_button=True,
                #     object_fit="contain",
                #     preview=True,
                #     interactive=True,  # 允许用户拖拽上传
                # )
                image_gallery = gr.Gallery(
                    label="input images",
                    columns=4,
                    height="300px",
                    show_download_button=True,
                    object_fit="contain",
                    preview=True,
                )
            with gr.Column(scale=4):
                with gr.Column():
                    gr.Markdown("**3D Reconstruction**", height=20)
                    log_output = gr.Markdown(
                        "Please upload a video or images, then click Reconstruct.", elem_classes=["custom-log"]
                    )
                    reconstruction_output = gr.Model3D(height=520, zoom_speed=0.5, pan_speed=0.5)

                with gr.Row():
                    submit_btn = gr.Button("Reconstruct", scale=1, variant="primary")
                    clear_btn = gr.ClearButton(
                        [input_video, input_images, reconstruction_output, log_output, target_dir_output],
                        scale=1,
                    )
                with gr.Row():
                    frame_filter = gr.Dropdown(choices=["All"], value="All", label="Show Points from Frame")
                    with gr.Column():
                        show_cam = gr.Checkbox(label="Show Camera", value=True)
        outgallery = gr.Gallery(label='rgb depth normal matching', columns=4, object_fit="contain", height="100%")
        
        # ---------------------- Examples section ----------------------
        examples = [
            # video, images
            ["examples/stairs_1img.mp4", None],
            ["examples/temple_1img.mp4", None],
            ["examples/sofa_1img.mp4", None],
            ["examples/eth3d_2imgs.mp4", None],
            ["examples/bed_2imgs.mp4", None],
            ["examples/sofa_20imgs.mp4", None],
        ]
        gr.Markdown("Click any video to load an example.", elem_classes=["example-log"])
        
        def example_pipeline(input_video, input_images):
            """
            1) Copy example images to new target_dir
            2) Reconstruct
            3) Return model3D + logs + new_dir + updated dropdown + gallery
            """
            target_dir, image_paths = handle_uploads(input_video, input_images)
            # Always use "All" for frame_filter in examples
            frame_filter = "All"
            glbfile, imgs, log_msg, frame_filter = gradio_demo(
                target_dir, frame_filter
            )
            return glbfile, imgs, log_msg, frame_filter

        gr.Examples(
            examples=examples,
            inputs=[
                input_video,
                input_images,
            ],
            outputs=[
                reconstruction_output,
                outgallery,
                log_output,
                frame_filter,
            ],
            fn=example_pipeline,
            cache_examples=False,
            examples_per_page=10,
            preload=True,
            # elem_id="gradio-examples",
        )
        # -------------------------------------------------------------------------
        
        submit_btn.click(fn=clear_fields, inputs=[], outputs=[reconstruction_output]).then(
            fn=update_log, inputs=[], outputs=[log_output], show_progress="hidden"
        ).then(
            fn=gradio_demo,
            inputs=[
                target_dir_output,
                frame_filter
            ],
            outputs=[reconstruction_output, outgallery, log_output, frame_filter],
        )
        # -------------------------------------------------------------------------
        # Real-time Visualization Updates
        # -------------------------------------------------------------------------
        
        frame_filter.change(
            update_visualization,
            [
                target_dir_output,
                frame_filter,
                show_cam,
            ],
            [reconstruction_output, log_output],
        )
        
        show_cam.change(
            update_visualization,
            [
                target_dir_output,
                frame_filter,
                show_cam,
            ],
            [reconstruction_output, log_output],
        )
        
        # -------------------------------------------------------------------------
        # Auto-update gallery whenever user uploads or changes their files
        # -------------------------------------------------------------------------
        input_video.change(
            fn=update_gallery_on_upload,
            inputs=[input_video, input_images],
            outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
        )
        input_images.change(
            fn=update_gallery_on_upload,
            inputs=[input_video, input_images],
            outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
        )
        demo.queue(max_size=1).launch(share=False, server_name=server_name, server_port=server_port, ssl_verify=False)


def get_args():
    parser = argparse.ArgumentParser("Dens3r Local Gradio Demo")
    parser.add_argument("--server_name", type=str, default="127.0.0.1", 
                        help="server url, default is 127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7688, 
                        help=("will start gradio app on this port (if available). If None, will search for an available port starting at 7860."))
    
    parser.add_argument("--ckpt_path", type=str, required=True, 
                        help="path to the model weights")
    parser.add_argument("--device", type=str, default='cuda', 
                        help="pytorch device")
    
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = get_args()
    model = load_model(args.ckpt_path, args.device) # model is a global param
    main_demo(args)
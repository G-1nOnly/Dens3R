import blenderproc as bproc

import sys
import bpy
from pathlib import Path
sys.path.append(str(Path.home() / 'dev/blender/BlenderToolbox'))
import BlenderToolBox as bt

import argparse
import imagesize
from pathlib import Path
from loguru import logger
# from blcore.world import set_background_color

import cv2
import numpy as np
from PIL import Image


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose  # T44_c2w


def load_all_poses(cam_path, img_dir):
    img_paths = parse_images_from_camera_dict(cam_path, img_dir)
    camera_dict = np.load(cam_path)
    world_mats = [camera_dict[f"world_mat_{idx}"].astype(np.float32) for idx in range(len(img_paths))]
    scale_mats = [camera_dict[f"scale_mat_{idx}"].astype(np.float32) for idx in range(len(img_paths))]

    all_poses = {}
    for img_path, world_mat, scale_mat in zip(img_paths, world_mats, scale_mats):
        P = world_mat @ scale_mat
        P = P[:3, :4]
        K44, T44_c2w = load_K_Rt_from_P(None, P)  # (4, 4)
        all_poses[img_path.stem] = {'K33': K44[:3, :3], 'T44_c2w': T44_c2w}
    return all_poses


def parse_images_from_camera_dict(camera_path, image_dir):
    """Parse images from camera dict, which is useful when there are too many frames.
    
    TODO: images not in the camera dict can be used during validation.
    """
    camera_dict = np.load(camera_path)
    suffix = next(image_dir.iterdir()).suffix
    _sorted_img_keys = sorted([k for k in camera_dict.keys() if 'image_name' in k],
                               key=lambda x: int(x.split('_')[-1]))
    image_paths = [(image_dir / camera_dict[k]).with_suffix(suffix) for k in _sorted_img_keys]
    assert all(map(lambda x: x.exists(), image_paths)), 'Some images are missing!'
    return image_paths


def load_object_aabb(object_anno_path, camera_anno_path) -> np.ndarray:
    obj_scale_mat = np.load(object_anno_path)["scale_mat_0"].astype(np.float32)
    cam_scale_mat = np.load(camera_anno_path)["scale_mat_0"].astype(np.float32)
    
    obj_bbox_min = np.array([-1.0, -1.0, -1.0, 1.0])
    obj_bbox_max = np.array([1.0, 1.0, 1.0, 1.0])
    obj_bbox_min = np.linalg.inv(cam_scale_mat) @ obj_scale_mat @ obj_bbox_min[:, None]
    obj_bbox_max = np.linalg.inv(cam_scale_mat) @ obj_scale_mat @ obj_bbox_max[:, None]
    obj_aabb = np.stack([obj_bbox_min[:3, 0], obj_bbox_max[:3, 0]], 0)
    return obj_aabb


def set_background_color(color):
    assert len(color) == 4
    world = bpy.context.scene.world
    world.node_tree.nodes["Background"].inputs[0].default_value = color
    
    # avoid world color affecting foreground color
    world.cycles_visibility.camera = True
    world.cycles_visibility.diffuse = False
    world.cycles_visibility.glossy = False
    world.cycles_visibility.transmission = False
    world.cycles_visibility.scatter = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, default='data/auto_recon/CO3D_V2/chair/380_45214_90438')
    parser.add_argument('--exp_dir', type=Path, default='outputs/neusfacto-wbg-reg_sep-plane-nerf_co3d-chair-380-45214-90438/neus-facto-wbg-reg_sep-plane-nerf/2023-03-10_005316')
    parser.add_argument('--mesh_relpath', type=str, default='extracted_mesh_res-512_max-component.ply')
    parser.add_argument('--camera_relpath', type=str, default='sfm-transfer_sdfstudio-dff_extend-0.15/cameras_sfm-transfer_norm-obj-side-2.0.npz')
    parser.add_argument('--object_relpath', type=str, default='sfm-transfer_sdfstudio-dff_extend-0.15/objects_sfm-transfer_norm-obj-side-2.0.npz')
    parser.add_argument('--image_reldir', type=str, default='images')
    parser.add_argument('--render_image_name', type=str, default='frame000136.jpg')
    parser.add_argument('--light_mode', choices=['sun', '3point'], default='sun')
    parser.add_argument('--sun_light_angle', type=float, nargs=3, default=[137, -197, 51])
    parser.add_argument('--sun_light_strength', type=float, default=2)
    parser.add_argument('--ambient_light', type=float, nargs=3, default=[0.1, 0.1, 0.1])
    parser.add_argument('--mesh_rgb', type=int, nargs=3, default=[144, 210, 236])
    parser.add_argument('--scale_aabb_to_unit_sphere', action='store_true')   # for NeuS
    # 380: [137, -197, 51]
    
    args = parser.parse_args()
    
    return args


def compute_ground_poision(obj_aabb):
    y = float(obj_aabb[1, 1])
    x = float(obj_aabb[:, 0].mean())
    z = float(obj_aabb[:, 2].mean())
    return (x, y, z)


def scale_aabb_to_unit_sphere(all_poses, obj_aabb):
    obj_diag_len = np.linalg.norm(obj_aabb[1] - obj_aabb[0])
    scale_factor = 2 / obj_diag_len
    for key in all_poses.keys():
        all_poses[key]['T44_c2w'][:3, 3] *= scale_factor
    obj_aabb *= scale_factor


def main():
    # TODO: rotate the world frame to match the blender world (such that default settings work properly)
    args = parse_args()
    mesh_path = args.exp_dir / args.mesh_relpath
    camera_path = args.data_dir / args.camera_relpath
    object_anno_path = args.data_dir / args.object_relpath
    image_dir = args.data_dir / args.image_reldir
    image_paths = sorted(image_dir.iterdir())
    render_image_name = args.render_image_name.split('.')[0]
    save_path = args.exp_dir / f'{render_image_name}.png'
    
    all_poses = load_all_poses(camera_path, image_dir)
    obj_aabb = load_object_aabb(object_anno_path, camera_path)
    if args.scale_aabb_to_unit_sphere:
        logger.info('Scale object AABB to unit sphere...')
        scale_aabb_to_unit_sphere(all_poses, obj_aabb)
    
    print(obj_aabb)
    image_width, image_height = imagesize.get(str(image_paths[0]))
    bproc.init()
    bproc.renderer.set_max_amount_of_samples(300)
    bproc.renderer.set_output_format(enable_transparency=True)
    # bt.blenderInit(image_width, image_height, 300, 1.5, True)
    # bpy.context.scene.render.film_transparent = True
    
    # load object mesh
    obj = bproc.loader.load_obj(str(mesh_path))[0]
    
    # set white bg
    # bt.set_background([1., 1., 1., 1.], is_transparent=False)
    # set_background_color([16.3, 16.3, 16.3, 1.0])
    
    # set material
    RGBA = np.array([*args.mesh_rgb, 255.0]) / 255.0
    # RGBA = (144.0 / 255, 210.0 / 255, 236.0 / 255, 1)
    meshColor = bt.colorObj(RGBA, 0.5, 1.0, 1.0, 0.0, 2.0)
    bt.setMat_plastic(obj.blender_obj, meshColor)
    
    # set invisible plane (shadow catcher)
    ground_position = compute_ground_poision(obj_aabb)
    logger.info(f'Ground position: {ground_position}')
    bt.invisibleGround(location=ground_position, rotation=(90, 0, 0), shadowBrightness=0.9)
    
    # set light
    if args.light_mode == 'sun':
        ## Option2: simple sun light
        lightAngle = args.sun_light_angle
        strength = args.sun_light_strength
        shadowSoftness = 0.3
        sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)
    elif args.light_mode == '3point':
        ## Option1: Three Point Light System
        bt.setLight_threePoints(radius=4, height=10, intensity=1700, softness=6, keyLoc='left')    
    
    ## set ambient light
    bt.setLight_ambient(color=(*args.ambient_light, 1))

    ## set gray shadow to completely white with a threshold (optional but recommended)
    bt.shadowThreshold(alphaThreshold=0.05, interpolationMode='CARDINAL')
    
    # set cameras for each frame (assume all cameras share the same K)
    K_33 = next(iter(all_poses.values()))['K33']
    K_33[0, 1] = 0.0  # blender doesn't support non-zero skew, but load_K_Rt_from_P leads to a tiny skew
    bproc.camera.set_intrinsics_from_K_matrix(K_33, int(image_width), int(image_height))
    
    pose = all_poses[render_image_name]
    T44_c2w_gl = bproc.math.change_source_coordinate_frame_of_transformation_matrix(
        pose['T44_c2w'], ['X', '-Y', '-Z'])
    bproc.camera.add_camera_pose(T44_c2w_gl)
    
    # render
    data = bproc.renderer.render()
    colors = data['colors']
    Image.fromarray(colors[0]).save(save_path)
    logger.info(f'Saved rendered image to {save_path}')
    
    # save blende file for debug
    blend_path = args.exp_dir / 'debug.blend'
    bpy.ops.wm.save_mainfile(filepath=str(blend_path.absolute()))
    logger.info(f'Saved blend file to {blend_path}')


if __name__ == "__main__":
    main()

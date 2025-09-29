# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# base class for implementing datasets
# --------------------------------------------------------
import PIL.Image
import PIL.Image as Image
import numpy as np
import torch
import copy
import random
from scipy.spatial.transform import Rotation

from dust3r.datasets.utils.mast3r_cropping import (extract_correspondences_from_pts3d,
                                            gen_random_crops, in2d_rect, crop_to_homography)

# import mast3r.utils.path_to_dust3r  # noqa
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset, view_name, is_good_type  # noqa
from dust3r.datasets.utils.transforms import ImgNorm
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates, geotrf, depthmap_to_camera_coordinates
import dust3r.datasets.utils.cropping as cropping

class MASt3RBaseStereoViewDataset(BaseStereoViewDataset):
    def __init__(self, *,  # only keyword arguments
                 split=None,
                 resolution=None,  # square_size or (width, height) or list of [(width,height), ...]
                 transform=ImgNorm,
                 aug_crop=False,
                 aug_swap=True,
                 aug_monocular=False,
                 aug_portrait_or_landscape=True,  # automatic choice between landscape/portrait when possible
                 aug_rot90=False,
                 aug_blur=True,
                 n_corres=0,
                 nneg=0,
                 n_tentative_crops=4,
                 transform_view2_normals=False,
                 seed=None):
        super().__init__(split=split, resolution=resolution, transform=transform, aug_crop=aug_crop, seed=seed)
        self.is_metric_scale = False  # by default a dataset is not metric scale, subclasses can overwrite this

        self.aug_swap = aug_swap
        self.aug_monocular = aug_monocular
        self.aug_portrait_or_landscape = aug_portrait_or_landscape
        self.aug_rot90 = aug_rot90
        self.transform_view2_normals = transform_view2_normals
        self.aug_blur = aug_blur
        
        self.defalut_mono_dataset_prob = 0.05

        self.n_corres = n_corres
        self.nneg = nneg
        assert self.n_corres == 'all' or isinstance(self.n_corres, int) or (isinstance(self.n_corres, list) and len(
            self.n_corres) == self.num_views), f"Error, n_corres should either be 'all', a single integer or a list of length {self.num_views}"
        assert self.nneg == 0 or self.n_corres != 'all'
        self.n_tentative_crops = n_tentative_crops

    def _swap_view_aug(self, views):
        if self._rng.random() < 0.5:
            views.reverse()

    def _crop_resize_if_necessary(self, image, depthmap, intrinsics, resolution, 
                                  rng=None, info=None, offset_factor=0.5, mask=None, normalmap=None, seg_label=None):
        """ This function:
            - first downsizes the image with LANCZOS inteprolation,
                which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # transpose the resolution if necessary
        W, H = image.size  # new size
        assert resolution[0] >= resolution[1]
        if H > 1.1 * W:
            # image is portrait mode
            resolution = resolution[::-1]
        elif 0.9 < H / W < 1.1 and resolution[0] != resolution[1]:
            # image is square, so we chose (portrait, landscape) randomly
            if rng.integers(2) and self.aug_portrait_or_landscape:
                resolution = resolution[::-1]

        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        # image, depthmap, intrinsics = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)
        if seg_label is None:
            image, depthmap, intrinsics, normalmap = \
                cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution, normalmap=normalmap)
        else:
            image, depthmap, intrinsics, normalmap, seg_label = \
                cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution, normalmap=normalmap, seg_label=seg_label)

        # actual cropping (if necessary) with bilinear interpolation
        # offset_factor = 0.5
        intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=offset_factor)
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        # image, depthmap, intrinsics2 = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)
        if seg_label is None:
            image, depthmap, intrinsics2, normalmap = \
                cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox, normalmap=normalmap)
        else:
            image, depthmap, intrinsics2, normalmap, seg_label = \
                cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox, normalmap=normalmap, seg_label=seg_label)
        if seg_label is None:
            return image, depthmap, intrinsics2, normalmap
        else:
            return image, depthmap, intrinsics2, normalmap, seg_label

    def generate_crops_from_pair(self, view1, view2, resolution, aug_crop_arg, n_crops=4, rng=np.random):
        views = [view1, view2]

        if aug_crop_arg is False:
            # compatibility
            for i in range(2):
                view = views[i]
                # view['img'], view['depthmap'], view['camera_intrinsics'] = self._crop_resize_if_necessary(view['img'],
                #                                                                                           view['depthmap'],
                #                                                                                           view['camera_intrinsics'],
                #                                                                                           resolution,
                #                                                                                           rng=rng)
                view['img'], view['depthmap'], view['camera_intrinsics'], view['normalmap'] = \
                    self._crop_resize_if_necessary(view['img'], view['depthmap'], view['camera_intrinsics'], resolution, rng=rng, normalmap=view['normalmap'])
                view['pts3d'], view['valid_mask'] = depthmap_to_absolute_camera_coordinates(view['depthmap'],
                                                                                            view['camera_intrinsics'],
                                                                                            view['camera_pose'])
            return

        # extract correspondences
        corres = extract_correspondences_from_pts3d(*views, target_n_corres=None, rng=rng)

        # generate 4 random crops in each view
        view_crops = []
        crops_resolution = []
        corres_msks = []
        for i in range(2):

            if aug_crop_arg == 'auto':
                S = min(views[i]['img'].size)
                R = min(resolution)
                aug_crop = S * (S - R) // R
                aug_crop = max(.1 * S, aug_crop)  # for cropping: augment scale of at least 10%, and more if possible
            else:
                aug_crop = aug_crop_arg

            # tranpose the target resolution if necessary
            assert resolution[0] >= resolution[1]
            W, H = imsize = views[i]['img'].size
            crop_resolution = resolution
            if H > 1.1 * W:
                # image is portrait mode
                crop_resolution = resolution[::-1]
            elif 0.9 < H / W < 1.1 and resolution[0] != resolution[1]:
                # image is square, so we chose (portrait, landscape) randomly
                if rng.integers(2):
                    crop_resolution = resolution[::-1]

            crops = gen_random_crops(imsize, n_crops, crop_resolution, aug_crop=aug_crop, rng=rng)
            view_crops.append(crops)
            crops_resolution.append(crop_resolution)

            # compute correspondences
            corres_msks.append(in2d_rect(corres[i], crops))

        # compute IoU for each
        intersection = np.float32(corres_msks[0]).T @ np.float32(corres_msks[1])
        # select best pair of crops
        best = np.unravel_index(intersection.argmax(), (n_crops, n_crops))
        crops = [view_crops[i][c] for i, c in enumerate(best)]

        # crop with the homography
        for i in range(2):
            view = views[i]
            imsize, K_new, R, H = crop_to_homography(view['camera_intrinsics'], crops[i], crops_resolution[i])
            # imsize, K_new, H = upscale_homography(imsize, resolution, K_new, H)

            # update camera params
            K_old = view['camera_intrinsics']
            view['camera_intrinsics'] = K_new
            view['camera_pose'] = view['camera_pose'].copy()
            view['camera_pose'][:3, :3] = view['camera_pose'][:3, :3] @ R

            # apply homography to image and depthmap
            homo8 = (H / H[2, 2]).ravel().tolist()[:8]
            view['img'] = view['img'].transform(imsize, Image.Transform.PERSPECTIVE,
                                                homo8,
                                                resample=Image.Resampling.BICUBIC)

            depthmap2 = depthmap_to_camera_coordinates(view['depthmap'], K_old)[0] @ R[:, 2]
            view['depthmap'] = np.array(Image.fromarray(depthmap2).transform(
                imsize, Image.Transform.PERSPECTIVE, homo8))
            if 'normalmap' in view:
                normalmap2 = (R @ view['normalmap'].reshape(-1, 3).T).T.reshape(view['normalmap'].shape)
                normalmap_slice1 = np.array(Image.fromarray(normalmap2[..., 0]).transform(
                    imsize, Image.Transform.PERSPECTIVE, homo8))
                normalmap_slice2 = np.array(Image.fromarray(normalmap2[..., 1]).transform(
                    imsize, Image.Transform.PERSPECTIVE, homo8))
                normalmap_slice3 = np.array(Image.fromarray(normalmap2[..., 2]).transform(
                    imsize, Image.Transform.PERSPECTIVE, homo8))
                view['normalmap'] = np.stack([normalmap_slice1, normalmap_slice2, normalmap_slice3], axis=-1)

            if 'track_labels' in view:
                # convert from uint64 --> uint32, because PIL.Image cannot handle uint64
                mapping, track_labels = np.unique(view['track_labels'], return_inverse=True)
                track_labels = track_labels.astype(np.uint32).reshape(view['track_labels'].shape)

                # homography transformation
                res = np.array(Image.fromarray(track_labels).transform(imsize, Image.Transform.PERSPECTIVE, homo8))
                view['track_labels'] = mapping[res]  # mapping back to uint64

            # recompute 3d points from scratch
            view['pts3d'], view['valid_mask'] = depthmap_to_absolute_camera_coordinates(view['depthmap'],
                                                                                        view['camera_intrinsics'],
                                                                                        view['camera_pose'])
            
    def __getNewitem__(self, idx):
        idx_update = random.randint(0, self.__len__()-1)
        if isinstance(idx, tuple):
            # the idx is specifying the aspect-ratio
            idx, ar_idx = idx
            return self.__getitem__((idx_update, ar_idx))
        else:
            return self.__getitem__(idx_update)

    def __getitem__(self, idx):
        idx_copy = copy.deepcopy(idx)
        if isinstance(idx, tuple):
            # the idx is specifying the aspect-ratio
            idx, ar_idx = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0

        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, '_rng'):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # over-loaded code
        resolution = self._resolutions[ar_idx]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        views = self._get_views(idx, resolution, self._rng)
        assert len(views) == self.num_views
        
        # set low quality's dataset's normal to zero
        dataset_type = views[0]["dataset"]
        if dataset_type in [
            # "ABO",
            # "ACENNRDatasetEasySceneRawData",
            # "ACENNRDatasetHQSceneRawData",
            # "AceNNRRaw",
            # "AceNNR_v1",
            # "Behavior_v1",
            # "BlendedMVS",
            # "GObjaverse",
            # "GObjaverseXL",
            # "GTASFM",
            # "GTAV",
            # "HyperSim",
            # "ScanNet++",
            # "Spring",
            # "StaticThings3D",
            # "Structured3D",
            # "Infinigen",
            # "IRS",
            # "MatrixCity",
            "UnrealStereo4K",
            # "UrbanSyn",
            # "VirtureKitti",
            
            "Argoverse1Stereo",
            "arkitscenes",
            "Co3d_v2",
            "DIMLIndoorDataset",
            "DL3DV",
            "Habitat",
            "IHome_panorama",
            "IHome_v1",
            "InteriorNet",
            "KenBurns",
            "MegaDepth",
            "MonoDepth",
            "NianticMapFree",
            "Objaverse_v1",
            "Synthia",
            "Tartanair",
            "TaskonomyDataset",
            "Waymo",
            "wildrgbd",
        ]:
            views[0]["normalmap"] = np.zeros_like(views[0]["normalmap"])
            views[1]["normalmap"] = np.zeros_like(views[1]["normalmap"])
        
        # apply motion blur
        if self.aug_blur:
            for v, view in enumerate(views):
                view['img'] = motion_blur(view['img'])

        for v, view in enumerate(views):
            assert 'pts3d' not in view, f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
            view['idx'] = (idx, ar_idx, v)
            view['is_metric_scale'] = self.is_metric_scale

            assert 'camera_intrinsics' in view
            if 'camera_pose' not in view:
                view['camera_pose'] = np.full((4, 4), np.nan, dtype=np.float32)
            else:
                assert np.isfinite(view['camera_pose']).all(), f'NaN in camera pose for view {view_name(view)}'
            assert 'pts3d' not in view
            assert 'valid_mask' not in view
            assert np.isfinite(view['depthmap']).all(), f'NaN in depthmap for view {view_name(view)}'

            pts3d, pts3d_view, valid_mask = depthmap_to_absolute_camera_coordinates(**view, return_pts_view=True)

            view['pts3d'] = pts3d
            view['pts3d_view'] = pts3d_view
            view['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)

        self.generate_crops_from_pair(views[0], views[1], resolution=resolution,
                                      aug_crop_arg=self.aug_crop,
                                      n_crops=self.n_tentative_crops,
                                      rng=self._rng)
        for v, view in enumerate(views):
            # encode the image
            width, height = view['img'].size
            view['true_shape'] = np.int32((height, width))
            view['img'] = self.transform(view['img'])
            # Pixels for which depth is fundamentally undefined
            view['sky_mask'] = (view['depthmap'] < 0)

        if self.aug_swap:
            self._swap_view_aug(views)

        if self.aug_monocular:
            if self._rng.random() < self.aug_monocular:
                views = [copy.deepcopy(views[0]) for _ in range(len(views))]

        # automatic extraction of correspondences from pts3d + pose
        if self.n_corres > 0 and ('corres' not in view):
            corres1, corres2, valid = extract_correspondences_from_pts3d(*views, self.n_corres,
                                                                         self._rng, nneg=self.nneg)
            views[0]['corres'] = corres1
            views[1]['corres'] = corres2
            views[0]['valid_corres'] = valid
            views[1]['valid_corres'] = valid
            
            # check if valid:
            if "seg_label" not in views[0]:
                if valid.sum() < 10:
                    print(f"Valid corres less than 10!")
                    return self.__getNewitem__(idx_copy)

        if self.aug_rot90 is False:
            pass
        elif self.aug_rot90 == 'same':
            rotate_90(views, k=self._rng.choice(4))
        elif self.aug_rot90 == 'diff':
            rotate_90(views[:1], k=self._rng.choice(4))
            rotate_90(views[1:], k=self._rng.choice(4))
        else:
            raise ValueError(f'Bad value for {self.aug_rot90=}')

        # check data-types metric_scale
        for v, view in enumerate(views):
            if 'corres' not in view:
                view['corres'] = np.full((self.n_corres, 2), np.nan, dtype=np.float32)

            # check all datatypes
            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"
            K = view['camera_intrinsics']

            # check shapes
            assert view['depthmap'].shape == view['img'].shape[1:]
            assert view['depthmap'].shape == view['pts3d'].shape[:2]
            assert view['depthmap'].shape == view['valid_mask'].shape

        # just for debug vis:
        # for view_idx in range(2):
        #     import cv2
        #     from dust3r.utils.image import normal2color
        #     normalmap = views[view_idx]["normalmap"]
        #     normalmap_color = normal2color(normalmap)
        #     cv2.imshow(f"normal_{view_idx}_ori", normalmap_color[..., [2,1,0]])
        
        # for filter bad normal GT data:
        if 'normalmap' in views[0] and 'normalmap' in views[1]:
            for view in views:
                pts_view_space = copy.deepcopy(view['pts3d_view']) # H,W,3
                normal_view_space = copy.deepcopy(view['normalmap']) # H,W,3
                valid_mask = copy.deepcopy(view['valid_mask']) # H,W
                
                pts_direction = (pts_view_space) / (np.linalg.norm(pts_view_space, axis=-1, keepdims=True) + 1e-12)
                normal_view_space[..., 1] = -1. * normal_view_space[..., 1]
                dir_dot = (normal_view_space * pts_direction).sum(axis=-1)

                dir_invalid_mask = dir_dot > np.cos(89 * np.pi / 180)
                dir_invalid_mask_extend = dir_invalid_mask[..., None].repeat(3, -1)
                
                normal_view_space[dir_invalid_mask_extend] = 0.
                normal_view_space[..., 1] = -1. * normal_view_space[..., 1]
                view['normalmap'] = normal_view_space
                view['valid_mask'] = valid_mask & (~dir_invalid_mask)
                # check if valid:
                if view['valid_mask'].sum() < 50:
                    print("Valid normal less than 50!")
                    return self.__getNewitem__(idx_copy)
        views[0].pop("pts3d_view")
        views[1].pop("pts3d_view")

        # last thing done!
        for view in views:
            # transpose to make sure all views are the same size
            # transpose_to_landscape(view)
            transpose_to_landscape_no_flip(view, self._rng)
            # this allows to check whether the RNG is is the same state each time
            view['rng'] = int.from_bytes(self._rng.bytes(4), 'big')
            
        # transform normal in view2 to view1 axis!
        if self.transform_view2_normals:
            if 'normalmap' in views[0] and 'normalmap' in views[1]:
                c2w0 = views[0]['camera_pose']
                c2w1 = views[1]['camera_pose']
                rotation_1_to_0 = c2w0[:3, :3].T @ c2w1[:3, :3]
                rotation_1_to_0 = np.diag([1., -1., 1.]) @ rotation_1_to_0 @ np.diag([1., -1., 1.])
                ori_shape = views[1]['normalmap'].shape
                views[0]['normalmap_transformed'] = views[0]['normalmap']
                views[1]['normalmap_transformed'] = (rotation_1_to_0 @ views[1]['normalmap'].reshape(-1, 3).T).T.reshape(ori_shape)

        return views

def transpose_to_landscape_no_flip(view, rng, revert=False):
    height, width = view['true_shape']

    if width < height:
        if revert:
            height, width = width, height
        
        # 随机顺时针/逆时针旋转
        if rng.random() > 0.5:
            k = 1
        else:
            k = -1
        rotate_90_single_view(view, k=k)

def transpose_to_landscape(view, revert=False):
    height, width = view['true_shape']

    if width < height:
        if revert:
            height, width = width, height

        # rectify portrait to landscape
        assert view['img'].shape == (3, height, width)
        view['img'] = view['img'].swapaxes(1, 2)

        assert view['valid_mask'].shape == (height, width)
        view['valid_mask'] = view['valid_mask'].swapaxes(0, 1)

        assert view['sky_mask'].shape == (height, width)
        view['sky_mask'] = view['sky_mask'].swapaxes(0, 1)

        assert view['depthmap'].shape == (height, width)
        view['depthmap'] = view['depthmap'].swapaxes(0, 1)

        assert view['pts3d'].shape == (height, width, 3)
        view['pts3d'] = view['pts3d'].swapaxes(0, 1)
        
        if 'normalmap' in view:
            assert view['normalmap'].shape == (height, width, 3)
            view['normalmap'] = view['normalmap'].swapaxes(0, 1)
            shape_bak = view['normalmap'].shape
            swap_xy_matrix = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]])
            view['normalmap'] = (swap_xy_matrix @ (view['normalmap'].reshape(-1, 3).T)).T.reshape(shape_bak)

        # transpose x and y pixels
        view['camera_intrinsics'] = view['camera_intrinsics'][[1, 0, 2]]

        # transpose correspondences x and y
        view['corres'] = view['corres'][:, [1, 0]]

def rotate_90_single_view(view, k=1):
    from scipy.spatial.transform import Rotation
    RT = np.eye(4, dtype=np.float32)
    RT[:3, :3] = Rotation.from_euler('z', 90 * k, degrees=True).as_matrix()

    view['img'] = torch.rot90(view['img'], k=k, dims=(-2, -1))  # WARNING!! dims=(-1,-2) != dims=(-2,-1)
    view['depthmap'] = np.rot90(view['depthmap'], k=k).copy()
    if 'normalmap' in view:
        normalmap_tmp = np.rot90(view['normalmap'], k=k).copy()
        view['normalmap'] = (RT[:3, :3] @ normalmap_tmp.reshape(-1, 3).T).T.reshape(normalmap_tmp.shape)
    view['camera_pose'] = view['camera_pose'] @ RT

    RT2 = np.eye(3, dtype=np.float32)
    RT2[:2, :2] = RT[:2, :2] * ((1, -1), (-1, 1))
    H, W = view['depthmap'].shape
    if k % 4 == 0:
        pass
    elif k % 4 == 1:
        # top-left (0,0) pixel becomes (0,H-1)
        RT2[:2, 2] = (0, H - 1)
    elif k % 4 == 2:
        # top-left (0,0) pixel becomes (W-1,H-1)
        RT2[:2, 2] = (W - 1, H - 1)
    elif k % 4 == 3:
        # top-left (0,0) pixel becomes (W-1,0)
        RT2[:2, 2] = (W - 1, 0)
    else:
        raise ValueError(f'Bad value for {k=}')

    view['camera_intrinsics'][:2, 2] = geotrf(RT2, view['camera_intrinsics'][:2, 2])
    if k % 2 == 1:
        K = view['camera_intrinsics']
        np.fill_diagonal(K, K.diagonal()[[1, 0, 2]])

    pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)
    view['pts3d'] = pts3d
    view['valid_mask'] = np.rot90(view['valid_mask'], k=k).copy()
    view['sky_mask'] = np.rot90(view['sky_mask'], k=k).copy()

    view['corres'] = geotrf(RT2, view['corres']).round().astype(view['corres'].dtype)
    view['true_shape'] = np.int32((H, W))

def rotate_90(views, k=1):
    for view in views:
        rotate_90_single_view(view, k=k)
        
def motion_blur(img):
    import os
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
    import albumentations as alb
    transform = alb.Compose([
            alb.MotionBlur(
                blur_limit=(3, 11),
                # angle_range=(0, 360),
                # direction_range=(-1.0, 1.0),
                # allow_shifted=True,
                p=.05,
            )
        ])
    img_transformed = transform(image=np.array(img))['image']
    return Image.fromarray(img_transformed)

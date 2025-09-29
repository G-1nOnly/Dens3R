# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# post process function for all heads: extract 3D points/confidence from output
# --------------------------------------------------------
import torch


def postprocess(out, depth_mode, conf_mode):
    """
    extract 3D points/confidence from prediction head output
    """
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,3
    res = dict(pts3d=reg_dense_depth(fmap[:, :, :, 0:3], mode=depth_mode))

    if conf_mode is not None:
        res['conf'] = reg_dense_conf(fmap[:, :, :, 3], mode=conf_mode)
    return res

def postprocess_mask(out, depth_mode, conf_mode):
    # out: B,2,H,W
    # out = torch.nn.functional.softmax(out, dim=1)
    mask_pred = out.permute(0, 2, 3, 1) # B,H,W,1
    # mask_pred = mask_pred.argmax(dim=1)
    res = dict(mask=mask_pred)
    return res

def reg_dense_depth(xyz, mode):
    """
    extract 3D points from prediction head output
    """
    mode, vmin, vmax = mode

    no_bounds = (vmin == -float('inf')) and (vmax == float('inf'))
    assert no_bounds

    if mode == 'linear':
        if no_bounds:
            return xyz  # [-inf, +inf]
        return xyz.clip(min=vmin, max=vmax)

    # distance to origin
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)

    if mode == 'square':
        return xyz * d.square()

    if mode == 'exp':
        return xyz * torch.expm1(d)

    raise ValueError(f'bad {mode=}')


def reg_dense_conf(x, mode):
    """
    extract confidence from prediction head output
    """
    mode, vmin, vmax = mode
    if mode == 'exp':
        return vmin + x.exp().clip(max=vmax-vmin)
    if mode == 'sigmoid':
        return (vmax - vmin) * torch.sigmoid(x) + vmin
    raise ValueError(f'bad {mode=}')


def reg_dense_normal(normal, mode=None):
    # TODO: mode is not used yet
    mode, vmin, vmax = 'linear', -1.0, 1.0
    
    if mode == "linear":
        normal = normal.clip(min=vmin, max=vmax)
    else:
        # distance to origin
        d = normal.norm(dim=-1, keepdim=True)
        normal = normal / d.clip(min=1e-8)
        if mode == 'square':
            normal = normal * d.square()
        if mode == 'exp':
            normal = normal * torch.expm1(d)
        else:
            raise NotImplementedError(f"normal mode: {mode} not implemented")
    
    # normalize normal
    norm_x, norm_y, norm_z = torch.split(normal, 1, dim=-1)
    norm = torch.sqrt(norm_x**2.0 + norm_y**2.0 + norm_z**2.0)
    norm = norm.clip(min=1e-12)
    normal = torch.cat([norm_x / norm, norm_y / norm, norm_z / norm], dim=-1)
    
    return normal

def reg_dense_kappa(kappa, mode):
    # mode is not used yet
    min_kappa = 0.01
    kappa = torch.nn.functional.elu(kappa) + 1.0 + min_kappa
    
    return kappa
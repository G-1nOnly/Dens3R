# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# MASt3R model class
# --------------------------------------------------------

# Modifications Copyright (C) <Alibaba Group>
# Changes: model definition
# This is an adaptation and is distributed under the same license (CC BY-NC-SA 4.0).
# SPDX-License-Identifier: CC-BY-NC-SA-4.0(non-commercial use only)

import os
import torch
import torch.nn.functional as F

from mast3r.catmlp_dpt_head import mast3r_head_factory
import mast3r.utils.path_to_dust3r  # noqa
from dust3r.model import AsymmetricCroCo3DStereo  # noqa
from dust3r.utils.misc import transpose_to_landscape  # noqa
from dust3r.utils.misc import freeze_all_params


inf = float('inf')


def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class AsymmetricMASt3R(AsymmetricCroCo3DStereo):
    def __init__(self, desc_mode=('norm'), desc_conf=False, desc_conf_mode=None, use_normal=True,
                 has_mask=False, freeze_enc_dec=False, **kwargs):
        self.desc_mode = desc_mode
        self.desc_conf = desc_conf
        self.desc_conf_mode = desc_conf_mode
        self.use_normal = use_normal
        
        self.has_mask = has_mask
        
        super().__init__(**kwargs)

    def freeze_encoder_decoder(self):
        to_be_frozen = [
            self.patch_embed, 
            self.enc_blocks, self.enc_norm,
            self.decoder_embed, self.dec_blocks, self.dec_norm,
        ]
        freeze_all_params(to_be_frozen)
        
    def freeze_matching_net(self):
        for name, param in self.downstream_head1.named_parameters():
            if "head_local_features" in name:
                param.requires_grad = False
        
    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        return super().load_state_dict(new_ckpt, **kw)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricMASt3R, cls).from_pretrained(pretrained_model_name_or_path, **kw)
        
    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, normal_mode, patch_size, img_size, **kw):
        assert img_size[0] % patch_size == 0 and img_size[
            1] % patch_size == 0, f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.normal_mode = normal_mode
        self.conf_mode = conf_mode
        if self.desc_conf_mode is None:
            self.desc_conf_mode = conf_mode
        
        assert output_mode == "unify"
        
        self.ptsmatch_local_head = mast3r_head_factory(head_type='catmlp+dpt', 
                                                        output_mode='pts3d+desc24', 
                                                        net=self, has_conf=bool(conf_mode), has_mask=False)
        self.local_head = transpose_to_landscape(self.ptsmatch_local_head, activate=landscape_only)
        
        self.ptsmatch_global_head = mast3r_head_factory(head_type='catmlp+dpt', 
                                                        output_mode='pts3d+desc24', 
                                                        net=self, has_conf=bool(conf_mode), has_mask=False)
        self.global_head = transpose_to_landscape(self.ptsmatch_global_head, activate=landscape_only)
        
        self.normal_local_head = mast3r_head_factory(head_type='dpt', 
                                                        output_mode='normal', 
                                                        net=self, has_conf=False, has_mask=False)
        self.normal_head = transpose_to_landscape(self.normal_local_head, activate=landscape_only)

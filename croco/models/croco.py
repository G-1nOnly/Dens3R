# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# CroCo model during pretraining
# --------------------------------------------------------


import math
import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True # for gpu >= Ampere and pytorch >= 1.12
from functools import partial

from models.blocks import Block, DecoderBlock, PatchEmbed, ATMBlock
from models.pos_embed import get_2d_sincos_pos_embed, RoPE2D 
from models.masking import RandomMask


class CroCoNet(nn.Module):

    def __init__(self,
                 img_size=224,           # input image size
                 patch_size=16,          # patch_size 
                 mask_ratio=0.9,         # ratios of masked tokens 
                 enc_embed_dim=768,      # encoder feature dimension
                 enc_depth=12,           # encoder depth 
                 enc_num_heads=12,       # encoder number of heads in the transformer block 
                 dec_embed_dim=512,      # decoder feature dimension 
                 dec_depth=8,            # decoder depth 
                 dec_num_heads=16,       # decoder number of heads in the transformer block 
                 mlp_ratio=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 norm_im2_in_dec=True,   # whether to apply normalization of the 'memory' = (second image) in the decoder 
                 pos_embed='cosine',     # positional embedding (either cosine or RoPE100)
                ):
                
        super(CroCoNet, self).__init__()
                
        # patch embeddings  (with initialization done as in MAE)
        self._set_patch_embed(img_size, patch_size, enc_embed_dim)

        # mask generations
        self._set_mask_generator(self.patch_embed.num_patches, mask_ratio)

        self.pos_embed = pos_embed
        self.patch_size = patch_size
        if pos_embed=='cosine':
            # positional embedding of the encoder 
            enc_pos_embed = get_2d_sincos_pos_embed(enc_embed_dim, int(self.patch_embed.num_patches**.5), n_cls_token=0)
            self.register_buffer('enc_pos_embed', torch.from_numpy(enc_pos_embed).float())
            # positional embedding of the decoder  
            dec_pos_embed = get_2d_sincos_pos_embed(dec_embed_dim, int(self.patch_embed.num_patches**.5), n_cls_token=0)
            self.register_buffer('dec_pos_embed', torch.from_numpy(dec_pos_embed).float())
            # pos embedding in each block
            self.rope = None # nothing for cosine 
        elif pos_embed.startswith('RoPE'): # eg RoPE100 
            self.enc_pos_embed = None # nothing to add in the encoder with RoPE
            self.dec_pos_embed = None # nothing to add in the decoder with RoPE
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(pos_embed[len('RoPE'):])
            self.rope = RoPE2D(freq=freq)
        elif pos_embed.startswith("DINO"):
            self.interpolate_offset = None
            self.rope = None
            self.interpolate_antialias = False
            self.enc_pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, enc_embed_dim)) # learnable
            self.dec_pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, dec_embed_dim)) # learnable
            # test: add RoPE with enc_pos_embed together
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            self.rope = RoPE2D(freq=100)
        elif pos_embed.startswith("ATMRoPE"):
            self.enc_pos_embed = None # nothing to add in the encoder with RoPE
            self.dec_pos_embed = None # nothing to add in the decoder with RoPE
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(pos_embed[len('ATMRoPE'):])
            self.rope = RoPE2D(freq=freq)
            # fxz: ATM block
            self.atm_block = ATMBlock(enc_embed_dim, enc_num_heads, qkv_bias=False, attn_drop=0., drop=0.)
        else:
            raise NotImplementedError('Unknown pos_embed '+pos_embed)

        # transformer for the encoder 
        self.enc_depth = enc_depth
        self.enc_embed_dim = enc_embed_dim
        self.enc_blocks = nn.ModuleList([
            Block(enc_embed_dim, enc_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, rope=self.rope)
            for i in range(enc_depth)])
        self.enc_norm = norm_layer(enc_embed_dim)
        
        # masked tokens 
        self._set_mask_token(dec_embed_dim)

        # decoder 
        self._set_decoder(enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec)
        
        # prediction head 
        self._set_prediction_head(dec_embed_dim, patch_size)
        
        # initializer weights
        self.initialize_weights()           

    # according to DINOv2
    def interpolate_pose_encoding_dust3r(self, x, pos_embed_code, img_w, img_h, fuzzy_pe=True):
        # x shape: N, patched, enc_dim or dec_dim
        previous_dtype = x.dtype
        B, patch_num, dim = x.shape
        # when using grid_sample this is not neccesary:
        # if patch_num == pos_embed_code.shape[1] and img_w == img_h:
        #     return pos_embed_code
        w0 = img_w // self.patch_size
        h0 = img_h // self.patch_size
        
        kwargs = {}
        M = int(math.sqrt(pos_embed_code.shape[1]))  # Recover the number of patches in each dimension
        assert pos_embed_code.shape[1] == M * M
        
        # if self.interpolate_offset:
        #     # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
        #     # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
        #     sx = float(w0 + self.interpolate_offset) / M
        #     sy = float(h0 + self.interpolate_offset) / M
        #     kwargs["scale_factor"] = (sx, sy)
        # else:
        #     # Simply specify an output size instead of a scale factor
        #     kwargs["size"] = (w0, h0)
        # pos_embed_code = nn.functional.interpolate(
        #     pos_embed_code.reshape(1, M, M, dim).permute(0, 3, 1, 2),
        #     mode="bicubic",
        #     antialias=self.interpolate_antialias,
        #     **kwargs,
        # )
        
        # add fuzzy positional encoding [grid_sample]; pos_embed_code: B,M*M,dim
        grid_h = torch.linspace(-1, 1, h0)
        grid_w = torch.linspace(-1, 1, w0)
        mesh_h, mesh_w = torch.meshgrid(grid_h, grid_w, indexing="ij")
        grid_pos = torch.stack((mesh_h, mesh_w), dim=-1)[None, ...].to(x.device) # B,Hout,Wout,2
        
        # add noise to grid_pos here: Fuzzy_Positional_Encoding
        if fuzzy_pe:
            h_interval, w_interval = 2. / h0, 2. / w0
            fuzzy_noise = torch.rand_like(grid_pos) # [0,1)
            fuzzy_noise[..., -2] = fuzzy_noise[..., -2] * h_interval - h_interval / 2.
            fuzzy_noise[..., -1] = fuzzy_noise[..., -1] * w_interval - w_interval / 2.
            grid_pos = grid_pos + fuzzy_noise
        
        pos_embed_code = nn.functional.grid_sample(pos_embed_code.reshape(1, M, M, dim).permute(0, 3, 1, 2), grid_pos, 
                                                   mode="bilinear",
                                                   padding_mode="zeros",
                                                   align_corners=False)
        
        # assert (w0, h0) == pos_embed_code.shape[-2:]
        assert (h0, w0) == pos_embed_code.shape[-2:]
        pos_embed_code = pos_embed_code.permute(0, 2, 3, 1).view(1, -1, dim).repeat(B, 1, 1)
        return pos_embed_code.to(previous_dtype)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, enc_embed_dim)

    def _set_mask_generator(self, num_patches, mask_ratio):
        self.mask_generator = RandomMask(num_patches, mask_ratio)
        
    def _set_mask_token(self, dec_embed_dim):
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_embed_dim))
        
    def _set_decoder(self, enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        # transfer from encoder to decoder 
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # transformer for the decoder 
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, norm_mem=norm_im2_in_dec, rope=self.rope)
            for i in range(dec_depth)])
        # final norm layer 
        self.dec_norm = norm_layer(dec_embed_dim)
        
    def _set_prediction_head(self, dec_embed_dim, patch_size):
         self.prediction_head = nn.Linear(dec_embed_dim, patch_size**2 * 3, bias=True)
        
        
    def initialize_weights(self):
        # patch embed 
        self.patch_embed._init_weights()
        # mask tokens
        if self.mask_token is not None: torch.nn.init.normal_(self.mask_token, std=.02)
        # linears and layer norms
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def _encode_image(self, image, do_mask=False, return_all_blocks=False):
        """
        image has B x 3 x img_size x img_size 
        do_mask: whether to perform masking or not
        return_all_blocks: if True, return the features at the end of every block 
                           instead of just the features from the last block (eg for some prediction heads)
        """
        # embed the image into patches  (x has size B x Npatches x C) 
        # and get position if each return patch (pos has size B x Npatches x 2)
        x, pos = self.patch_embed(image)              
        # add positional embedding without cls token  
        if self.enc_pos_embed is not None: 
            x = x + self.enc_pos_embed[None,...]
        # apply masking 
        B,N,C = x.size()
        if do_mask:
            masks = self.mask_generator(x)
            x = x[~masks].view(B, -1, C)
            posvis = pos[~masks].view(B, -1, 2)
        else:
            B,N,C = x.size()
            masks = torch.zeros((B,N), dtype=bool)
            posvis = pos
        # now apply the transformer encoder and normalization        
        if return_all_blocks:
            out = []
            for blk in self.enc_blocks:
                x = blk(x, posvis)
                out.append(x)
            out[-1] = self.enc_norm(out[-1])
            return out, pos, masks
        else:
            for blk in self.enc_blocks:
                x = blk(x, posvis)
            x = self.enc_norm(x)
            return x, pos, masks
 
    def _decoder(self, feat1, pos1, masks1, feat2, pos2, return_all_blocks=False):
        """
        return_all_blocks: if True, return the features at the end of every block 
                           instead of just the features from the last block (eg for some prediction heads)
                           
        masks1 can be None => assume image1 fully visible 
        """
        # encoder to decoder layer 
        visf1 = self.decoder_embed(feat1)
        f2 = self.decoder_embed(feat2)
        # append masked tokens to the sequence
        B,Nenc,C = visf1.size()
        if masks1 is None: # downstreams
            f1_ = visf1
        else: # pretraining 
            Ntotal = masks1.size(1)
            f1_ = self.mask_token.repeat(B, Ntotal, 1).to(dtype=visf1.dtype)
            f1_[~masks1] = visf1.view(B * Nenc, C)
        # add positional embedding
        if self.dec_pos_embed is not None:
            f1_ = f1_ + self.dec_pos_embed
            f2 = f2 + self.dec_pos_embed
        # apply Transformer blocks
        out = f1_
        out2 = f2 
        if return_all_blocks:
            _out, out = out, []
            for blk in self.dec_blocks:
                _out, out2 = blk(_out, out2, pos1, pos2)
                out.append(_out)
            out[-1] = self.dec_norm(out[-1])
        else:
            for blk in self.dec_blocks:
                out, out2 = blk(out, out2, pos1, pos2)
            out = self.dec_norm(out)
        return out

    def patchify(self, imgs):
        """
        imgs: (B, 3, H, W)
        x: (B, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        
        return x

    def unpatchify(self, x, channels=3):
        """
        x: (N, L, patch_size**2 *channels)
        imgs: (N, 3, H, W)
        """
        patch_size = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, patch_size, patch_size, channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], channels, h * patch_size, h * patch_size))
        return imgs

    def forward(self, img1, img2):
        """
        img1: tensor of size B x 3 x img_size x img_size
        img2: tensor of size B x 3 x img_size x img_size
        
        out will be    B x N x (3*patch_size*patch_size)
        masks are also returned as B x N just in case 
        """
        # encoder of the masked first image 
        feat1, pos1, mask1 = self._encode_image(img1, do_mask=True)
        # encoder of the second image 
        feat2, pos2, _ = self._encode_image(img2, do_mask=False)
        # decoder 
        decfeat = self._decoder(feat1, pos1, mask1, feat2, pos2)
        # prediction head 
        out = self.prediction_head(decfeat)
        # get target
        target = self.patchify(img1)
        return out, mask1, target

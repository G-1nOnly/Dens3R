from typing import Optional

import torch
from einops import rearrange
from positional_encodings.torch_encodings import PositionalEncoding3D
from torch import Tensor, nn
from torch.nn import functional as F

from .encoding import NeRFEncoding
from .utils import trunc_normal_
from .vision_transformer import Block, Mlp


class PointTransformer(nn.Module):
    def __init__(
        self,
        voxel_resolution=128,
        in_chans=384,
        num_classes=2,
        embed_dim=384,
        depth=2,
        num_heads=6,
        mlp_ratio=4.0,
        pos_enc_type="discrete_learned",
        pos_enc_merge="sum",
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        linear_attention=False,
        **kwargs,
    ):
        super().__init__()
        self.voxel_resolution = voxel_resolution
        self.num_features = self.embed_dim = embed_dim
        self.voxel_embed = VoxelEmbedding(voxel_resolution=voxel_resolution, in_chans=in_chans, embed_dim=embed_dim)
        max_n_voxels = self.voxel_embed.max_n_voxels

        assert num_classes == 2  # otherwise, create more cls_tokens (DETR-style set-prediction)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # optional: if not provided
        trunc_normal_(self.cls_token, std=0.02)

        # positional embedding
        # -- pos.enc type
        self.pos_enc_type = pos_enc_type
        if pos_enc_type == "discrete_learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, max_n_voxels + 1, embed_dim))
            trunc_normal_(self.pos_embed, std=0.02)
        elif pos_enc_type == "discrete_sine":
            # FIXME: the ranges of sine pos.enc is incompatible w/ DINO features?
            self.register_buffer(
                "pos_embed",
                torch.cat(
                    [
                        torch.zeros(1, 1, embed_dim),  # for [cls] token
                        PositionalEncoding3D(embed_dim)(
                            torch.zeros(
                                1,
                                *(
                                    [
                                        voxel_resolution,
                                    ]
                                    * 3
                                ),
                                embed_dim,
                            )
                        ).reshape(1, -1, embed_dim),
                    ],
                    dim=1,
                ),
                persistent=False,
            )
        elif pos_enc_type == "continuous_sine":
            # use: https://github.com/nerfstudio-project/nerfstudio/blob/b8f85fb603e426309697f7590db3e2c34b9a0d66/nerfstudio/field_components/encodings.py#L1
            # with min_freq & max_freq from logscale: https://github.com/kwea123/nerf_pl/blob/b7e44f661b9c391cf10760bf19b9243c019cfbe8/models/nerf.py
            assert embed_dim % 6 == 0
            n_freqs = int(embed_dim / 6)
            min_freq_exp, max_freq_exp = 0, n_freqs - 1
            self.pos_embed = NeRFEncoding(3, n_freqs, min_freq_exp, max_freq_exp, include_input=False)
        elif pos_enc_type == "RFF":
            raise NotImplementedError
        elif pos_enc_type is None:
            self.pos_embed = None
        else:
            raise NotImplementedError(f"Unknown pos.enc: {pos_enc_type}")
        # -- pos.enc merge
        self.pos_enc_merge = pos_enc_merge
        if pos_enc_merge == "sum":
            self.pos_feat_merger = lambda x, y: x + y
        elif pos_enc_merge == "mlp":  # TODO: donot down-project pretrained DINO feature before merging
            self.pos_feat_merger = PosFeatMerger(embed_dim, embed_dim, int(embed_dim * mlp_ratio), embed_dim)
        else:
            raise ValueError(f"Unknown pos_enc_merge: {pos_enc_merge}")
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    linear_attention=linear_attention,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # (optional) segmentation head
        # self.head = nn.Linear(embed_dim, num_classes)  # num_classes - 1 for sigmoid

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x: Tensor):
        """Interpolate discrete (multi-scale) pos_enc volume w/ continuous point coordiantes.
        Args:
            x: (B, N, 3) input point cloud normalized to [-1, 1]^3
        Returns:
            pos_enc: (B, N+1, C)
        """
        assert x.min() >= -1 and x.max() <= 1

        vol_res = self.voxel_resolution
        class_pos_embed = self.pos_embed[:, 0]  # (1, C)
        patch_pos_embed = self.pos_embed[:, 1:]
        # volume_pos_embed = patch_pos_embed.reshape(1, vol_res, vol_res, vol_res, self.embed_dim)
        volume_pos_embed = rearrange(
            patch_pos_embed, "b (n0 n1 n2) c -> b c n0 n1 n2", n0=vol_res, n1=vol_res, n2=vol_res
        )
        point_pos_embed = F.grid_sample(volume_pos_embed, x[:, :, None, None, :], mode="nearest", align_corners=True)[
            ..., 0, 0
        ].permute(
            0, 2, 1
        )  # (B, N, C)

        return torch.cat((class_pos_embed.unsqueeze(0), point_pos_embed), dim=1)

    def prepare_tokens(self, xyz: Tensor, feat: Tensor, cls_token: Optional[Tensor] = None):
        """
        Args:
            xyz: (B, N, 3) point coordinates normalized to [-1, 1]^3
            feat: (B, N, C) point features
            cls_token: (B, C) instance-wise [cls] token
        Returns:
            token: (B, 1+N, C)
        """
        B, n_pts, nc = feat.shape
        feat = self.voxel_embed(xyz, feat)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        if cls_token is not None:
            cls_token = self.voxel_embed(None, cls_token[:, None])
        else:
            cls_token = self.cls_token.expand(B, -1, -1)

        feat = torch.cat((cls_token, feat), dim=1)  # (B, 1+N, C)

        # compute positional encoding
        if self.pos_enc_type in ["discrete_learned", "discrete_sine"]:
            pos_embed = self.interpolate_pos_encoding(xyz)  # (B, 1+N, C)
            feat = self.pos_feat_merger(pos_embed, feat)
            # feat = feat + pos_embed
        elif self.pos_enc_type in ["continuous_sine"]:
            pos_embed = torch.cat(
                [xyz.new_zeros((B, 1, self.embed_dim)), self.pos_embed(xyz)], dim=1  # [cls]  # pyright: ignore
            )
            feat = self.pos_feat_merger(pos_embed, feat)
            # feat = feat + pos_embed
        return self.pos_drop(feat)

    def forward(self, xyz: Tensor, feat: Tensor, cls_token: Optional[Tensor] = None) -> Tensor:
        tokens = self.prepare_tokens(xyz, feat, cls_token)  # (B, 1+N, C)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        return tokens  # tokens[:, 0]


class VoxelEmbedding(nn.Module):
    def __init__(self, voxel_resolution=128, in_chans=384, embed_dim=384):
        super().__init__()
        self.voxel_resolution = voxel_resolution
        self.voxel_embed = nn.Linear(in_chans, embed_dim)
        self.max_n_voxels = int(voxel_resolution**3)

    def forward(self, xyz: Tensor, feat: Tensor):
        # TODO: voxel-wise embedding
        feat = self.voxel_embed(feat)
        return feat


class PosFeatMerger(nn.Module):
    def __init__(self, pos_dim, feat_dim, hidden_dim, out_dim, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        self.mlp = Mlp(
            pos_dim + feat_dim, hidden_features=hidden_dim, out_features=out_dim, act_layer=act_layer, drop=drop
        )

    def forward(self, pos, feat):
        return self.mlp(torch.cat([pos, feat], dim=-1))


class TokenSegHead(nn.Module):
    def __init__(self, use_softmax=False, temperature=1.0, num_classes=2):
        super().__init__()
        self.use_softmax = use_softmax
        self.temperature = temperature
        self.num_classes = num_classes

    def forward(self, tokens: Tensor, cls_tokens: Tensor) -> Tensor:
        """
        Args:
            tokens: (B, N, C)
            cls_tokens: (B, n_cls, C)
        Returns:
            probs: (B, N, n_cls)
        """
        dot_prods = torch.einsum("bik,bjk->bij", tokens, cls_tokens)
        dot_prods *= self.temperature
        if self.use_softmax:
            assert dot_prods.shape[-1] == self.num_classes
            probs = torch.softmax(dot_prods, -1)
        else:  # element-wise sigmoid
            assert dot_prods.shape[-1] == self.num_classes - 1
            probs = torch.sigmoid(dot_prods)
        return probs

"""
Source: https://github.com/ShirAmir/dino-vit-features
"""
import argparse
import copy
import math
import shutil
import types
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import timm
import torch
from einops import rearrange, repeat
from loguru import logger
from PIL import Image
from torch import nn
from torch.nn.modules import utils as nn_utils
from torchvision import transforms
from tqdm import tqdm


class ViTExtractor:
    """This class facilitates extraction of features, descriptors, and saliency maps from a ViT.

    We use the following notation in the documentation of the module's methods:
    B - batch size
    h - number of heads. usually takes place of the channel dimension in pytorch's convention BxCxHxW
    p - patch size of the ViT. either 8 or 16.
    t - number of tokens. equals the number of patches + 1, e.g. HW / p**2 + 1. Where H and W are the height and width
    of the input image.
    d - the embedding dimension in the ViT.
    """

    def __init__(
        self,
        model_type: str = "dino_vits8",
        stride: int = 4,
        model: nn.Module = None,
        device: str = "cuda",
    ):
        """
        :param model_type: A string specifying the type of model to extract from.
                          [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 |
                          vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]
        :param stride: stride of first convolution layer. small stride -> higher resolution.
        :param model: Optional parameter. The nn.Module to extract from instead of creating a new one in ViTExtractor.
                      should be compatible with model_type.
        """
        self.model_type = model_type
        self.device = device
        if model is not None:
            self.model = model
        else:
            self.model = ViTExtractor.create_model(model_type)

        self.model = ViTExtractor.patch_vit_resolution(self.model, stride=stride)
        self.model.eval()
        self.model.to(self.device)
        self.p = self.model.patch_embed.patch_size
        self.stride = self.model.patch_embed.proj.stride

        self.mean = (0.485, 0.456, 0.406) if "dino" in self.model_type else (0.5, 0.5, 0.5)
        self.std = (0.229, 0.224, 0.225) if "dino" in self.model_type else (0.5, 0.5, 0.5)

        self._feats = []
        self.hook_handlers = []
        self.load_size = None
        self.num_patches = None  # (h, w)

    @staticmethod
    def create_model(model_type: str) -> nn.Module:
        """
        :param model_type: a string specifying which model to load. [dino_vits8 | dino_vits16 | dino_vitb8 |
                           dino_vitb16 | vit_small_patch8_224 | vit_small_patch16_224 | vit_base_patch8_224 |
                           vit_base_patch16_224]
        :return: the model
        """
        if "dino" in model_type:
            model = torch.hub.load("facebookresearch/dino:main", model_type)
        else:  # model from timm -- load weights from timm to dino model (enables working on arbitrary size images).
            temp_model = timm.create_model(model_type, pretrained=True)
            model_type_dict = {
                "vit_small_patch16_224": "dino_vits16",
                "vit_small_patch8_224": "dino_vits8",
                "vit_base_patch16_224": "dino_vitb16",
                "vit_base_patch8_224": "dino_vitb8",
            }
            model = torch.hub.load("facebookresearch/dino:main", model_type_dict[model_type])
            temp_state_dict = temp_model.state_dict()
            del temp_state_dict["head.weight"]
            del temp_state_dict["head.bias"]
            model.load_state_dict(temp_state_dict)
        return model

    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
        """
        Creates a method for position encoding interpolation.
        :param patch_size: patch size of the model.
        :param stride_hw: A tuple containing the new height and width stride respectively.
        :return: the interpolation method
        """

        def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
            dim = x.shape[-1]
            # compute number of tokens taking stride into account
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            assert (
                w0 * h0 == npatch
            ), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and 
                                            stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode="bicubic",
                align_corners=False,
                recompute_scale_factor=False,
            )
            assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        return interpolate_pos_encoding

    @staticmethod
    def patch_vit_resolution(model: nn.Module, stride: int) -> nn.Module:
        """
        change resolution of model output by changing the stride of the patch extraction.
        :param model: the model to change resolution for.
        :param stride: the new stride parameter.
        :return: the adjusted model
        """
        patch_size = model.patch_embed.patch_size
        if stride == patch_size:  # nothing to do
            return model

        stride = nn_utils._pair(stride)
        assert all(
            [(patch_size // s_) * s_ == patch_size for s_ in stride]
        ), f"stride {stride} should divide patch_size {patch_size}"

        # fix the stride
        model.patch_embed.proj.stride = stride
        # fix the positional encoding code
        model.interpolate_pos_encoding = types.MethodType(ViTExtractor._fix_pos_enc(patch_size, stride), model)
        return model

    def preprocess(
        self, image_path: Union[str, Path], load_size: Union[int, Tuple[int, int]] = None
    ) -> Tuple[torch.Tensor, Image.Image]:
        """
        Preprocesses an image before extraction.
        :param image_path: path to image to be extracted.
        :param load_size: optional. Size to resize image before the rest of preprocessing.
        :return: a tuple containing:
                    (1) the preprocessed image as a tensor to insert the model of shape BxCxHxW.
                    (2) the pil image in relevant dimensions
        """
        pil_image = Image.open(image_path).convert("RGB")
        if load_size is not None:
            pil_image = transforms.Resize(load_size, interpolation=transforms.InterpolationMode.LANCZOS)(pil_image)
        prep = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=self.mean, std=self.std)])
        prep_img = prep(pil_image)[None, ...]
        return prep_img, pil_image

    def _get_hook(self, facet: str):
        """
        generate a hook method for a specific block and facet.
        """
        facet_mapper = {"query": 0, "key": 1, "value": 2}
        hook = None
        if facet in ["attn", "token"]:

            def _hook(model, input, output):
                self._feats[facet].append(output)

            hook = _hook  # return _hook
        elif facet in ["query", "key", "value"]:
            facet_idx = facet_mapper[facet]

            def _inner_hook(module, input, output):
                input = input[0]
                B, N, C = input.shape
                qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
                self._feats[facet].append(qkv[facet_idx])  # (b, n_heads, n, c)

            hook = _inner_hook
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        return hook

    def _register_hooks(self, layers: List[int], facet: str) -> None:
        """
        register hook to extract features.
        :param layers: layers from which to extract features.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        """
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in layers:
                if facet in ["token", "cls_token"]:
                    self.hook_handlers.append(block.register_forward_hook(self._get_hook(facet)))
                elif facet == "attn":
                    self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_hook(facet)))
                elif facet in ["key", "query", "value"]:
                    self.hook_handlers.append(block.attn.register_forward_hook(self._get_hook(facet)))
                else:
                    raise TypeError(f"{facet} is not a supported facet.")

    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def _extract_features(
        self,
        batch: torch.Tensor,
        layers: Union[int, List[int]] = 11,
        facets: Union[str, List[str]] = "key",
    ) -> Dict[str, List[torch.Tensor]]:
        """
        extract features from the model
        :param batch: batch to extract features for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        :return : tensor of features.
                  if facet is 'key' | 'query' | 'value' has shape Bxhxtxd
                  if facet is 'attn' has shape Bxhxtxt
                  if facet is 'token' has shape Bxtxd
        """
        B, C, H, W = batch.shape
        self._feats = defaultdict(list)  # []
        if isinstance(facets, str):
            facets = [facets]
        for facet in facets:
            self._register_hooks(layers, facet)
        _ = self.model(batch)
        self._unregister_hooks()
        self.load_size = (H, W)
        self.num_patches = (1 + (H - self.p) // self.stride[0], 1 + (W - self.p) // self.stride[1])
        return self._feats

    def _log_bin(self, x: torch.Tensor, hierarchy: int = 2) -> torch.Tensor:
        """
        create a log-binned descriptor.
        :param x: tensor of features. Has shape Bµxhxtxd.
        :param hierarchy: how many bin hierarchies to use.
        """
        B = x.shape[0]
        num_bins = 1 + 8 * hierarchy

        bin_x = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1)  # Bx(t-1)x(dxh)
        bin_x = bin_x.permute(0, 2, 1)
        bin_x = bin_x.reshape(B, bin_x.shape[1], self.num_patches[0], self.num_patches[1])
        # Bx(dxh)xnum_patches[0]xnum_patches[1]
        sub_desc_dim = bin_x.shape[1]

        avg_pools = []
        # compute bins of all sizes for all spatial locations.
        for k in range(0, hierarchy):
            # avg pooling with kernel 3**kx3**k
            win_size = 3**k
            avg_pool = torch.nn.AvgPool2d(win_size, stride=1, padding=win_size // 2, count_include_pad=False)
            avg_pools.append(avg_pool(bin_x))

        bin_x = torch.zeros((B, sub_desc_dim * num_bins, self.num_patches[0], self.num_patches[1])).to(self.device)
        for y in range(self.num_patches[0]):
            for x in range(self.num_patches[1]):
                part_idx = 0
                # fill all bins for a spatial location (y, x)
                for k in range(0, hierarchy):
                    kernel_size = 3**k
                    for i in range(y - kernel_size, y + kernel_size + 1, kernel_size):
                        for j in range(x - kernel_size, x + kernel_size + 1, kernel_size):
                            if i == y and j == x and k != 0:
                                continue
                            if 0 <= i < self.num_patches[0] and 0 <= j < self.num_patches[1]:
                                bin_x[:, part_idx * sub_desc_dim : (part_idx + 1) * sub_desc_dim, y, x] = avg_pools[k][
                                    :, :, i, j
                                ]
                            else:  # handle padding in a more delicate way than zero padding
                                temp_i = max(0, min(i, self.num_patches[0] - 1))
                                temp_j = max(0, min(j, self.num_patches[1] - 1))
                                bin_x[:, part_idx * sub_desc_dim : (part_idx + 1) * sub_desc_dim, y, x] = avg_pools[k][
                                    :, :, temp_i, temp_j
                                ]
                            part_idx += 1
        bin_x = bin_x.flatten(start_dim=-2, end_dim=-1).permute(0, 2, 1).unsqueeze(dim=1)
        # Bx1x(t-1)x(dxh)
        return bin_x

    def extract_descriptors(
        self,
        batch: torch.Tensor,
        layer: int = 11,
        facet: Union[str, List[str]] = "key",
        bin: bool = False,
        include_cls: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        extract descriptors from the model
        :param batch: batch to extract descriptors for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options:
            ['key' | 'query' | 'value' | 'token' | 'cls_token']
        :param bin: apply log binning to the descriptor. default is False.
        :return: tensor of descriptors. Bx1xtxd' where d' is the dimension of the descriptors.
        """
        if isinstance(facet, str):
            facet = [facet]
        for _facet in facet:
            assert _facet in [
                "key",
                "query",
                "value",
                "token",
                "attn",
            ], f"""{_facet} is not a supported facet for descriptors.
                choose from ['key' | 'query' | 'value' | 'token' | 'attn'] """

        self._extract_features(batch, [layer], facet)
        descs = {}
        for _facet in facet:
            x = self._feats[_facet][0]

            if _facet == "attn":  # (B, h, t, t)
                # save multi-head cls attention maps
                cls_mh_attn_map = x[:, :, 0, 1:]  # (B, h, t-1)
                descs["cls_mh_attn_map"] = repeat(cls_mh_attn_map, "b h t -> b c t h", c=1)  # (B, 1, t-1, h)
                # save saliency map
                head_idxs = [0, 2, 4, 5]  # https://github.com/ShirAmir/dino-vit-features/issues/1
                cls_mh_attn_map = x[:, head_idxs, 0, 1:]
                cls_attn_map = cls_mh_attn_map.mean(dim=1)  # (B, t-1)
                _min, _max = cls_attn_map.min(dim=1)[0], cls_attn_map.max(dim=1)[0]
                cls_attn_map = (cls_attn_map - _min) / (_max - _min)
                descs["saliency_map"] = cls_attn_map[:, None, :, None]  # (B, 1, t-1, 1)
            else:
                if _facet == "token":
                    x.unsqueeze_(dim=1)  # Bx1xtxd
                    descs["cls_token"] = x[:, :, [0], :]  # (B, 1, 1, d) the cls token

                x_cls, x = x[:, :, [0], :], x[:, :, 1:, :]  # (b, h, 1, d), (b, h, n, d)
                if include_cls:
                    # assert not bin, "bin = True and include_cls = True are not supported together, set one of them False."
                    descs[f"cls_{_facet}"] = rearrange(x_cls, "b h n d -> b 1 n (d h)")  # (B, 1, n, (d, h))
                if not bin:  # b, h, n, c => b, n, c, h => b, 1, n, (c, h)
                    desc = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1).unsqueeze(dim=1)  # (b, 1, n, (d, h))
                else:
                    desc = self._log_bin(x)
                descs[_facet] = desc
        return descs

    def extract_saliency_maps(self, batch: torch.Tensor) -> torch.Tensor:
        """
        extract saliency maps. The saliency maps are extracted by averaging several attention heads from the last layer
        in of the CLS token. All values are then normalized to range between 0 and 1.
        :param batch: batch to extract saliency maps for. Has shape BxCxHxW.
        :return: a tensor of saliency maps. has shape Bxt-1
        """
        assert self.model_type == "dino_vits8", "saliency maps are supported only for dino_vits model_type."
        self._extract_features(batch, [11], "attn")
        head_idxs = [0, 2, 4, 5]
        curr_feats = self._feats[0]  # (B, h, t, t)
        cls_mh_attn_map = curr_feats[:, head_idxs, 0, 1:]  # (B, h, t-1)
        cls_attn_map = cls_mh_attn_map.mean(dim=1)  # (B, t-1), excluding cls-cls attention
        # normalize cls_attn_map
        temp_mins, temp_maxs = cls_attn_map.min(dim=1)[0], cls_attn_map.max(dim=1)[0]
        cls_attn_maps = (cls_attn_map - temp_mins) / (temp_maxs - temp_mins)  # (B, t-1), normalize to range [0,1]
        # normalize_cls_mh_attn_map
        temp_mins, temp_maxs = cls_mh_attn_map.min(dim=-1)[0], cls_mh_attn_map.max(dim=-1)[0]
        cls_mh_attn_map = (cls_mh_attn_map - temp_mins) / (temp_maxs - temp_mins)  # (B, h, t-1)
        return cls_attn_maps, cls_mh_attn_map

    def reshape_descriptor(self, descriptor: torch.Tensor, bin: bool = False) -> torch.Tensor:
        if bin:
            raise NotImplementedError()
        # descriptor: (b, 1, hxw, c)
        # return: (h, w, c)
        assert descriptor.shape[0] == 1
        h, w = self.num_patches
        descriptor = descriptor[0, 0].reshape(h, w, -1)
        return descriptor


@torch.no_grad()
def extract_single(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = ViTExtractor(args.model_type, args.stride, device=device)
    image_batch, image_pil = extractor.preprocess(args.image_path, args.load_size)
    # print(f"Image {args.image_path} is preprocessed to tensor of size {image_batch.shape}.")

    descriptors = {}
    raw_descriptors = extractor.extract_descriptors(image_batch.to(device), args.layer, args.facet, args.bin)
    for desc_name, desc in raw_descriptors.items():
        # print(f"{desc_name}: {desc.shape}")
        if "cls_" in desc_name:
            desc = desc[0, 0, 0, :]  # (c, ), ((d, n_heads), )
        else:
            if args.unflatten:
                desc = extractor.reshape_descriptor(desc, bin=args.bin)  # (h, w, c)
        descriptors[desc_name] = desc.cpu().numpy()

    if args.save_npz:
        output_path = args.output_path.with_suffix(".npz")
        npz_data = {}
        _kqv = [int(k in descriptors) for k in ["key", "query", "value"]]
        _sum_kqv = sum(_kqv)
        if _sum_kqv == 0:
            feature_key = "token"
        elif _sum_kqv == 1:
            feature_key = ["key", "query", "value"][np.argmax(_kqv)]
        else:
            raise RuntimeError("Unsupported configuration, only one key, query, value feature could be specified.")
        npz_data["features"] = descriptors.pop(feature_key)  # main feature
        npz_data.update(descriptors)
        # cls_mh_attn_map: (h, w, n_heads) | saliency_map: (h, w, 1)
        # token: (h, w, c) | cls_token: (c, ) | cls_{facet}: (c, )
        if args.fp16:
            npz_data = {k: v.astype(np.float16) for k, v in npz_data.items()}

        try:
            np.savez_compressed(output_path, **npz_data)
        except PermissionError:
            shutil.chown(output_path.parent, user="wangyuang", group="wangyuang")
            for p in output_path.parent.iterdir():
                shutil.chown(p, user="wangyuang", group="wangyuang")
            np.savez_compressed(output_path, **npz_data)
    else:
        raise NotImplementedError()
        # output_path = args.output_path.with_suffix('.pt')
        # torch.save(descriptors.cpu(), output_path)
    print(f"Descriptors saved to: {output_path}")


""" taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="Facilitate ViT Descriptor extraction.")
    parser.add_argument("--image_path", type=Path, required=True, help="path of the extracted image.")
    parser.add_argument(
        "--output_path",
        type=Path,
        required=False,
        help="path to file containing extracted descriptors.",
    )
    parser.add_argument("--load_size", default=224, type=int, help="load size of the input image.")
    parser.add_argument(
        "--stride",
        default=4,
        type=int,
        help="""stride of first convolution layer. small stride -> higher resolution.""",
    )
    parser.add_argument(
        "--model_type",
        default="dino_vits8",
        type=str,
        help="""type of model to extract. 
                Choose from [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 | 
                vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]""",
    )
    parser.add_argument(
        "--facet",
        default="key",
        type=str,
        nargs="+",
        choices=["key", "query", "value", "token", "attn"],
        help="""facet to create descriptors from.
                                options: ['key' | 'query' | 'value' | 'token']""",
    )
    parser.add_argument("--layer", default=11, type=int, help="layer to create descriptors from.")
    parser.add_argument("--bin", default="False", type=str2bool, help="create a binned descriptor if True.")
    parser.add_argument("--unflatten", action="store_true")
    parser.add_argument("--save_npz", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--data_root", type=Path)
    parser.add_argument("--image_dir_name", type=str, default="image")
    parser.add_argument("--output_dir_name", type=str, default="dino_feats")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.image_path.is_dir():  # extract a single sequence
        # TODO: support parallel extraction
        if not args.output_path.exists():
            args.output_path.mkdir()
        assert args.output_path.is_dir()
        image_paths, output_paths = [], []
        for img_path in args.image_path.iterdir():
            image_paths.append(img_path)
            output_paths.append(args.output_path / img_path.with_suffix(".pt").name)

        for img_p, output_p in tqdm(zip(image_paths, output_paths), desc="extracting features"):
            _args = copy.deepcopy(args)
            _args.image_path, _args.output_path = img_p, output_p
            extract_single(_args)
    elif args.image_path.is_file() and args.image_path.suffix == ".txt":  # extract a sequence of images
        if not args.output_path.exists():
            args.output_path.mkdir()
        with open(args.image_path, "r") as f:
            image_paths = [Path(p) for p in f.read().strip().split()]
        output_paths = [args.output_path / p.with_suffix(".pt").name for p in image_paths]
        logger.debug(f"{len(image_paths)} images parsed from {args.image_path}")

        for img_p, output_p in tqdm(zip(image_paths, output_paths), desc="extracting features"):
            _args = copy.deepcopy(args)
            _args.image_path, _args.output_path = img_p, output_p
            extract_single(_args)
        args.image_path.unlink()
    elif args.image_path.is_file():  # extract a single image
        extract_single(args)
    else:
        raise ValueError()


if __name__ == "__main__":
    main()

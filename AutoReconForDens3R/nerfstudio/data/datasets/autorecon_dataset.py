"""
AutoRecon dataset.
"""

from typing import Dict

import torch
from einops import repeat

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs, Semantics
from nerfstudio.data.datasets.base_dataset import GeneralizedDataset
from nerfstudio.data.utils.data_utils import get_image_mask_tensor_from_path


class AutoReconDataset(GeneralizedDataset):
    """

    Args:
        dataparser_outputs: description of where and how to read input images.
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        assert "semantics" in dataparser_outputs.metadata.keys()
        self.semantics = self.metadata["semantics"]

    def get_metadata(self, data: Dict) -> Dict:
        """binary mask as semantics"""
        img_h, img_w = data["image"].shape[:2]
        filepath = self.semantics.filenames[data["image_idx"]]
        binary_mask = get_image_mask_tensor_from_path(
            filepath=filepath, scale_factor=self.scale_factor).to(torch.float32)
        if binary_mask.shape[:2] != (img_h, img_w):
            binary_mask = torch.nn.functional.interpolate(
                repeat(binary_mask, "h w c -> b c h w", b=1),
                size=(img_h, img_w), mode="nearest")[0].permute(1, 2, 0)
        
        # mask = torch.zeros_like(binary_mask)
        semantics = binary_mask.to(torch.long)
        
        # if "mask" in data.keys():
        #     binary_mask = binary_mask & data["mask"]
        return {"semantics": semantics}

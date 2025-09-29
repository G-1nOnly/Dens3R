
from dataclasses import dataclass, field
from typing import Type

from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.data.datasets.autorecon_dataset import AutoReconDataset


@dataclass
class AutoReconDataManagerConfig(VanillaDataManagerConfig):
    """A semantic datamanager - required to use with .setup()"""

    _target: Type = field(default_factory=lambda: AutoReconDataManager)


class AutoReconDataManager(VanillaDataManager):  # pylint: disable=abstract-method
    """

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    def create_train_dataset(self) -> AutoReconDataset:
        return AutoReconDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split="train"),
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> AutoReconDataset:
        _eval_scale_factor = self.config.eval_camera_res_scale_factor
        scale_factor = (
            _eval_scale_factor
            if _eval_scale_factor is not None else self.config.camera_res_scale_factor
        )
        return AutoReconDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=scale_factor,
        )

import tqdm as _tqdm
from ray.experimental import tqdm_ray

from auto_decomp.utils import ray as ray_utils


def tqdm(*args, **kwargs):
    pbar = None
    if ray_utils.is_ray_environment():
        disable = kwargs.pop("disable", False)
        if disable:
            pbar = _tqdm.tqdm(*args, **kwargs, disable=True)
        else:
            pbar = tqdm_ray.tqdm(*args, **kwargs)
    else:
        pbar = _tqdm.tqdm(*args, **kwargs)
    return pbar

"""Scheduler for arbitrary weights"""
import math
import numpy as np
from functools import lru_cache


def cosine_annealing(cur_step, max_step=None, min_val=0.0, max_val=1.0):
    cur_val = min_val + (max_val - min_val) \
        * (1 + math.cos(cur_step * math.pi / max_step)) / 2
    return cur_val


@lru_cache(maxsize=128)
def _compute_exp_anneal_gamma(max_step, min_val, max_val):
    return np.power(min_val / max_val, 1 / max_step)


def exp_annealing(cur_step, max_step=None, min_val=0.0, max_val=1.0):
    gamma = _compute_exp_anneal_gamma(max_step, min_val, max_val)
    return max_val * np.power(gamma, cur_step)


def constant(cur_step, max_step=None, min_val=0.0, max_val=1.0):
    return max_val


# TODO: sine scheduler

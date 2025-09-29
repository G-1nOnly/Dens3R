
from rich.console import Console

import torch
import tinycudann as tcnn
from torch import nn

CONSOLE = Console(width=120)


class ProgressiveEncoding(nn.Module):
    """Progressive multi-resolution encoding proposed in NeuS2."""
    
    def __init__(
        self,
        encoding: tcnn.Encoding,
        n_levels: int = 16,  # number of resolution levels
        n_levels_init: int = 1,  # number of resolution levels to use in the beginning
        n_features_per_level: int = 2,
        n_scheduler_steps: int = 10000  # total number of steps of the scheduler
    ) -> None:
        super().__init__()
        self.encoding = encoding
        self.n_levels = n_levels
        if n_scheduler_steps == -1:  # enable progressive encoding but set all weights to 1
            self.n_levels_init = self.n_levels  # useful for testing / using pretrained models
        else:
            self.n_levels_init = n_levels_init
        self.n_features_per_level = n_features_per_level
        self.n_output_dims = self.encoding.n_output_dims
        assert self.encoding.n_output_dims == n_levels * n_features_per_level
        
        self.n_scheduler_steps = n_scheduler_steps
        self.step_size = n_scheduler_steps // n_levels
        self.cur_iter = 0  # TODO: make ProgressiveEncoding stateless and pass in cur_iter as an argument
        self.cur_level: int = -1
        # self.register_buffer("weights", self._get_init_weights(), persistent=False)  # persistent=True / nn.Parameter
        self.weights = nn.Parameter(self._get_init_weights(), requires_grad=False)
        CONSOLE.print(f'[bold green]Using progressive hash encoding training (n_iters={n_scheduler_steps})')
    
    def _get_init_weights(self):
        weights = torch.zeros(self.n_levels, self.n_features_per_level, dtype=torch.float32)
        weights[:self.n_levels_init, :] = 1.0
        return weights
    
    def forward(self, positions):
        if self.n_scheduler_steps == -1:
            return self.encoding(positions)
        
        cur_level = min(int(self.cur_iter // self.step_size) + self.n_levels_init - 1, self.n_levels - 1)
        if self.cur_level != cur_level:
            self.cur_level = cur_level
            self.weights[cur_level, :] = 1.0
        encoding = self.encoding(positions)  # (..., c)
        encoding = encoding * self.weights.view(-1)
        
        if self.training:
            self.cur_iter += 1
        return encoding

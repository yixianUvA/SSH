import time
import math
import warnings
from typing import Any, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse, DTCWTForward, DTCWTInverse
import pywt
# from haar_pytorch import HaarForward, HaarInverse


from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.other import transpose
from args import *  # Import all necessary arguments from the args module

args_l = get_args()



class FourierLayer(BaseTunerLayer):
    adapter_layer_names = ["spectrum"]

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.n_frequency = {}
        self.scale = {}
        self.spectrum = nn.ParameterDict({})
        self.indices = {}
        self.hierarchical_levels = {}  # Track how many indices at each level
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            in_features, out_features = base_layer.input_size, base_layer.output_size
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(self, adapter_name, n_frequency, scale, init_fourier_weights=None):
        if n_frequency <= 0:
            raise ValueError(f"`n_frequency` should be a positive integer value but the value passed is {n_frequency}")
        self.n_frequency[adapter_name] = n_frequency
        self.scale[adapter_name] = scale
        self.indices[adapter_name], self.hierarchical_levels[adapter_name] = self._select_hierarchical_indices(n_frequency)
        self.spectrum[adapter_name] = nn.Parameter(torch.randn(n_frequency), requires_grad=True)
  
        weight = getattr(self.get_base_layer(), "weight", None)
        if weight is not None:
            self.to(weight.device, dtype=weight.dtype)
        self.set_adapter(self.active_adapters)


    def _select_hierarchical_indices(self, n_frequency):
        """
        Select the spectrum indices in a hierarchical manner across Haar wavelet sub-bands (cA, cH, cV, cD).
        Allocate most indices to the approximation coefficients (cA) and the rest equally to cH, cV, and cD.
        """
        level_size = self.in_features // 2  # Assume we are splitting the matrix into four quadrants at level 1
        n_cA = int(n_frequency * 0.4)  # 50% of the indices allocated to cA (approximation)
        n_cHVD = n_frequency - n_cA  # The remaining indices are split among cH, cV, and cD (details)

        # Divide n_cHVD equally among the three detail components: cH, cV, and cD
        n_cH = n_cV = n_cD = n_cHVD // 3
        n_cHVD_remain = n_cHVD - (n_cH * 3)  # Remainder to make sure total indices == n_frequency

        indices = []
        hierarchical_levels = []

        # Randomly select n_cA indices for the approximation coefficients (cA)
        cA_indices = torch.randperm(level_size * level_size)[:n_cA]
        for idx in cA_indices:
            row = idx // level_size
            col = idx % level_size
            indices.append((row, col))
            hierarchical_levels.append('cA')

        # Randomly select indices for the detail components (cH, cV, cD)
        def select_indices_for_band(n_band, row_offset, col_offset, band_name):
            band_indices = torch.randperm(level_size * level_size)[:n_band]
            for idx in band_indices:
                row = row_offset + (idx // level_size)
                col = col_offset + (idx % level_size)
                indices.append((row, col))
                hierarchical_levels.append(band_name)

        # cH: Horizontal details (top-right quadrant)
        select_indices_for_band(n_cH + (n_cHVD_remain > 0), 0, level_size, 'cH')
        # cV: Vertical details (bottom-left quadrant)
        select_indices_for_band(n_cV + (n_cHVD_remain > 1), level_size, 0, 'cV')
        # cD: Diagonal details (bottom-right quadrant)
        select_indices_for_band(n_cD + (n_cHVD_remain > 2), level_size, level_size, 'cD')

        indices = torch.tensor(indices).t()  # Convert to tensor and transpose for easy indexing
        return indices, hierarchical_levels



    def get_delta_weight(self, adapter) -> torch.Tensor:
        spectrum = self.spectrum[adapter]
        indices = self.indices[adapter].to(spectrum.device)

        # Construct a sparse matrix directly using the indices
        sparse_tensor = torch.sparse_coo_tensor(indices, spectrum, (self.in_features, self.in_features), device=spectrum.device)

        # Perform the inverse Haar wavelet transform to reconstruct the full weight matrix
        # Move the tensor to CPU and convert to NumPy for PyWavelets operations
        coeffs2 = self._decompose_to_wavelet_bands(sparse_tensor.to_dense().detach().cpu())
        reconstructed_np = pywt.idwt2(coeffs2, wavelet='haar', mode='symmetric')

        # Convert back to PyTorch tensor and apply scaling
        #delta_w = torch.tensor(reconstructed_np, device=sparse_tensor.device) * self.scale[adapter]
        delta_w = torch.tensor(reconstructed_np, device=sparse_tensor.device) 
        return delta_w



    def _decompose_to_wavelet_bands(self, dense_s):
        rows, cols = dense_s.shape[-2], dense_s.shape[-1]
        half_rows, half_cols = rows // 2, cols // 2

        cA = dense_s[:half_rows, :half_cols]   
        cH = dense_s[:half_rows, half_cols:]   
        cV = dense_s[half_rows:, :half_cols]   
        cD = dense_s[half_rows:, half_cols:]   

        return cA, (cH, cV, cD)


class Linear(nn.Module, FourierLayer):
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        n_frequency: int = 0,
        scale: float = 0.1,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        init_fourier_weights: Union[bool, str] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        FourierLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, n_frequency, scale, init_fourier_weights)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype
        if self.disable_adapters:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.spectrum.keys():
                    continue
                delta_w = self.get_delta_weight(active_adapter) / 5
                x, delta_w = x.to(delta_w.dtype), delta_w.to(x.device)
                result += torch.einsum('ijk,kl->ijl', x, delta_w)

        return result.to(previous_dtype)

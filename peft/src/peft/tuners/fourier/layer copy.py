import time
import math
import warnings
from typing import Any, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse, DTCWTForward, DTCWTInverse
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
        n_cA = int(n_frequency * 0.6)  # 50% of the indices allocated to cA (approximation)
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

        # Convert the sparse tensor to a dense tensor
        dense_s = sparse_tensor.to_dense()

        # Perform the inverse Haar wavelet transform to reconstruct the full weight matrix
        coeffs = self._decompose_to_wavelet_bands(dense_s)

        # Use DWTInverse to reconstruct the tensor
        #idwt = DWTInverse(wave='haar').to(spectrum.device)
        
        idwt = DWTInverse(wave='sym4').to(spectrum.device)
        # idwt = DWTInverse(wave='coif3').to(spectrum.device)
        
        
        # Prepare coefficients in expected format: (yl, yh)
        cA, cH, cV, cD = coeffs

        # For DWTInverse, the inputs should be in NCHW format
        cA = cA.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        cH = cH.unsqueeze(0).unsqueeze(0)
        cV = cV.unsqueeze(0).unsqueeze(0)
        cD = cD.unsqueeze(0).unsqueeze(0)

        # Stack cH, cV, cD into yh
        yh = [(torch.stack([cH.squeeze(), cV.squeeze(), cD.squeeze()], dim=0).unsqueeze(0).unsqueeze(0))]

        # Reconstruct
        reconstructed_tensor = idwt((cA, yh))
        reconstructed_tensor = reconstructed_tensor.squeeze(0).squeeze(0)
        delta_w = reconstructed_tensor

        return delta_w



    def _decompose_to_wavelet_bands(self, dense_s):
        rows, cols = dense_s.shape[-2], dense_s.shape[-1]
        half_rows, half_cols = rows // 2, cols // 2

        cA = dense_s[:half_rows, :half_cols]   
        cH = dense_s[:half_rows, half_cols:]   
        cV = dense_s[half_rows:, :half_cols]   
        cD = dense_s[half_rows:, half_cols:]   

        return cA, cH, cV, cD


    def print_wavelet_coeffs_summary(self, coeffs2):
        """Print a compact summary for wavelet coefficients."""
        cA, (cH, cV, cD) = coeffs2
        print("Wavelet Coefficients Summary:")
        print(f"  cA -> Mean: {cA.mean():.2e}, Var: {cA.var():.2e}, Min: {cA.min():.2e}, Max: {cA.max():.2e}")
        print(f"  cH -> Mean: {cH.mean():.2e}, Var: {cH.var():.2e}, Min: {cH.min():.2e}, Max: {cH.max():.2e}")
        print(f"  cV -> Mean: {cV.mean():.2e}, Var: {cV.var():.2e}, Min: {cV.min():.2e}, Max: {cV.max():.2e}")
        print(f"  cD -> Mean: {cD.mean():.2e}, Var: {cD.var():.2e}, Min: {cD.min():.2e}, Max: {cD.max():.2e}")

    def print_reconstruction_summary(self, reconstructed_np):
        """Print a compact summary for the reconstructed matrix."""
        print("Reconstructed Matrix Summary:")
        print(f"  Mean: {np.mean(reconstructed_np):.2e}, Var: {np.var(reconstructed_np):.2e}, Min: {np.min(reconstructed_np):.2e}, Max: {np.max(reconstructed_np):.2e}")


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
        # Initialize variables to accumulate statistics
        delta_w_means_before, delta_w_vars_before = [], []
        delta_w_means_after, delta_w_vars_after = [], []

        if self.disable_adapters:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.spectrum.keys():
                    continue
                
                delta_w = self.get_delta_weight(active_adapter)

                delta_w_mean_before = torch.mean(delta_w).item()
                delta_w_var_before = torch.var(delta_w).item()


                # Apply scaling factor
                # delta_w = delta_w * self.scale[active_adapter]
                delta_w = delta_w / 20
                # After scaling
                delta_w_mean_after = torch.mean(delta_w).item()
                delta_w_var_after = torch.var(delta_w).item()


                # Collect the means and variances
                delta_w_means_before.append(delta_w_mean_before)
                delta_w_vars_before.append(delta_w_var_before)
                delta_w_means_after.append(delta_w_mean_after)
                delta_w_vars_after.append(delta_w_var_after)

                # if delta_w.requires_grad:
                #     print(f"Gradient is flowing through delta_w: {delta_w.requires_grad}")
                # else:
                #     print("Warning: No gradient is flowing through delta_w")

                #print(f"x shape: {x.shape}, delta_w shape: {delta_w.shape}")
                if delta_w.shape != (self.in_features, self.in_features):
                    delta_w = delta_w.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions if needed
                    delta_w = F.interpolate(delta_w, size=(self.in_features, self.in_features), mode='bilinear', align_corners=False)
                    delta_w = delta_w.squeeze()


                x, delta_w = x.to(delta_w.dtype), delta_w.to(x.device)

                result += torch.einsum('...i,ij->...j', x, delta_w)

                # Store the overall statistics
            self.delta_w_stats = {
                "mean_before": np.mean(delta_w_means_before),
                "var_before": np.mean(delta_w_vars_before),
                "mean_after": np.mean(delta_w_means_after),
                "var_after": np.mean(delta_w_vars_after)
            }

        return result.to(previous_dtype)

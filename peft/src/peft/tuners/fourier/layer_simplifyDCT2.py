from typing import Any, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as dct

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.other import transpose
from args import *  # Import all necessary arguments from the args module

args_l = get_args()

class FourierLayer(BaseTunerLayer):
    adapter_layer_names = ["spectrum"]

    def __init__(self, base_layer: nn.Module, **kwargs):
        super().__init__()
        self.base_layer = base_layer
        self.n_frequency = {}
        self.scale = {}
        self.spectrum = nn.ParameterDict({})
        self.indices = {}
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        if isinstance(base_layer, nn.Linear):
            self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

    # def initialize_spectral_indices(self, n_frequency):
    #     # Apply DCT to the base layer's weights
    #     weight_dct = dct.dct_2d(self.base_layer.weight.data)
        
    #     # Calculate the magnitude of each DCT coefficient
    #     magnitude_values = torch.abs(weight_dct)
        
    #     # Flatten the magnitude values and select indices based on the highest magnitude values
    #     magnitude_values_flat = magnitude_values.reshape(-1)
    #     _, indices = torch.topk(magnitude_values_flat, n_frequency)

    #     # Convert flat indices to 2D indices
    #     indices_2d = torch.stack([indices // self.in_features, indices % self.in_features], dim=0)
        
    #     return indices_2d
    
    # def initialize_spectral_indices(self, n_frequency):
    #     # Apply DCT to the base layer's weights
    #     weight_dct = dct.dct_2d(self.base_layer.weight.data)
        
    #     # Calculate the magnitude of each DCT coefficient
    #     magnitude_values = torch.abs(weight_dct)
        
    #     # Flatten the magnitude values
    #     magnitude_values_flat = magnitude_values.reshape(-1)
        
    #     # Select a portion based on magnitude
    #     topk_portion = int(0.5 * n_frequency)  # Select half based on magnitude
    #     _, top_indices = torch.topk(magnitude_values_flat, topk_portion)
        
    #     # Select the remaining portion randomly
    #     random_portion = n_frequency - topk_portion
    #     random_indices = torch.randperm(magnitude_values_flat.size(0), generator=torch.Generator().manual_seed(args_l.entry_seed))[:random_portion]
        
    #     # Combine both sets of indices
    #     combined_indices = torch.cat((top_indices, random_indices))
        
    #     # Convert flat indices to 2D indices
    #     indices_2d = torch.stack([combined_indices // self.in_features, combined_indices % self.in_features], dim=0)
    #     return indices_2d
    
    def initialize_spectral_indices(self, n_frequency, energy_threshold=0.95):
        # Apply DCT to the base layer's weights
        weight_dct = dct.dct_2d(self.base_layer.weight.data)
        
        # Flatten and sort the DCT coefficients by magnitude
        flattened_dct = torch.abs(weight_dct).flatten()
        sorted_indices = torch.argsort(flattened_dct, descending=True)
        
        # Calculate cumulative energy and select the top coefficients covering the energy_threshold
        cumulative_energy = torch.cumsum(flattened_dct[sorted_indices], dim=0)
        total_energy = cumulative_energy[-1]
        selected_energy_indices = sorted_indices[cumulative_energy / total_energy <= energy_threshold]

        # Ensure at least n_frequency are selected
        if len(selected_energy_indices) < n_frequency:
            selected_energy_indices = sorted_indices[:n_frequency]

        # Add diversity by selecting some high-frequency components
        high_freq_indices = sorted_indices[-(n_frequency//5):]
        selected_indices = torch.cat([selected_energy_indices, high_freq_indices])

        # Ensure unique indices and truncate to n_frequency
        selected_indices = torch.unique(selected_indices)[:n_frequency]

        # Convert flat indices to 2D indices
        indices_2d = torch.stack([selected_indices // self.in_features, selected_indices % self.in_features], dim=0)
        return indices_2d


    def update_layer(self, adapter_name, n_frequency, scale, init_fourier_weights=None):
        if n_frequency <= 0:
            raise ValueError(f"`n_frequency` should be a positive integer value but the value passed is {n_frequency}")
        self.n_frequency[adapter_name] = n_frequency
        self.scale[adapter_name] = scale
        
        # Initialize spectral indices based on magnitude
        self.indices[adapter_name] = self.initialize_spectral_indices(n_frequency)
        # self.indices[adapter_name] = torch.randperm(self.in_features * self.in_features, generator=torch.Generator().manual_seed(args_l.entry_seed))[:n_frequency]
        # self.indices[adapter_name] = torch.stack([self.indices[adapter_name] // self.in_features, self.indices[adapter_name] % self.in_features], dim=0)
        
        self.spectrum[adapter_name] = nn.Parameter(torch.randn(n_frequency), requires_grad=True)

        self.to(self.base_layer.weight.device)  # Ensure layer is on the correct device

    def idct2(self, x):
        return dct.idct_2d(x)
    
    # def regularization_loss(self, lambda_reg=0.01):
    #     """
    #     Compute the regularization loss for the DCT coefficients (L2 regularization).
        
    #     Parameters:
    #     - lambda_reg: Regularization strength (default 0.01)
        
    #     Returns:
    #     - reg_loss: The computed regularization loss
    #     """
    #     reg_loss = 0.0
    #     for adapter_name, spectrum in self.spectrum.items():
    #         reg_loss += lambda_reg * torch.sum(spectrum ** 2)
    #     return reg_loss

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
    ):
        super().__init__()
        FourierLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, n_frequency, scale, init_fourier_weights)
        
   

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype
        result = self.base_layer(x, *args, **kwargs)
        for active_adapter in self.active_adapters:
            if active_adapter not in self.spectrum.keys():
                continue
            
            spectrum = self.spectrum[active_adapter]
            indices = self.indices[active_adapter].to(spectrum.device)
            scale = self.scale[active_adapter]

            dense_s = torch.zeros((self.in_features, self.in_features), dtype=spectrum.dtype, device=spectrum.device)
            dense_s[indices[0, :], indices[1, :]] = spectrum

            if spectrum.dtype == torch.bfloat16:
                dense_s = dense_s.to(torch.float16)
            
            delta_w = self.idct2(dense_s) * scale
            x, delta_w = x.to(spectrum.dtype), delta_w.to(spectrum.dtype)
            result += torch.einsum('ijk,kl->ijl', x, delta_w)

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        return f"fourier.{super().__repr__()}"

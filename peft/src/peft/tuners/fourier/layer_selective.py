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

    def partition_frequency_domain(self, shape, partitions=3):
        """
        Partition the frequency domain into `partitions` distinct regions.
        
        Args:
            shape: Shape of the DCT-transformed weight matrix (e.g., (in_features, out_features)).
            partitions: Number of partitions to create in the frequency domain.
        
        Returns:
            List of masks, one for each partition.
        """
        in_features, out_features = shape
        rows, cols = np.ogrid[:in_features, :out_features]
        distance_from_origin = np.sqrt((rows - in_features // 2)**2 + (cols - out_features // 2)**2)

        max_distance = np.max(distance_from_origin)
        partition_size = max_distance / partitions

        masks = []
        for i in range(partitions):
            mask = (distance_from_origin >= i * partition_size) & (distance_from_origin < (i + 1) * partition_size)
            masks.append(torch.tensor(mask, dtype=torch.bool))
        
        return masks

    def hybrid_selection_within_partition(self, weight_dct, mask, n_select, energy_ratio=0.7):
        """
        Perform hybrid selection within a given frequency partition.
        
        Args:
            weight_dct: The DCT-transformed weights.
            mask: Boolean mask for the partition.
            n_select: Number of frequencies to select from this partition.
            energy_ratio: Ratio of frequencies selected based on energy.
        
        Returns:
            Selected indices within the partition.
        """
        partitioned_weights = weight_dct[mask]
        magnitude_values = torch.abs(partitioned_weights)

        n_energy_select = int(n_select * energy_ratio)
        n_random_select = n_select - n_energy_select

        # Select top magnitudes
        top_magnitudes, top_indices = torch.topk(magnitude_values, n_energy_select)

        # Random selection for diversity
        random_indices = torch.randperm(magnitude_values.numel(), generator=torch.Generator().manual_seed(args_l.entry_seed))[:n_random_select]

        selected_indices = torch.cat([top_indices, random_indices])
        return selected_indices

    def stratified_frequency_selection(self, weight_dct, n_frequency, partitions=3, energy_ratio=0.7):
        """
        Perform stratified sampling to select frequencies across different partitions.
        
        Args:
            weight_dct: The DCT-transformed weights.
            n_frequency: Total number of frequencies to select.
            partitions: Number of partitions in the frequency domain.
            energy_ratio: Ratio of frequencies selected based on energy.
        
        Returns:
            Indices of selected frequencies.
        """
        masks = self.partition_frequency_domain(weight_dct.shape, partitions=partitions)
        n_per_partition = n_frequency // partitions

        selected_indices = []
        for mask in masks:
            partition_indices = self.hybrid_selection_within_partition(weight_dct, mask, n_per_partition, energy_ratio)
            selected_indices.append(partition_indices)

        # Combine and ensure unique indices
        selected_indices = torch.cat(selected_indices).unique()

        # If fewer indices are selected, perform random selection to fill the gap
        if len(selected_indices) < n_frequency:
            remaining_indices = torch.randperm(weight_dct.numel(), generator=torch.Generator().manual_seed(args_l.entry_seed))
            selected_indices = torch.cat([selected_indices, remaining_indices])[:n_frequency]

        return selected_indices

    def initialize_spectral_indices(self, n_frequency, energy_threshold=0.95, partitions=3, energy_ratio=0.7):
        # Apply DCT to the base layer's weights
        weight_dct = dct.dct_2d(self.base_layer.weight.data)
        
        # Perform stratified frequency selection
        selected_indices = self.stratified_frequency_selection(weight_dct, n_frequency, partitions, energy_ratio)

        # Convert flat indices to 2D indices
        indices_2d = torch.stack([selected_indices // self.in_features, selected_indices % self.in_features], dim=0)
        return indices_2d

    def update_layer(self, adapter_name, n_frequency, scale, init_fourier_weights=None):
        if n_frequency <= 0:
            raise ValueError(f"`n_frequency` should be a positive integer value but the value passed is {n_frequency}")
        self.n_frequency[adapter_name] = n_frequency
        self.scale[adapter_name] = scale
        
        self.indices[adapter_name] = self.initialize_spectral_indices(n_frequency)
        
        self.spectrum[adapter_name] = nn.Parameter(torch.randn(n_frequency), requires_grad=True)

        self.to(self.base_layer.weight.device)  # Ensure layer is on the correct device

    def idct2(self, x):
        return dct.idct_2d(x)

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

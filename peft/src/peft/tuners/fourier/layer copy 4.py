from typing import Any, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

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
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs
        #self.scale_param = nn.Parameter(torch.tensor(71.0))

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


    def compute_energy(self, transformed_weights):
        """
        Compute the energy of the transformed weights (DHT coefficients).
        The energy is typically the magnitude squared of the DHT coefficients.
        """
        return torch.abs(transformed_weights)**2

    def energy_based_selection1(self, weight_dht, n_select, energy_threshold):
        # Compute energy of the DHT coefficients
        energy = self.compute_energy(weight_dht)
        
        # Flatten the energy tensor and sort it based on magnitude
        flat_energy = energy.flatten()
        sorted_energy, sorted_indices = torch.sort(flat_energy, descending=True)
        
        # Select exactly `n_select` indices based on the highest energy contributions
        selected_indices = sorted_indices[:n_select]
        
        return selected_indices

    def energy_and_random_selection(self, weight_dht, n_select, energy_ratio=0.7):
        """
        Perform a combination of energy-based and random frequency selection.
        
        Args:
            weight_dht: The DHT-transformed weights.
            n_select: Total number of frequencies to select.
            energy_ratio: The ratio of frequencies to select based on energy.
        
        Returns:
            Indices of selected frequencies.
        """
        # Compute energy of the DHT coefficients
        energy = self.compute_energy(weight_dht)
        
        # Flatten the energy tensor and sort it based on magnitude
        flat_energy = energy.flatten()
        sorted_energy, sorted_indices = torch.sort(flat_energy, descending=True)
        
        # Number of frequencies to select based on energy
        n_energy_select = int(n_select * energy_ratio)
        n_random_select = n_select - n_energy_select
        
        # Step 1: Select top frequencies based on energy
        energy_selected_indices = sorted_indices[:n_energy_select]

        # Step 2: Select remaining frequencies randomly from the unselected frequencies
        remaining_indices = sorted_indices[n_energy_select:]  # Exclude already selected by energy
        random_indices = torch.randperm(remaining_indices.size(0), generator=torch.Generator().manual_seed(args_l.entry_seed))[:n_random_select]
        random_selected_indices = remaining_indices[random_indices]

        # Combine energy-based selected and randomly selected indices
        combined_indices = torch.cat([energy_selected_indices, random_selected_indices])

        return combined_indices


    def energy_and_cumulative_random_selection(self, weight_dht, n_select, energy_threshold=0.95):
        """
        Perform a combination of energy-based selection using cumulative energy function
        and random selection from the remaining frequencies.
        
        Args:
            weight_dht: The DHT-transformed weights.
            n_select: The total number of frequencies to select.
            energy_threshold: Threshold for cumulative energy selection (e.g., 0.95 for 95%).
        
        Returns:
            Indices of selected frequencies.
        """
        # Compute energy of the DHT coefficients
        energy = self.compute_energy(weight_dht)
        
        # Flatten the energy tensor and sort it based on magnitude
        flat_energy = energy.flatten()
        sorted_energy, sorted_indices = torch.sort(flat_energy, descending=True)
        
        # Cumulative sum of energy to find the cutoff point where the energy threshold is met
        cumulative_energy = torch.cumsum(sorted_energy, dim=0)
        total_energy = cumulative_energy[-1]
        
        # Select indices where cumulative energy reaches the desired threshold
        cutoff_index = (cumulative_energy / total_energy >= energy_threshold).nonzero(as_tuple=True)[0][0]
        energy_selected_indices = sorted_indices[:cutoff_index]

        # Cast indices to long
        energy_selected_indices = energy_selected_indices.long()

        # Step 1: Select as many top energy frequencies as possible, limited by n_select
        n_energy_select = min(cutoff_index, n_select)
        energy_selected_indices = sorted_indices[:n_energy_select].long()  # Ensure indices are long
        
        # Step 2: Randomly select the remaining frequencies from the unselected part
        n_random_select = n_select - n_energy_select
        if n_random_select > 0:
            remaining_indices = sorted_indices[n_energy_select:].long()  # Cast remaining indices to long
            random_indices = torch.randperm(remaining_indices.size(0), generator=torch.Generator().manual_seed(args_l.entry_seed))[:n_random_select]
            random_selected_indices = remaining_indices[random_indices]
        else:
            random_selected_indices = torch.tensor([]).to(energy_selected_indices.device).long()  # No random selection if all selected by energy

        # Combine energy-based selected and randomly selected indices
        combined_indices = torch.cat([energy_selected_indices, random_selected_indices])

        return combined_indices


    def initialize_spectral_indices(self, n_frequency, energy_threshold=0.95, energy_ratio=0.8):
        """
        Initialize the indices of frequencies to be used based on DHT, with part energy-based and part random selection.
        
        Args:
            n_frequency: The total number of frequencies to select.
            energy_threshold: Threshold for selecting frequencies based on their contribution to energy.
            energy_ratio: The proportion of frequencies selected based on energy.
        
        Returns:
            2D indices of selected frequencies.
        """
        # Apply DHT to the base layer's weights
        weight_dht =self.dht2(self.base_layer.weight.data)
        
        # Perform hybrid energy and random frequency selection
        selected_indices = self.energy_and_cumulative_random_selection(weight_dht, n_frequency, energy_ratio)

        # Convert flat indices to 2D indices (for matrix-style operations)
        indices_2d = torch.stack([selected_indices // self.in_features, selected_indices % self.in_features], dim=0)
        return indices_2d


    def energy_based_selection(self, weight_dht, n_select, energy_threshold):
        """
        Perform energy-based frequency selection on the DHT-transformed weights.
        
        Args:
            weight_dht: The DHT-transformed weights.
            n_select: The number of frequencies to select.
            energy_threshold: Threshold to keep the most significant frequencies based on energy.
        
        Returns:
            Indices of selected frequencies that contribute to the most energy.
        """
        # Compute energy of the DHT coefficients
        energy = self.compute_energy(weight_dht)
        
        # Flatten the energy tensor and sort it based on magnitude
        flat_energy = energy.flatten()
        sorted_energy, sorted_indices = torch.sort(flat_energy, descending=True)
        
        # Cumulative sum of energy to find the cutoff point where the energy threshold is met
        cumulative_energy = torch.cumsum(sorted_energy, dim=0)
        total_energy = cumulative_energy[-1]
        
        # Select indices where cumulative energy reaches the desired threshold or number of frequencies is reached
        cutoff_index = (cumulative_energy / total_energy >= energy_threshold).nonzero(as_tuple=True)[0][0]
        selected_indices = sorted_indices[:min(cutoff_index, n_select)]
        print(f"cutoff_index is {cutoff_index}")
        
        return selected_indices

    # def initialize_spectral_indices(self, n_frequency, energy_threshold=0.5):
    #     """
    #     Initialize the indices of frequencies to be used based on DHT and energy-based selection.
        
    #     Args:
    #         n_frequency: The total number of frequencies to select.
    #         energy_threshold: Threshold for selecting frequencies based on their contribution to energy.
        
    #     Returns:
    #         2D indices of selected frequencies.
    #     """
    #     # Apply DHT to the base layer's weights
    #     weight_dht = self.dht2(self.base_layer.weight.data)
        
    #     # Perform energy-based frequency selection
    #     selected_indices = self.energy_based_selection1(weight_dht, n_frequency, energy_threshold)

    #     # Convert flat indices to 2D indices (for matrix-style operations)
    #     indices_2d = torch.stack([selected_indices // self.in_features, selected_indices % self.in_features], dim=0)
    #     return indices_2d


    def update_layer(self, adapter_name, n_frequency, scale, init_fourier_weights=None):
        if n_frequency <= 0:
            raise ValueError(f"`n_frequency` should be a positive integer value but the value passed is {n_frequency}")
        self.n_frequency[adapter_name] = n_frequency
        self.scale[adapter_name] = scale
        # if n_frequency > 0:
        #     if args_l.share_entry:
        #         self.indices[adapter_name] = torch.randperm(self.in_features * self.in_features, generator=torch.Generator().manual_seed(args_l.entry_seed))[:n_frequency]
        #         print('\033[32m Using shared entry... \033[0m')
        #     else:
        #         self.indices[adapter_name] = torch.randperm(self.in_features * self.in_features)[:n_frequency]

        self.indices[adapter_name] = self.initialize_spectral_indices(n_frequency)
            
        #self.indices[adapter_name] = torch.stack([self.indices[adapter_name] // self.in_features, self.indices[adapter_name] % self.in_features], dim=0)
        self.spectrum[adapter_name] = nn.Parameter(torch.randn(n_frequency), requires_grad=True)


  
        weight = getattr(self.get_base_layer(), "weight", None)
        if weight is not None:
            self.to(weight.device, dtype=weight.dtype)
        self.set_adapter(self.active_adapters)


    # Efficient 1D DHT with optional normalization
    def dht1(self, x, norm=None):
        N = x.size(-1)
        # Compute the FFT of x
        X = torch.fft.fft(x, dim=-1)
        # Compute the DHT using real and imaginary parts
        H = X.real - X.imag
        # Apply normalization if required
        if norm == 'ortho':
            H = H / math.sqrt(N)
        else:
            H = H / N
        return H

    # Efficient 2D DHT with optional normalization
    def dht2(self, x, norm=None):
        x = self.dht1(x, norm=norm)
        x = self.dht1(x.transpose(-1, -2), norm=norm).transpose(-1, -2)

        return x




    # Efficient 1D IDHT with optional normalization
    def idht1(self, H, norm=None):
        # The inverse DHT is the same as the forward DHT
        return self.dht1(H, norm=norm)

    # Efficient 2D IDHT with optional normalization
    def idht2(self, H, norm=None):
        return self.dht2(H, norm=norm)


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
        delta_w_means_before, delta_w_vars_before = [], []
        delta_w_means_after, delta_w_vars_after = [], []

        result = self.base_layer(x, *args, **kwargs)
        for active_adapter in self.active_adapters:
            if active_adapter not in self.spectrum.keys():
                continue
            
            spectrum = self.spectrum[active_adapter]
            indices = self.indices[active_adapter].to(spectrum.device)
            scale = self.scale[active_adapter]

            dense_s = torch.zeros((self.in_features, self.in_features), dtype=spectrum.dtype, device=spectrum.device)



            # nn.init.normal_(dense_s, mean=0.0, std=0.02)
            dense_s[indices[0, :], indices[1, :]] = spectrum

            if spectrum.dtype == torch.bfloat16:
                dense_s = dense_s.to(torch.float16)


            #delta_w = self.idst2(dense_s) 
            delta_w = self.idht2(dense_s) 

            delta_w_mean_before = torch.mean(delta_w).item()
            delta_w_var_before = torch.var(delta_w).item()

            delta_w = delta_w * 49

            # After scaling
            delta_w_mean_after = torch.mean(delta_w).item()
            delta_w_var_after = torch.var(delta_w).item()


            # Collect the means and variances
            delta_w_means_before.append(delta_w_mean_before)
            delta_w_vars_before.append(delta_w_var_before)
            delta_w_means_after.append(delta_w_mean_after)
            delta_w_vars_after.append(delta_w_var_after)


            x, delta_w = x.to(spectrum.dtype), delta_w.to(spectrum.dtype)
            result += torch.einsum('ijk,kl->ijl', x, delta_w)

        self.delta_w_stats = {
            "mean_before": np.mean(delta_w_means_before),
            "var_before": np.mean(delta_w_vars_before),
            "mean_after": np.mean(delta_w_means_after),
            "var_after": np.mean(delta_w_vars_after)
        }

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "fourier." + rep

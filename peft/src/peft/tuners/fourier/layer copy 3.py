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

    def update_layer(self, adapter_name, n_frequency, scale, init_fourier_weights=None):
        if n_frequency <= 0:
            raise ValueError(f"`n_frequency` should be a positive integer value but the value passed is {n_frequency}")
        self.n_frequency[adapter_name] = n_frequency
        self.scale[adapter_name] = scale
        if n_frequency > 0:
            if args_l.share_entry:
                self.indices[adapter_name] = torch.randperm(self.in_features * self.in_features, generator=torch.Generator().manual_seed(args_l.entry_seed))[:n_frequency]
                print('\033[32m Using shared entry... \033[0m')
            else:
                self.indices[adapter_name] = torch.randperm(self.in_features * self.in_features)[:n_frequency]
                
            self.indices[adapter_name] = torch.stack([self.indices[adapter_name] // self.in_features, self.indices[adapter_name] % self.in_features], dim=0)
            self.spectrum[adapter_name] = nn.Parameter(torch.randn(n_frequency), requires_grad=True)
            #nn.init.normal_(self.spectrum[adapter_name], mean=0.0, std=0.02)
            #nn.init.uniform_(self.spectrum[adapter_name], a=-0.1, b=0.1)
            # gain = 1  # You can experiment with different values, 1 works well for orthogonal transforms
            # nn.init.kaiming_uniform_(self.spectrum[adapter_name], a=gain)

  
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
        #x = torch.relu(x)
        # x = F.gelu(x) 
        #x = F.leaky_relu(x, negative_slope=0.01)

        # Apply a frequency weighting mask
        # freq_weights = self.get_frequency_weights(x.size(-1), device=x.device)
        # x = x * freq_weights

        return x

    def get_frequency_weights(self, N, device):
        # Create a frequency weighting mask emphasizing lower frequencies
        freq_indices = torch.arange(N, device=device)
        freq_weights = 1 / (1 + freq_indices.float())
        return freq_weights


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
            #nn.init.kaiming_uniform_(dense_s, a=math.sqrt(5))
            # gain = 5  # You can experiment with different values, 1 works well for orthogonal transforms
            # nn.init.kaiming_uniform_(dense_s, a=gain)
            # nn.init.xavier_uniform_(dense_s)


            # nn.init.normal_(dense_s, mean=0.0, std=0.02)
            dense_s[indices[0, :], indices[1, :]] = spectrum

            if spectrum.dtype == torch.bfloat16:
                dense_s = dense_s.to(torch.float16)


            #delta_w = self.idst2(dense_s) 
            delta_w = self.idht2(dense_s) 

            delta_w_mean_before = torch.mean(delta_w).item()
            delta_w_var_before = torch.var(delta_w).item()

            delta_w = delta_w * scale


            # if delta_w.requires_grad:
            #     print(f"Gradient is flowing through delta_w: {delta_w.requires_grad}")
            # else:
            #     print("Warning: No gradient is flowing through delta_w")


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

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
  
        weight = getattr(self.get_base_layer(), "weight", None)
        if weight is not None:
            self.to(weight.device, dtype=weight.dtype)
        self.set_adapter(self.active_adapters)

    # Efficient 1D DST-II using PyTorch's FFT and padding
    def dst(self, x, norm=None):
        N = x.size(-1)
        # Create a symmetrically extended sequence
        x_extended = torch.cat([torch.zeros_like(x[..., :1]), x, torch.zeros_like(x[..., :1]), -x.flip([-1])], dim=-1)
        # Compute the FFT
        X = torch.fft.fft(x_extended, dim=-1)
        # Extract the DST-II coefficients (imaginary parts)
        Y = -X[..., 1:N+1].imag
        # Apply normalization if required
        if norm == 'ortho':
            Y = Y * math.sqrt(2 / (N + 1))
        else:
            Y = Y * 2
        return Y

    # Efficient 1D IDST-II using PyTorch's IFFT and padding
    def idst(self, X, norm=None):
        N = X.size(-1)
        # Apply normalization if required
        if norm == 'ortho':
            X = X * math.sqrt(2 / (N + 1))
        else:
            X = X / 2
        # Construct the extended sequence for IFFT
        zeros = torch.zeros_like(X[..., :1])
        X_extended = torch.cat([zeros, -X, zeros, X.flip([-1])], dim=-1)
        # Compute the IFFT
        x = torch.fft.ifft(X_extended, dim=-1)
        # Extract the real part
        x_real = x[..., 1:N+1].real
        return x_real


    # Efficient 2D DST-II
    def dst2(self, x, norm=None):
        x = self.dst(x, norm=norm)
        x = self.dst(x.transpose(-1, -2), norm=norm).transpose(-1, -2)
        return x

    def idst2(self, X, norm=None):
        X = self.idst(X, norm=norm)
        X = self.idst(X.transpose(-1, -2), norm=norm).transpose(-1, -2)
        return X



        # Efficient 1D DCT-II using PyTorch's FFT and padding
    def dct(self, x, norm=None):
        N = x.size(-1)
        x = torch.cat([x, x.flip([-1])], dim=-1)  # Padding with the flipped version of the input
        X = torch.fft.fft(x, dim=-1)  # Compute FFT along the last dimension
        Y = X[..., :N].real  # Keep only the real part for DCT
        if norm == 'ortho':  # Orthogonal normalization
            Y = Y * math.sqrt(2 / N)
            Y[..., 0] = Y[..., 0] / math.sqrt(2)  # First element needs different scaling
        return Y

    # Efficient 1D IDCT-II using PyTorch's IFFT and padding
    def idct(self,X, norm=None):
        N = X.size(-1)
        if norm == 'ortho':
            X = X * math.sqrt(2 / N)
            X[..., 0] = X[..., 0] * math.sqrt(2)  # Reverse scaling for the first element
        x = torch.cat([X, X[..., 1:].flip([-1])], dim=-1)  # Symmetrically pad for IFFT
        x = torch.fft.ifft(x, dim=-1).real  # Compute IFFT and keep the real part
        return x[..., :N]

    # Efficient 2D DCT-II
    def dct2(self, x, norm=None):
        # Apply DCT-II on both dimensions
        x = self.dct(x.transpose(-1, -2), norm=norm).transpose(-1, -2)  # DCT along rows
        x = self.dct(x, norm=norm)  # DCT along columns
        return x

    # Efficient 2D IDCT-II
    def idct2(self, X, norm=None):
        # Apply IDCT-II on both dimensions
        X = self.idct(X.transpose(-1, -2), norm=norm).transpose(-1, -2)  # IDCT along rows
        X = self.idct(X, norm=norm)  # IDCT along columns
        return X


    # Efficient 1D DHT with optional normalization
    def dht1(self, x, norm=None):
        N = x.size(-1)
        # Generate the cas (cos + sin) matrix
        n = torch.arange(N, device=x.device).float().unsqueeze(0)
        k = torch.arange(N, device=x.device).float().unsqueeze(1)
        theta = (2 * math.pi * k * n) / N
        cas_matrix = torch.cos(theta) + torch.sin(theta)  # cas(theta) = cos(theta) + sin(theta)
        
        # Perform the DHT by matrix multiplication
        result = torch.matmul(x, cas_matrix)
        
        # Apply normalization if required
        if norm == 'ortho':
            result = result * (1 / math.sqrt(N))
        
        return result

    # Efficient 2D DHT with optional normalization
    def dht2(self, x, norm=None):
        # Apply 1D DHT along rows (dimension -1) and then along columns (dimension -2)
        x = self.dht1(x.transpose(-1, -2), norm=norm).transpose(-1, -2)  # DHT along rows
        x = self.dht1(x, norm=norm)  # DHT along columns
        return x

    # Efficient 1D IDHT with optional normalization
    def idht1(self, X, norm=None):
        # Inverse DHT is the same as forward DHT
        return self.dht1(X, norm=norm)

    # Efficient 2D IDHT with optional normalization
    def idht2(self, X, norm=None):
        # Apply 1D IDHT along rows (dimension -1) and then along columns (dimension -2)
        X = self.idht1(X.transpose(-1, -2), norm=norm).transpose(-1, -2)  # IDHT along rows
        X = self.idht1(X, norm=norm)  # IDHT along columns
        return X


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
            dense_s[indices[0, :], indices[1, :]] = spectrum

            if spectrum.dtype == torch.bfloat16:
                dense_s = dense_s.to(torch.float16)


            delta_w = self.idst2(dense_s) 
            #delta_w = self.idht2(dense_s) 

            delta_w_mean_before = torch.mean(delta_w).item()
            delta_w_var_before = torch.var(delta_w).item()

            delta_w = delta_w * 1.5e-1

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

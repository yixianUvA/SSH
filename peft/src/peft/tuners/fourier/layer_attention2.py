import time
import pdb
import math
import warnings
from typing import Any, List, Optional, Union
from args import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as dct



from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.other import transpose
args_l = get_args()

def FFT_SHIFT(matrix):
    m_clone = matrix.clone()
    m, n = m_clone.shape
    m = int(m / 2)
    n = int(n / 2)

    for i in range(m):
        for j in range(n):
            m_clone[i][j] = matrix[m + i][n + j]
            m_clone[m + i][n + j] = matrix[i][j]
            m_clone[m + i][j] = matrix[i][j + n]
            m_clone[i][j + n] = matrix[m + i][j]
    return m_clone

class FourierLayer(BaseTunerLayer):
    adapter_layer_names = ["spectrum"]

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.n_frequency = {}
        self.scale = {}
        self.spectrum = nn.ParameterDict({})
        self.indices = {}
        self.attention_weights = nn.ParameterDict({})  # Initialize attention weights
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
            if args_l.set_bias:
                d = self.in_features
                center_frequency = args_l.fc  # D_0 
                width = args_l.width   # W
                order = 2 # 2n
                rows, cols = np.ogrid[:d, :d]
                distance = np.sqrt((rows - d / 2)**2 + (cols - d / 2)**2)
                mask_gs = torch.tensor(np.exp(-(distance * width / (distance**2 - center_frequency**2))**(-2)))
                mask_gs = FFT_SHIFT(mask_gs)
                samples = torch.multinomial(mask_gs.view(-1), 1000, replacement=True)
                samples = torch.stack([samples // d, samples % d], dim=1).T
                self.indices[adapter_name] = samples
                print('\033[32m Using frequency bias... \033[0m')
            elif args_l.share_entry:
                self.indices[adapter_name] = torch.randperm(self.in_features * self.in_features, generator=torch.Generator().manual_seed(args_l.entry_seed))[:n_frequency]
                print('\033[32m Using shared entry... \033[0m')
            else:
                self.indices[adapter_name] = torch.randperm(self.in_features * self.in_features)[:n_frequency]
                
            self.indices[adapter_name] = torch.stack([self.indices[adapter_name] // self.in_features, self.indices[adapter_name] % self.in_features], dim=0)
            self.spectrum[adapter_name] = nn.Parameter(torch.randn(n_frequency), requires_grad=True)
            self.attention_weights[adapter_name] = nn.Parameter(torch.ones(self.in_features), requires_grad=True)  # Initialize attention weights with correct size
  
        weight = getattr(self.get_base_layer(), "weight", None)
        if weight is not None:
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)

    def compute_attention_scores(self, adapter_name, x):
        spectrum = self.spectrum[adapter_name]
        indices = self.indices[adapter_name].to(spectrum.device)
        dense_s = torch.sparse_coo_tensor(indices, spectrum, torch.Size([self.in_features, self.in_features])).to_dense()
        attention_scores = torch.matmul(x, dense_s.T)
        return attention_scores

    # def apply_attention_weights(self, adapter_name, attention_scores):
    #     try:
    #         attention_weights = self.attention_weights[adapter_name]
    #         attention_weights = F.softmax(attention_weights, dim=0)
            
    #         # Debugging: Print the shapes
    #         print(f"attention_scores.shape: {attention_scores.shape}")
    #         print(f"attention_weights.shape: {attention_weights.shape}")
            
    #         # Adjust dimensions for broadcasting
    #         if attention_scores.dim() == 3:
    #             weighted_attention_scores = attention_scores * attention_weights.unsqueeze(0).unsqueeze(0)
    #         else:
    #             weighted_attention_scores = attention_scores * attention_weights.unsqueeze(0)
                
    #         return weighted_attention_scores
    #     except Exception as e:
    #         print(f"Error in apply_attention_weights: {e}")
    #         print(f"attention_scores.shape: {attention_scores.shape}")
    #         print(f"attention_weights.shape: {attention_weights.shape}")
    #         raise
    def apply_attention_weights(self, adapter_name, attention_scores):
        attention_weights = self.attention_weights[adapter_name]
        attention_weights = F.softmax(attention_weights, dim=0)
        
        # Reshape attention_weights to match the input shape
        attention_weights = attention_weights.view(1, 1, -1)
        
        weighted_attention = attention_scores * attention_weights
        return weighted_attention
            
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
    ) -> None:
        super().__init__()
        FourierLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, n_frequency, scale, init_fourier_weights)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )

        if adapter_names is None:
            adapter_names = self.active_adapters

        for active_adapter in adapter_names:
            if active_adapter in self.spectrum.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.spectrum.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        device = self.spectrum[adapter].device
        dtype = self.spectrum[adapter].dtype

        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        spectrum = self.spectrum[adapter]
        indices = self.indices[adapter].to(spectrum.device)

        dense_s = torch.zeros((self.in_features, self.in_features), dtype=spectrum.dtype, device=spectrum.device)
        dense_s[indices[0, :], indices[1, :]] = spectrum

        if spectrum.dtype == torch.bfloat16:
            dense_s = dense_s.to(torch.float16)

        weight = self.idct2(dense_s) * self.scale[adapter]

        if cast_to_fp32:
            weight = weight.float()

        output_tensor = weight

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)
            self.weight[adapter] = weight.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
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
                
                # attention_scores = self.compute_attention_scores(active_adapter, x)
                # weighted_delta_w = self.apply_attention_weights(active_adapter, attention_scores)
                attention_scores = self.compute_attention_scores(active_adapter, x)
                weighted_attention = self.apply_attention_weights(active_adapter, attention_scores)
                
                # # Ensure dimensions align correctly for the einsum operation
                # if weighted_delta_w.dim() == 3:
                #     weighted_delta_w = weighted_delta_w.mean(dim=0)  # Adjust dimension if needed

                # # Adjust dimensions for einsum
                # if weighted_delta_w.size(1) != delta_w.size(1):
                #     weighted_delta_w = weighted_delta_w.permute(0, 2, 1)
                    
                # Ensure weighted_delta_w is 3D
                # if weighted_delta_w.dim() == 2:
                #     weighted_delta_w = weighted_delta_w.unsqueeze(0)
                
                # # Adjust dimensions for einsum
                # if weighted_delta_w.size(1) != delta_w.size(0):
                #     weighted_delta_w = weighted_delta_w.transpose(1, 2)
                    
                # Compute adaptation
                adaptation = torch.einsum('bi,ij->bj', x.view(-1, self.in_features), delta_w)
                adaptation = adaptation.view(x.size(0), x.size(1), -1)
                
                # Apply weighted attention
                weighted_adaptation = adaptation * weighted_attention
            
                result += weighted_adaptation


        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "fourier." + rep

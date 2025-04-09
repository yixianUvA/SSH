import time
import math
import warnings
from typing import Any, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from pytorch_wavelets import DWTForward, DWTInverse, DTCWTForward, DTCWTInverse
from haar_pytorch import HaarForward, HaarInverse


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
        # self.dwt = DWTForward(J=1, wave='haar', mode='zero')
        #self.idwt = DWTInverse(J=1,wave='haar', mode='zero')
        # self.dwt = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b')
        # self.idwt = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')

        # self.dwt = DWTForward(J=3, wave='haar')
        # self.idwt = DWTInverse(wave='haar')
        self.dwt =  HaarForward()
        self.idwt = HaarInverse()

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
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)



    # def idwt2(self, Yl, Yh):
    #     return self.idwt((Yl, Yh))

    # def idwt2(self, x):
    #     return self.idwt(x.unsqueeze(0).unsqueeze(0))

    def idwt2(self, x):
        # Ensure the input has a number of channels divisible by 4
        x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        # Check the shape of the tensor
        _, channels, _, _ = x.shape

        # If channels are not divisible by 4, pad the tensor
        if channels % 4 != 0:
            pad_size = 4 - (channels % 4)
            x = F.pad(x, (0, 0, 0, 0, 0, pad_size))

        return self.idwt(x)





    
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

        dense_s = torch.sparse.FloatTensor(indices, spectrum, torch.Size([self.in_features, self.in_features]))

        if spectrum.dtype == torch.bfloat16:
            dense_s = dense_s.to(torch.float16)

        dense_s = dense_s  # Adding batch and channel dimensions
        Yl, Yh = self.dwt(dense_s)
        weight = self.idwt2(Yl, Yh) * 300

        if cast_to_fp32:
            weight = weight.float()

        #output_tensor = weight
        output_tensor = weight.squeeze(0).squeeze(0)

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

                #delta_w = self.idct2(dense_s) * scale

                #dense_s = dense_s.unsqueeze(0).unsqueeze(0)  # Adding batch and channel dimensions
                # Yl, Yh = self.dwt(dense_s)
                # delta_w = self.idwt2(dense_s) * scale
                # delta_w = delta_w.squeeze(0).squeeze(0) 
                delta_w = self.idwt2(dense_s).squeeze(0).squeeze(0)
                delta_w = delta_w* scale 

                x, delta_w = x.to(spectrum.dtype), delta_w.to(spectrum.dtype)
                print(f"x shape: {x.shape}, delta_w shape: {delta_w.shape}")
                result += torch.einsum('ijk,kl->ijl', x, delta_w)
                

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "fourier." + rep

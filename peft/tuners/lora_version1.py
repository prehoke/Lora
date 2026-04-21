# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import PeftConfig, PeftType, transpose
from ..utils.sparsegen import GlobalSparsegen
from transformers.utils import ModelOutput


@dataclass
class LoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.Lora`].

    Args:
        r (`int`): Lora attention dimension
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        merge_weights (`bool`):
            Whether to merge the weights of the Lora layers with the base transformer model in `eval` mode.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        enable_lora ( `List[bool]`): Used with `lora.MergedLinear`.
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=32, metadata={"help": "Lora alpha"})
    lora_nums: int = field(default=2, metadata={"help": "Numbers of Lora"})
    blc_alpha: int = field(default=0, metadata={"help": "Alpha of blcloss"})
    blc_weight: int = field(default=0, metadata={"help": "Weight of blcloss"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    sparsegen: bool = field(
        default=False,
        metadata={
            "help": "Whether to use sparsegen. If True, the model will be trained with sparsegen."
            "If False, the model will be trained with softmax."
        },
    )
    sparsegen_init: str = field(
        default="kaiming",
        metadata={
            "help": "Initialization strategy for sparsegen MLP. Options: 'kaiming', 'dense', 'sparse', 'zeros'."
            "'sparse' initializes for sparse routing (λ≈1, high threshold), 'dense' initializes for dense routing (λ≈0, low threshold)."
        },
    )
    lora_freeze: bool = field(
        default=False,
        metadata={"help": "Whether to freeze LoRA modules and only train the router"}
    )
    router_freeze: bool = field(
        default=False,
        metadata={"help": "Whether to freeze router and only train the LoRA modules"}
    )
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the Lora model"}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    enable_lora: Optional[List[bool]] = field(default=None, metadata={"help": "Used with `lora.MergedLinear`."})
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    sparsegen_cfg: Optional[Dict[str, Union[List[int], int]]] = field(
        default=None,
        metadata={
            "help": "Configuration for sparsegen. If provided, it should contain 'input_sizes' and 'hidden_size'. "
            "'input_sizes' is a list of input sizes for each layer, and 'hidden_size' is the hidden size for the sparsegen MLP."
        },
    )
    continuous_mole: bool = field(
        default=False,
        metadata={"help": "Use Continuous MoLE instead of discrete multi-LoRA experts."},
    )
    cmole_rank_k: int = field(
        default=2,
        metadata={"help": "Low-rank residual rank k for M = D + U V^T."},
    )
    cmole_router_hidden: int = field(
        default=128,
        metadata={"help": "Hidden size of the Continuous MoLE router."},
    )
    cmole_use_lowrank: bool = field(
        default=True,
        metadata={"help": "Whether to use the U V^T residual in Continuous MoLE."},
    )
    cmole_diag_scale: float = field(
        default=0.1,
        metadata={"help": "Scale for diagonal modulation around identity: d = 1 + s * tanh(raw_d)."},
    )
    cmole_lowrank_scale: float = field(
        default=0.01,
        metadata={"help": "Scale for low-rank factors U and V at initialization/forward."},
    )

    def __post_init__(self):
        self.peft_type = PeftType.LORA


@dataclass
class LoraOutput(ModelOutput):
    results: Optional[torch.FloatTensor] = None
    routing_weight: Optional[torch.FloatTensor] = None
    gate_score: Optional[torch.FloatTensor] = None
    lam : Optional[torch.FloatTensor] = None


class LoraModel(torch.nn.Module):
    """
    Creates Low Rank Adapter (Lora) model from a pretrained transformers model.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM, LoraConfig >>> from peft import LoraModel, LoraConfig >>>
        config = LoraConfig(
            peft_type="LORA", task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, target_modules=["q", "v"],
            lora_dropout=0.01, )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> lora_model = LoraModel(config, model)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    def __init__(self, config, model): # LoraConfig, CasualLM
        super().__init__()
        self.peft_config = config
        self.model = model
        self.sparsegen_fn = None
        
        if self.peft_config.sparsegen_cfg["enabled"]:
            input_sizes = self.peft_config.sparsegen_cfg["input_sizes"]
            hidden_size = self.peft_config.sparsegen_cfg["hidden_sizes"]         
            
            self.sparsegen_fn = GlobalSparsegen(
                input_sizes=input_sizes,
                hidden_size=hidden_size,
                init_strategy=self.peft_config.sparsegen_init,
            )

        self._find_and_replace()

        mark_only_lora_as_trainable(
            self.model, self.peft_config.bias, 
            self.peft_config.lora_freeze,
            self.peft_config.router_freeze, 
            True if self.sparsegen_fn else False
        )
        self.forward = self.model.forward

    def _find_and_replace(self):
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if (loaded_in_4bit or loaded_in_8bit):
            raise ImportError(
                "To use Lora with 8-bit or 4-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        
        # Get total number of layers from the model config
        total_layers = getattr(self.model.config, 'num_hidden_layers', 12)
        
        kwargs = {
            "sparsegen_fn": self.sparsegen_fn,
            "r": self.peft_config.r,
            "lora_alpha": self.peft_config.lora_alpha,
            "lora_dropout": self.peft_config.lora_dropout,
            "lora_nums": self.peft_config.lora_nums,
            "blc_alpha": self.peft_config.blc_alpha,
            "blc_weight": self.peft_config.blc_weight,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
            "merge_weights": (self.peft_config.merge_weights or self.peft_config.inference_mode)
            and not is_hf_device_map_available,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)
            if target_module_found: # here
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None

                # Extract layer index from the module key
                layer_idx = self._extract_layer_index(key)

                # import pdb; pdb.set_trace();
                if key.endswith("q_proj"):
                    proj_heads = getattr(self.model.config, "num_attention_heads")
                elif key.endswith("v_proj"):
                    proj_heads = getattr(
                        self.model.config,
                        "num_key_value_heads",
                        getattr(self.model.config, "num_attention_heads"),
                    )
                else:
                    proj_heads = getattr(self.model.config, "num_attention_heads")
                if self.peft_config.continuous_mole:
                    new_module = ContinuousMoLELinear(
                        target.in_features,
                        target.out_features,
                        bias=bias,
                        layer_idx=layer_idx,
                        total_layers=total_layers,
                        num_heads=proj_heads,
                        num_key_value_heads=getattr(
                            self.model.config, "num_key_value_heads", proj_heads
                        ),
                        cmole_rank_k=self.peft_config.cmole_rank_k,
                        cmole_router_hidden=self.peft_config.cmole_router_hidden,
                        cmole_use_lowrank=self.peft_config.cmole_use_lowrank,
                        cmole_diag_scale=self.peft_config.cmole_diag_scale,
                        cmole_lowrank_scale=self.peft_config.cmole_lowrank_scale,
                        **kwargs,
                    )

                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def unfreeze_sparsegen_for_training(self):
        """Unfreeze sparsegen parameters for training"""
        if self.peft_config.sparsegen:
            for n, p in self.model.named_parameters():
                if "sparsegen" in n:
                    p.requires_grad = True
    
    def freeze_sparsegen_for_eval(self):
        """Freeze sparsegen parameters for evaluation"""
        if self.peft_config.sparsegen:
            for n, p in self.model.named_parameters():
                if "sparsegen" in n:
                    p.requires_grad = False

    def _extract_layer_index(self, key):
        """Extract layer index from module key like 'encoder.layer.0.attention.self.query'"""
        import re
        # Look for patterns like 'layer.X' or 'layers.X' or 'h.X' (for different model architectures)
        patterns = [
            r'\.layer\.(\d+)\.',  # BERT-style: encoder.layer.0.attention
            r'\.layers\.(\d+)\.',  # Some models use 'layers'
            r'\.h\.(\d+)\.',       # GPT-style: transformer.h.0.attn
            r'encoder\.(\d+)\.',   # Direct encoder numbering
            r'decoder\.(\d+)\.',   # Direct decoder numbering
        ]
       
        for pattern in patterns:
            match = re.search(pattern, key)
            if match:
                return int(match.group(1))
        
        # If no layer index found, return 0 as default
        return 0


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `lora_only` to work
def mark_only_lora_as_trainable(
    model: nn.Module, 
    bias: str = "none", 
    lora_freeze: bool = False,  
    router_freeze: bool = False, 
    sparsegen: bool=False) -> None:
    for n, p in model.named_parameters():
        # include route_weight
        if (
        "lora_" not in n
        and "to_d" not in n
        and "to_u" not in n
        and "to_v" not in n
        ):
            p.requires_grad = False
        if lora_freeze:
            if "lora_" in n:
                p.requires_grad = False

        if not router_freeze:
            if "lora_route" in n or "to_d" in n or "to_u" in n or "to_v" in n:
                p.requires_grad = True
        if sparsegen:
            if "sparsegen" in n:
                # For evaluation, freeze sparsegen parameters by default
                p.requires_grad = True

    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class LoraLayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False

class ContinuousMoLELinear(nn.Linear, LoraLayer):
    """
    Continuous MoLE version of a LoRA linear layer.

    It replaces discrete expert routing:
        sum_i route_i(x) * B_i A_i
    with a continuous latent modulation:
        B_h M_{b,s,h} A

    where
        M_{b,s,h} = Diag(d_{b,s,h}) + U_{b,s,h} V_{b,s,h}^T
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):  
        # self.use_lowrank = kwargs.get("cmole_use_lowrank", True)
        nn.Linear.__init__(self, in_features, out_features, bias=kwargs.get("bias", True))
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        self.weight.requires_grad = False

        # ----- infer head layout from model config -----
        # passed from _find_and_replace
        self.num_heads = kwargs["num_heads"]
        self.num_key_value_heads = kwargs.get("num_key_value_heads", self.num_heads)

        # q_proj uses num_heads; v_proj uses num_key_value_heads
        # to keep replacement generic, caller passes the right num_heads for this module
        assert out_features % self.num_heads == 0, (
            f"out_features={out_features} must be divisible by num_heads={self.num_heads}"
        )
        self.head_dim = out_features // self.num_heads

        # ----- Continuous MoLE hyperparams -----
        self.rank_k = kwargs.get("cmole_rank_k", 2)
        self.router_hidden = kwargs.get("cmole_router_hidden", 128)
        self.use_lowrank = kwargs.get("cmole_use_lowrank", True)
        self.diag_scale = kwargs.get("cmole_diag_scale", 0.1)
        self.lowrank_scale = kwargs.get("cmole_lowrank_scale", 0.01)

        # ----- shared basis A -----
        # A: (r, in_features)
        self.lora_A_shared = nn.Linear(in_features, r, bias=False)

        # ----- head-wise B -----
        # one B_h per head block, implemented as a single parameter:
        # B_heads: (num_heads, head_dim, r)
        self.lora_B_heads = nn.Parameter(torch.zeros(self.num_heads, self.head_dim, r))

        # ----- router -----
        # input z = [x, c_token, c_head]  -> dim = in_features + 2
        self.lora_route = nn.Sequential(
            nn.Linear(in_features + 2, self.router_hidden, bias=False),
            nn.GELU(),
            nn.Linear(self.router_hidden, self.router_hidden, bias=False),
            nn.GELU(),
        )
        self.to_d = nn.Linear(self.router_hidden, r, bias=False)

        if self.use_lowrank:
            self.to_u = nn.Linear(self.router_hidden, r * self.rank_k, bias=False)
            self.to_v = nn.Linear(self.router_hidden, r * self.rank_k, bias=False)

        self.scaling = self.lora_alpha / self.r if self.r > 0 else 1.0

        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

        if hasattr(self, "lora_A_shared"):
            nn.init.kaiming_uniform_(self.lora_A_shared.weight, a=math.sqrt(5))

        if hasattr(self, "lora_B_heads"):
            nn.init.zeros_(self.lora_B_heads)

        if hasattr(self, "to_d"):
            nn.init.zeros_(self.to_d.weight)
        if hasattr(self, "to_u"):
            # nn.init.zeros_(self.to_u.weight)
            nn.init.kaiming_uniform_(self.to_u.weight, a=math.sqrt(5))
        if hasattr(self, "to_v"):
            nn.init.zeros_(self.to_v.weight)
            # nn.init.kaiming_uniform_(self.to_v.weight, a=math.sqrt(5))
        # if self.use_lowrank:
        #     nn.init.zeros_(self.to_u.weight)
        #     nn.init.zeros_(self.to_v.weight)

    def _compute_consensus(self, head_out):
        """
        head_out: [B, S, H, Dh]
        returns:
            c_token: [B, S, 1]
            c_head:  [B, S, H]
        """
        center = head_out.mean(dim=2, keepdim=True)  # [B, S, 1, Dh]
        cos_sim = F.cosine_similarity(head_out, center, dim=-1)  # [B, S, H]
        c_head = 1.0 - cos_sim
        c_token = c_head.mean(dim=2, keepdim=True)  # [B, S, 1]
        return c_token, c_head

    def forward(self, x: torch.Tensor):
        """
        x: [B, S, in_features]
        return: LoraOutput with results [B, S, out_features]
        """
        base = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if self.disable_adapters or self.r == 0 or self.merged:
            return base

        x_drop = self.lora_dropout(x)                                  # [B, S, Din]

        # shared reduction: u = A x
        u = self.lora_A_shared(x_drop)                                 # [B, S, r]

        # cheap proxy for head states:
        # project u to head axis by repeating; then derive consensus from modulated latent carrier
        # shape: [B, S, H, r]
        u_head = u.unsqueeze(2).expand(-1, -1, self.num_heads, -1)

        # use current B_h to create a lightweight head state for consensus
        # head_out_proxy: [B, S, H, Dh]
        head_out_proxy = torch.einsum("hdr,bshr->bshd", self.lora_B_heads, u_head)

        c_token, c_head = self._compute_consensus(head_out_proxy)      # [B,S,1], [B,S,H]

        # router input z = [x, c_token, c_head]
        x_expand = x_drop.unsqueeze(2).expand(-1, -1, self.num_heads, -1)   # [B,S,H,Din]
        c_token_expand = c_token.unsqueeze(2).expand(-1, -1, self.num_heads, -1)  # [B,S,H,1]
        c_head_expand = c_head.unsqueeze(-1)                           # [B,S,H,1]

        z = torch.cat([x_expand, c_token_expand, c_head_expand], dim=-1)  # [B,S,H,Din+2]

        h = self.lora_route(z)                                         # [B,S,H,Hid]

        # diagonal gate
        d_raw = self.to_d(h)                                           # [B,S,H,r]
        d = 1.0 + self.diag_scale * torch.tanh(d_raw)                  # [B,S,H,r]

        # diagonal modulation
        diag_part = d * u_head                                         # [B,S,H,r]

        if self.use_lowrank:
            U = self.lowrank_scale * self.to_u(h).view(
                x.shape[0], x.shape[1], self.num_heads, self.r, self.rank_k
            )                                                          # [B,S,H,r,k]
            V = self.lowrank_scale * self.to_v(h).view(
                x.shape[0], x.shape[1], self.num_heads, self.r, self.rank_k
            )                                                          # [B,S,H,r,k]

            # (U V^T) u = U (V^T u)
            vt_u = torch.einsum("bshrk,bshr->bshk", V, u_head)         # [B,S,H,k]
            lowrank_part = torch.einsum("bshrk,bshk->bshr", U, vt_u)   # [B,S,H,r]
            u_mod = diag_part + lowrank_part                           # [B,S,H,r]
        else:
            U, V = None, None
            u_mod = diag_part

        # head-wise projection by B_h
        # B_heads: [H, Dh, r], u_mod: [B,S,H,r]
        lora_out_h = torch.einsum("hdr,bshr->bshd", self.lora_B_heads, u_mod)  # [B,S,H,Dh]
        lora_result = lora_out_h.reshape(x.shape[0], x.shape[1], self.out_features)

        result = base + lora_result * self.scaling

        return LoraOutput(
            results=result,
            routing_weight=None,   # no discrete expert routing in Continuous MoLE
            gate_score=d,          # expose diagonal gate for logging
            lam=None,
        )

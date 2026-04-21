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
import os
from pathlib import Path

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
    cmole_beta_scale: int = field(
        default=2,
    )
    cmole_use_diag_residual: bool = field(
        default=True,
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
                    if getattr(self.model.config, "architectures") == ['LlamaForCausalLM']:
                        proj_heads = getattr(self.model.config, "num_key_value_heads") * 4
                    else:
                        proj_heads = getattr(self.model.config, "num_attention_heads")
                if self.peft_config.continuous_mole:
                    new_module = ContinuousMoLELinearHouseholder(
                        target.in_features,
                        target.out_features,
                        bias=bias,
                        layer_idx=layer_idx,
                        module_name=key,
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
                        cmole_beta_scale=self.peft_config.cmole_beta_scale,
                        cmole_use_diag_residual= self.peft_config.cmole_use_diag_residual,
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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousMoLELinearHouseholder(nn.Linear, LoraLayer):
    """
    Householder-like upgraded version of your ContinuousMoLELinear.

    Replaces:
        u_mod = d * u_head

    with:
        u_mod = u_head - beta * <u_head, w> * w

    where
        w     : input-conditioned head-specific direction in latent rank space
        beta  : input-conditioned scalar in a small range
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
        nn.Linear.__init__(self, in_features, out_features, bias=kwargs.get("bias", True))
        LoraLayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        self.fan_in_fan_out = fan_in_fan_out
        self.weight.requires_grad = False

        # ----- infer head layout -----
        self.num_heads = kwargs["num_heads"]
        self.num_key_value_heads = kwargs.get("num_key_value_heads", self.num_heads)

        assert out_features % self.num_heads == 0, (
            f"out_features={out_features} must be divisible by num_heads={self.num_heads}"
        )
        self.head_dim = out_features // self.num_heads

        # ----- router / modulation hyperparams -----
        self.router_hidden = kwargs.get("cmole_router_hidden", 128)
        self.use_token_consensus = kwargs.get("cmole_use_token_consensus", True)

        # Householder-like params
        # beta in (-beta_scale, beta_scale); beta_scale=2.0 gives classic Householder-like range
        self.beta_scale = kwargs.get("cmole_beta_scale", 2.0)
        self.beta_init_zero = kwargs.get("cmole_beta_init_zero", True)
        self.w_eps = kwargs.get("cmole_w_eps", 1e-6)

        # optional residual diagonal path if you want to mix old/new behavior
        self.use_diag_residual = kwargs.get("cmole_use_diag_residual", False)
        self.diag_scale = kwargs.get("cmole_diag_scale", 0.1)

        # ----- shared basis A -----
        self.lora_A_shared = nn.Linear(in_features, r, bias=False)

        # ----- head-wise B -----
        self.lora_B_heads = nn.Parameter(torch.zeros(self.num_heads, self.head_dim, r))

        # ----- router -----
        token_in_dim = r + (1 if self.use_token_consensus else 0)

        self.token_router_in = nn.Linear(token_in_dim, self.router_hidden, bias=False)
        self.head_router_in = nn.Linear(1, self.router_hidden, bias=False)

        # learned embedding per head
        self.head_embed = nn.Parameter(torch.zeros(self.num_heads, self.router_hidden))

        self.lora_route_post = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.router_hidden, self.router_hidden, bias=False),
            nn.GELU(),
        )

        # ----- Householder-like outputs -----
        # direction in latent rank space
        self.to_w = nn.Linear(self.router_hidden, r, bias=False)

        # scalar beta per (b,s,h)
        self.to_beta = nn.Linear(self.router_hidden, 1, bias=False)

        # optional diagonal residual gate
        if self.use_diag_residual:
            self.to_d = nn.Linear(self.router_hidden, r, bias=False)

        self.scaling = self.lora_alpha / self.r if self.r > 0 else 1.0

        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
        
        print("householder-like")


        self.debug_save_dir = "/home/hke3/LD-MoLE/vis/saved_var"
        self.debug_save_every = 1
        self.debug_save_dtype = torch.float16
        self.debug_batch_idx = 0
        self.layerindex = kwargs.get("layer_idx")
        self.module_name=kwargs.get("module_name")
        self.debug_save_fields = ["w", "beta", "base_h","lora_out_h"]
        
    def _save_debug_tensors(self, base_h, lora_out_h, w, beta, u_head, u_mod):
        if self.debug_save_dir is None:
            return

        from pathlib import Path
        save_dir = Path(self.debug_save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.debug_batch_idx % self.debug_save_every != 0:
            self.debug_batch_idx += 1
            return

        field_map = {
            "w": w,
            "beta": beta,
            "u_head": u_head,
            "u_mod": u_mod,
            "base_h": base_h,
            "lora_out_h":lora_out_h
        }

        obj = {}
        for k in self.debug_save_fields:
            obj[k] = field_map[k].detach().to("cpu", dtype=self.debug_save_dtype)

        save_path = save_dir / f"{self.layerindex}_{self.module_name}_batch_{self.debug_batch_idx:06d}.pt"
        torch.save(obj, save_path)
        self.debug_batch_idx += 1
    
    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

        if hasattr(self, "lora_A_shared"):
            nn.init.kaiming_uniform_(self.lora_A_shared.weight, a=math.sqrt(5))

        if hasattr(self, "lora_B_heads"):
            nn.init.zeros_(self.lora_B_heads)

        if hasattr(self, "head_embed"):
            nn.init.zeros_(self.head_embed)

        if hasattr(self, "to_w"):
            nn.init.kaiming_uniform_(self.to_w.weight, a=math.sqrt(5))

        if hasattr(self, "to_beta"):
            if self.beta_init_zero:
                nn.init.zeros_(self.to_beta.weight)
            else:
                nn.init.kaiming_uniform_(self.to_beta.weight, a=math.sqrt(5))

        if hasattr(self, "to_d"):
            nn.init.zeros_(self.to_d.weight)

    def _compute_consensus(self, head_out: torch.Tensor):
        """
        head_out: [B, S, H, Dh]
        returns:
            c_token: [B, S, 1]
            c_head:  [B, S, H]
        """
        center = head_out.mean(dim=2, keepdim=True)          # [B, S, 1, Dh]
        cos_sim = F.cosine_similarity(head_out, center, dim=-1)  # [B, S, H]
        c_head = 1.0 - cos_sim
        c_token = c_head.mean(dim=2, keepdim=True)           # [B, S, 1]
        return c_token, c_head

    def _householder_like(self, u_head: torch.Tensor, h: torch.Tensor):
        """
        u_head: [B, S, H, r]
        h:      [B, S, H, Hid]
        returns:
            u_mod: [B, S, H, r]
            beta:  [B, S, H, 1]
            w:     [B, S, H, r]
        """
        # direction
        w_raw = self.to_w(h)  # [B, S, H, r]
        w = F.normalize(w_raw, p=2, dim=-1, eps=self.w_eps)

        # scalar beta
        beta_raw = self.to_beta(h)  # [B, S, H, 1]

        # identity-friendly init:
        # if to_beta starts at zero => beta starts at 0 => u_mod ~= u_head
        beta = self.beta_scale * torch.tanh(beta_raw)

        # projection <u, w>
        proj = (u_head * w).sum(dim=-1, keepdim=True)  # [B, S, H, 1]

        # Householder-like update
        u_mod = u_head - beta * proj * w

        # optional residual diagonal gate
        if self.use_diag_residual:
            d_raw = self.to_d(h)
            d = 1.0 + self.diag_scale * torch.tanh(d_raw)
            u_mod = d * u_mod

        return u_mod, beta, w

    def forward(self, x: torch.Tensor):
        """
        x: [B, S, in_features]
        """
        B, S, _ = x.shape

        base = F.linear(
            x,
            transpose(self.weight, self.fan_in_fan_out),
            bias=self.bias,
        )

        if self.disable_adapters or self.r == 0:
            return LoraOutput(
                results=base,
                routing_weight=None,
                gate_score=None,
                lam=None,
            )

        x_drop = self.lora_dropout(x)                      # [B, S, Din]

        # shared latent
        u = self.lora_A_shared(x_drop)                    # [B, S, r]
        u_head = u.unsqueeze(2).expand(-1, -1, self.num_heads, -1)  # [B, S, H, r]

        # base output -> per-head consensus signal
        base_h = base.view(B, S, self.num_heads, self.head_dim)
        c_token, c_head = self._compute_consensus(base_h.detach())

        # token path
        if self.use_token_consensus:
            token_in = torch.cat([u, c_token], dim=-1)    # [B, S, r+1]
        else:
            token_in = u                                  # [B, S, r]

        g_token = self.token_router_in(token_in)          # [B, S, Hid]
        g_token = g_token.unsqueeze(2)                    # [B, S, 1, Hid]

        # head path
        g_head = self.head_router_in(c_head.unsqueeze(-1))   # [B, S, H, Hid]
        g_head = g_head + self.head_embed.view(1, 1, self.num_heads, self.router_hidden)

        # fused router state
        h = self.lora_route_post(g_token + g_head)        # [B, S, H, Hid]

        # Householder-like latent routing
        u_mod, beta, w = self._householder_like(u_head, h)  # [B, S, H, r]
        
        # head-wise projection
        lora_out_h = torch.einsum("hdr,bshr->bshd", self.lora_B_heads, u_mod)  # [B, S, H, Dh]
        lora_result = lora_out_h.reshape(B, S, self.out_features)
        self._save_debug_tensors(base_h, lora_out_h, w, beta, u_head, u_mod)
        result = base + lora_result * self.scaling

        return LoraOutput(
            results=result,
            routing_weight=None,
            gate_score=beta,   # you can also return w or proj if you want to analyze
            lam=None,
        )
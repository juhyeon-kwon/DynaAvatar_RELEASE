# -*- coding: utf-8 -*-
# @Organization  : Alibaba XR-Lab
# @Author        : Xiaodong Gu & Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-03-1 17:49:25
# @Function      : transformer_block

import pdb
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from accelerate.logging import get_logger
from diffusers.utils import is_torch_version
logger = get_logger(__name__)

# post module (submission version)
class TransformerDecoder(nn.Module):
    """
    Transformer blocks that process the input and optionally use condition and modulation.
    """

    motion_embed_type = ["sd3_mm_cond", "sd3_mm_bh_cond", "sd3_mm_dynamic_cond"]

    def __init__(
        self,
        block_type: str,
        num_layers: int,
        num_heads: int,
        inner_dim: int,
        cond_dim: int = None,
        mod_dim: int = None,
        gradient_checkpointing=False,
        eps: float = 1e-6,
        is_dynamic = False
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.block_type = block_type
        self.is_dynamic = is_dynamic

        # qw00n; sd3_mm_bh_cond
        if (
            block_type == "sd3_cond"
            or block_type == "sd3_mm_cond"
            or block_type == "sd3_mm_bh_cond" ##
            or block_type == "sd3_mm_dynamic_cond" ## qw00n; not used 
        ):
            # dual_attention_layers = list(range(num_layers//2))
            dual_attention_layers = []
            self.layers = nn.ModuleList(
                [
                    self._block_fn(inner_dim, cond_dim, mod_dim)(
                        num_heads=num_heads,
                        eps=eps,
                        context_pre_only=i == num_layers - 1,
                        use_dual_attention=(
                            True if i in dual_attention_layers else False
                        ),
                        is_dynamic=False # hard code to prevent alternating condition
                    )
                    for i in range(num_layers)
                ]
            )
            ## qw00n; post dynamic layers
            if self.is_dynamic:
                from LHM.models.transformer_dit import SD3DynamicMMJointTransformerBlock
                num__dynamic_layers = 5
                self.dynamic_layers = nn.ModuleList(
                    [
                        partial(
                            SD3DynamicMMJointTransformerBlock, dim=inner_dim, qk_norm="rms_norm"
                        )(
                            num_heads=num_heads,
                            eps=eps,
                            context_pre_only=i == num__dynamic_layers - 1,
                            use_dual_attention=(
                                True if i in dual_attention_layers else False
                            ),
                        )
                        for i in range(num__dynamic_layers) # TODO
                    ]
                )
        else:
            self.layers = nn.ModuleList(
                [
                    self._block_fn(inner_dim, cond_dim, mod_dim)(
                        num_heads=num_heads,
                        eps=eps,
                    )
                    for _ in range(num_layers)
                ]
            )

        self.norm = nn.LayerNorm(inner_dim, eps=eps)

        if self.block_type in [
            "cogvideo_cond",
            "sd3_cond",
            "sd3_mm_cond",
            "sd3_mm_bh_cond", ##
        ]:
            self.linear_cond_proj = nn.Linear(cond_dim, inner_dim)
            
        if self.block_type in [
            "sd3_mm_dynamic_cond" ## qw00n;
        ]:
            self.linear_cond_proj_dynamic = nn.Linear(cond_dim, inner_dim)

    @property
    def block_type(self):
        return self._block_type

    @block_type.setter
    def block_type(self, block_type):
        assert block_type in [
            "basic",
            "cond",
            "mod",
            "cond_mod",
            "sd3_cond",
            "sd3_mm_cond",
            "sd3_mm_bh_cond", ##
            "sd3_mm_dynamic_cond",
            "cogvideo_cond",
        ], f"Unsupported block type: {block_type}"
        self._block_type = block_type

    def _block_fn(self, inner_dim, cond_dim, mod_dim):
        assert inner_dim is not None, f"inner_dim must always be specified"
        if self.block_type == "basic":
            assert (
                cond_dim is None and mod_dim is None
            ), f"Condition and modulation are not supported for BasicBlock"
            from .block import BasicBlock

            logger.debug(f"Using BasicBlock")
            return partial(BasicBlock, inner_dim=inner_dim)
        elif self.block_type == "cond":
            assert (
                cond_dim is not None
            ), f"Condition dimension must be specified for ConditionBlock"
            assert (
                mod_dim is None
            ), f"Modulation dimension is not supported for ConditionBlock"
            from .block import ConditionBlock

            logger.debug(f"Using ConditionBlock")
            return partial(ConditionBlock, inner_dim=inner_dim, cond_dim=cond_dim)
        elif self.block_type == "mod":
            logger.error(f"modulation without condition is not implemented")
            raise NotImplementedError(
                f"modulation without condition is not implemented"
            )
        elif self.block_type == "cond_mod":
            assert (
                cond_dim is not None and mod_dim is not None
            ), f"Condition and modulation dimensions must be specified for ConditionModulationBlock"
            from .block import ConditionModulationBlock

            logger.debug(f"Using ConditionModulationBlock")
            return partial(
                ConditionModulationBlock,
                inner_dim=inner_dim,
                cond_dim=cond_dim,
                mod_dim=mod_dim,
            )
        elif self.block_type == "cogvideo_cond":
            logger.debug(f"Using CogVideoXBlock")
            from LHM.models.transformer_dit import CogVideoXBlock

            # assert inner_dim == cond_dim, f"inner_dim:{inner_dim}, cond_dim:{cond_dim}"
            return partial(CogVideoXBlock, dim=inner_dim, attention_bias=True)
        elif self.block_type == "sd3_cond":
            logger.debug(f"Using SD3JointTransformerBlock")
            from LHM.models.transformer_dit import SD3JointTransformerBlock

            return partial(SD3JointTransformerBlock, dim=inner_dim, qk_norm="rms_norm")
        elif self.block_type == "sd3_mm_cond":
            logger.debug(f"Using SD3MMJointTransformerBlock")
            from LHM.models.transformer_dit import SD3MMJointTransformerBlock

            return partial(
                SD3MMJointTransformerBlock, dim=inner_dim, qk_norm="rms_norm"
            )
        elif self.block_type == "sd3_mm_bh_cond": ##
            logger.debug(f"Using SD3MMJointTransformerBlock")
            from LHM.models.transformer_dit import SD3BodyHeadMMJointTransformerBlock

            return partial(
                SD3BodyHeadMMJointTransformerBlock, dim=inner_dim, qk_norm="rms_norm"
            )
        elif self.block_type == "sd3_mm_dynamic_cond": # qw00n;
            logger.debug(f"attach Dynamic TransformerBlock")
            from LHM.models.transformer_dit import SD3DynamicMMJointTransformerBlock

            return partial(
                SD3DynamicMMJointTransformerBlock, dim=inner_dim, qk_norm="rms_norm"
            )
        else:
            raise ValueError(
                f"Unsupported block type during runtime: {self.block_type}"
            )

    def assert_runtime_integrity(
        self, x: torch.Tensor, cond: torch.Tensor, mod: torch.Tensor
    ):
        assert x is not None, f"Input tensor must be specified"
        if self.block_type == "basic":
            assert (
                cond is None and mod is None
            ), f"Condition and modulation are not supported for BasicBlock"
        elif "cond" in self.block_type:
            assert (
                cond is not None and mod is None
            ), f"Condition must be specified and modulation is not supported for ConditionBlock"
        elif self.block_type == "mod":
            raise NotImplementedError(
                f"modulation without condition is not implemented"
            )
        else:
            assert (
                cond is not None and mod is not None
            ), f"Condition and modulation must be specified for ConditionModulationBlock"

    def forward_layer(
        self, layer: nn.Module, x: torch.Tensor, cond: torch.Tensor, mod: torch.Tensor
    ):
        if self.block_type == "basic":
            return layer(x)
        elif self.block_type == "cond":
            return layer(x, cond)
        elif self.block_type == "mod":
            return layer(x, mod)
        else:
            return layer(x, cond, mod)

    def forward(
        self,
        x: torch.Tensor, 
        img_cond: torch.Tensor = None,
        motion_cond: torch.Tensor = None,
        mod: torch.Tensor = None,
        temb: torch.Tensor = None,
        temb2: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass of the transformer model.
        Args:
            x (torch.Tensor): Input tensor of shape [N, L, D].
            cond (torch.Tensor, optional): Conditional tensor of shape [N, L_cond, D_cond] or None. Defaults to None.
            mod (torch.Tensor, optional): Modulation tensor of shape [N, D_mod] or None. Defaults to None.
            temb (torch.Tensor, optional): Modulation tensor of shape [N, D_mod] or None. Defaults to None.  # For SD3_MM_Cond, temb means MotionCLIP
        Returns:
            torch.Tensor: Output tensor of shape [N, L, D].
        """
        # qw00n;
        # x: [B, N, 1024] => query points
        # cond: [B, num_merged_tokens, 1536] => image merged tokens
        # mod: None
        # temb: [B, 2048] => global context vector
        self.assert_runtime_integrity(x, img_cond, mod)

        if self.block_type in [
            "cogvideo_cond",
            "sd3_cond",
            "sd3_mm_cond",
            "sd3_mm_bh_cond",
            "sd3_mm_dynamic_cond"
        ]:
            img_cond = self.linear_cond_proj(img_cond)

            def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)
                        return custom_forward

            for layer in self.layers:
                # qw00n; training setup
                if self.training and self.gradient_checkpointing:

                    

                    ckpt_kwargs: Dict[str, Any] = (
                        {"use_reentrant": False}
                        if is_torch_version(">=", "1.11.0")
                        else {}
                    )

                    if self.block_type not in self.motion_embed_type:
                        x, cond = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(layer),
                            x,
                            cond,
                            **ckpt_kwargs,
                        )
                    else: ##
                        if self.is_dynamic:
                            x, img_cond = torch.utils.checkpoint.checkpoint(
                                create_custom_forward(layer),
                                x,
                                img_cond,
                                motion_cond,
                                temb,
                                temb2,
                                **ckpt_kwargs,
                            )
                        else:
                            x, img_cond = torch.utils.checkpoint.checkpoint(
                                create_custom_forward(layer),
                                x,
                                img_cond,
                                None,
                                temb,
                                None,
                                **ckpt_kwargs,
                            )
                else:
                    if self.is_dynamic:
                        x, img_cond = layer(
                            hidden_states=x,
                            encoder_hidden_states=img_cond,
                            encoder_hidden_states2=motion_cond,
                            temb=temb,
                            temb2=temb2
                            # image_rotary_emb=None,
                        )
                    else:
                        x, img_cond = layer(
                            hidden_states=x,
                            encoder_hidden_states=img_cond,
                            encoder_hidden_states2=motion_cond,
                            temb=temb,
                            temb2=temb2
                            # image_rotary_emb=None,
                        )

            x = self.norm(x)

            # post dynamic layers
            if self.is_dynamic:
                for dynamic_layer in self.dynamic_layers:
                    if self.training and self.gradient_checkpointing:
                        x, motion_cond = torch.utils.checkpoint.checkpoint(
                                        create_custom_forward(dynamic_layer),
                                        x,
                                        
                                        motion_cond,

                                        temb2,
                                        **ckpt_kwargs,
                                    )
                    else:
                        x, motion_cond = dynamic_layer(
                            hidden_states=x,
                            encoder_hidden_states2=motion_cond,
                            temb2=temb2
                            # image_rotary_emb=None,
                        )

                x = self.norm(x)

            else:
                pass


        else:
            for layer in self.layers:
                x = self.forward_layer(layer, x, cond, mod)
            x = self.norm(x)

        return x

    def forward_dynamic(
        self,
        x: torch.Tensor, 
        img_cond: torch.Tensor = None,
        motion_cond: torch.Tensor = None,
        mod: torch.Tensor = None,
        temb: torch.Tensor = None,
        temb2: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass of the transformer model.
        Args:
            x (torch.Tensor): Input tensor of shape [N, L, D].
            cond (torch.Tensor, optional): Conditional tensor of shape [N, L_cond, D_cond] or None. Defaults to None.
            mod (torch.Tensor, optional): Modulation tensor of shape [N, D_mod] or None. Defaults to None.
            temb (torch.Tensor, optional): Modulation tensor of shape [N, D_mod] or None. Defaults to None.  # For SD3_MM_Cond, temb means MotionCLIP
        Returns:
            torch.Tensor: Output tensor of shape [N, L, D].
        """
        # qw00n;
        # x: [B, N, 1024] => query points
        # img_cond: [B, num_merged_tokens, 1536] => image merged tokens
        # motion_cond: [B, T, 1024]
        # mod: None
        # temb: [B, 1024] => target pose token
        #self.assert_runtime_integrity(x, cond, mod)

        if self.block_type in [
            "cogvideo_cond",
            "sd3_cond",
            "sd3_mm_cond",
            "sd3_mm_bh_cond",
            "sd3_mm_dynamic_cond"
        ]:
            img_cond = self.linear_cond_proj_dynamic(img_cond) # qw00n; another projection layer
            for layer in self.layers:
                # qw00n; training setup
                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = (
                        {"use_reentrant": False}
                        if is_torch_version(">=", "1.11.0")
                        else {}
                    )

                    if self.block_type not in self.motion_embed_type:
                        x, cond = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(layer),
                            x,
                            cond,
                            **ckpt_kwargs,
                        )
                    else: ##
                        x, img_cond, motion_cond = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(layer),
                            x,
                            img_cond,
                            motion_cond,
                            temb,
                            temb2,
                            **ckpt_kwargs,
                        )
                else:
                    x, img_cond, motion_cond = layer(
                        hidden_states=x,
                        encoder_hidden_states=img_cond,
                        encoder_hidden_states2=motion_cond,
                        temb=temb,
                        temb2=temb2
                        # image_rotary_emb=None,
                    )

            x = self.norm(x)
        else:
            for layer in self.layers:
                #x = self.forward_layer(layer, x, cond, mod)
                x, img_cond, motion_cond = layer(
                        hidden_states=x,
                        encoder_hidden_states=img_cond,
                        encoder_hidden_states2=motion_cond,
                        temb=temb,
                        temb2=temb2
                        # image_rotary_emb=None,
                    )
            x = self.norm(x)

        return x


    """
    Transformer blocks that process the input and optionally use condition and modulation.
    """

    motion_embed_type = ["sd3_mm_cond", "sd3_mm_bh_cond", "sd3_mm_dynamic_cond"]

    def __init__(
        self,
        block_type: str,
        num_layers: int,
        num_heads: int,
        inner_dim: int,
        cond_dim: int = None,
        mod_dim: int = None,
        gradient_checkpointing=False,
        eps: float = 1e-6,
        is_dynamic = False
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.block_type = block_type
        self.is_dynamic = is_dynamic

        
        # qw00n; sd3_mm_bh_cond
        if (
            block_type == "sd3_cond"
            or block_type == "sd3_mm_cond"
            or block_type == "sd3_mm_bh_cond" ##
            or block_type == "sd3_mm_dynamic_cond" ## qw00n;
        ):
            # dual_attention_layers = list(range(num_layers//2))
            dual_attention_layers = []
            self.layers = nn.ModuleList(
                [
                    self._block_fn(inner_dim, cond_dim, mod_dim)(
                        num_heads=num_heads,
                        eps=eps,
                        context_pre_only=i == num_layers - 1,
                        use_dual_attention=(
                            True if i in dual_attention_layers else False
                        ),
                        is_dynamic=self.is_dynamic
                    )
                    for i in range(num_layers)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    self._block_fn(inner_dim, cond_dim, mod_dim)(
                        num_heads=num_heads,
                        eps=eps,
                    )
                    for _ in range(num_layers)
                ]
            )

        self.norm = nn.LayerNorm(inner_dim, eps=eps)

        if self.block_type in [
            "cogvideo_cond",
            "sd3_cond",
            "sd3_mm_cond",
            "sd3_mm_bh_cond", ##
        ]:
            self.linear_cond_proj = nn.Linear(cond_dim, inner_dim)
            
        if self.block_type in [
            "sd3_mm_dynamic_cond" ## qw00n;
        ]:
            self.linear_cond_proj_dynamic = nn.Linear(cond_dim, inner_dim)

    @property
    def block_type(self):
        return self._block_type

    @block_type.setter
    def block_type(self, block_type):
        assert block_type in [
            "basic",
            "cond",
            "mod",
            "cond_mod",
            "sd3_cond",
            "sd3_mm_cond",
            "sd3_mm_bh_cond", ##
            "sd3_mm_dynamic_cond",
            "cogvideo_cond",
        ], f"Unsupported block type: {block_type}"
        self._block_type = block_type

    def _block_fn(self, inner_dim, cond_dim, mod_dim):
        assert inner_dim is not None, f"inner_dim must always be specified"
        if self.block_type == "basic":
            assert (
                cond_dim is None and mod_dim is None
            ), f"Condition and modulation are not supported for BasicBlock"
            from .block import BasicBlock

            logger.debug(f"Using BasicBlock")
            return partial(BasicBlock, inner_dim=inner_dim)
        elif self.block_type == "cond":
            assert (
                cond_dim is not None
            ), f"Condition dimension must be specified for ConditionBlock"
            assert (
                mod_dim is None
            ), f"Modulation dimension is not supported for ConditionBlock"
            from .block import ConditionBlock

            logger.debug(f"Using ConditionBlock")
            return partial(ConditionBlock, inner_dim=inner_dim, cond_dim=cond_dim)
        elif self.block_type == "mod":
            logger.error(f"modulation without condition is not implemented")
            raise NotImplementedError(
                f"modulation without condition is not implemented"
            )
        elif self.block_type == "cond_mod":
            assert (
                cond_dim is not None and mod_dim is not None
            ), f"Condition and modulation dimensions must be specified for ConditionModulationBlock"
            from .block import ConditionModulationBlock

            logger.debug(f"Using ConditionModulationBlock")
            return partial(
                ConditionModulationBlock,
                inner_dim=inner_dim,
                cond_dim=cond_dim,
                mod_dim=mod_dim,
            )
        elif self.block_type == "cogvideo_cond":
            logger.debug(f"Using CogVideoXBlock")
            from LHM.models.transformer_dit import CogVideoXBlock

            # assert inner_dim == cond_dim, f"inner_dim:{inner_dim}, cond_dim:{cond_dim}"
            return partial(CogVideoXBlock, dim=inner_dim, attention_bias=True)
        elif self.block_type == "sd3_cond":
            logger.debug(f"Using SD3JointTransformerBlock")
            from LHM.models.transformer_dit import SD3JointTransformerBlock

            return partial(SD3JointTransformerBlock, dim=inner_dim, qk_norm="rms_norm")
        elif self.block_type == "sd3_mm_cond":
            logger.debug(f"Using SD3MMJointTransformerBlock")
            from LHM.models.transformer_dit import SD3MMJointTransformerBlock

            return partial(
                SD3MMJointTransformerBlock, dim=inner_dim, qk_norm="rms_norm"
            )
        elif self.block_type == "sd3_mm_bh_cond": ##
            logger.debug(f"Using SD3MMJointTransformerBlock")
            from LHM.models.transformer_dit import SD3BodyHeadMMJointTransformerBlock

            return partial(
                SD3BodyHeadMMJointTransformerBlock, dim=inner_dim, qk_norm="rms_norm"
            )
        elif self.block_type == "sd3_mm_dynamic_cond": # qw00n;
            logger.debug(f"attach Dynamic TransformerBlock")
            from LHM.models.transformer_dit import SD3DynamicMMJointTransformerBlock

            return partial(
                SD3DynamicMMJointTransformerBlock, dim=inner_dim, qk_norm="rms_norm"
            )
        else:
            raise ValueError(
                f"Unsupported block type during runtime: {self.block_type}"
            )

    def assert_runtime_integrity(
        self, x: torch.Tensor, cond: torch.Tensor, mod: torch.Tensor
    ):
        assert x is not None, f"Input tensor must be specified"
        if self.block_type == "basic":
            assert (
                cond is None and mod is None
            ), f"Condition and modulation are not supported for BasicBlock"
        elif "cond" in self.block_type:
            assert (
                cond is not None and mod is None
            ), f"Condition must be specified and modulation is not supported for ConditionBlock"
        elif self.block_type == "mod":
            raise NotImplementedError(
                f"modulation without condition is not implemented"
            )
        else:
            assert (
                cond is not None and mod is not None
            ), f"Condition and modulation must be specified for ConditionModulationBlock"

    def forward_layer(
        self, layer: nn.Module, x: torch.Tensor, cond: torch.Tensor, mod: torch.Tensor
    ):
        if self.block_type == "basic":
            return layer(x)
        elif self.block_type == "cond":
            return layer(x, cond)
        elif self.block_type == "mod":
            return layer(x, mod)
        else:
            return layer(x, cond, mod)

    def forward(
        self,
        x: torch.Tensor, 
        img_cond: torch.Tensor = None,
        motion_cond: torch.Tensor = None,
        mod: torch.Tensor = None,
        temb: torch.Tensor = None,
        temb2: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass of the transformer model.
        Args:
            x (torch.Tensor): Input tensor of shape [N, L, D].
            cond (torch.Tensor, optional): Conditional tensor of shape [N, L_cond, D_cond] or None. Defaults to None.
            mod (torch.Tensor, optional): Modulation tensor of shape [N, D_mod] or None. Defaults to None.
            temb (torch.Tensor, optional): Modulation tensor of shape [N, D_mod] or None. Defaults to None.  # For SD3_MM_Cond, temb means MotionCLIP
        Returns:
            torch.Tensor: Output tensor of shape [N, L, D].
        """
        # qw00n;
        # x: [B, N, 1024] => query points
        # cond: [B, num_merged_tokens, 1536] => image merged tokens
        # mod: None
        # temb: [B, 2048] => global context vector
        self.assert_runtime_integrity(x, img_cond, mod)

        if self.block_type in [
            "cogvideo_cond",
            "sd3_cond",
            "sd3_mm_cond",
            "sd3_mm_bh_cond",
            "sd3_mm_dynamic_cond"
        ]:
            img_cond = self.linear_cond_proj(img_cond)
            for layer in self.layers:
                # qw00n; training setup
                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = (
                        {"use_reentrant": False}
                        if is_torch_version(">=", "1.11.0")
                        else {}
                    )

                    if self.block_type not in self.motion_embed_type:
                        x, cond = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(layer),
                            x,
                            cond,
                            **ckpt_kwargs,
                        )
                    else: ##
                        if self.is_dynamic:
                            x, img_cond, motion_cond = torch.utils.checkpoint.checkpoint(
                                create_custom_forward(layer),
                                x,
                                img_cond,
                                motion_cond,
                                temb,
                                temb2,
                                **ckpt_kwargs,
                            )
                        else:
                            x, img_cond = torch.utils.checkpoint.checkpoint(
                                create_custom_forward(layer),
                                x,
                                img_cond,
                                None,
                                temb,
                                None,
                                **ckpt_kwargs,
                            )
                else:
                    if self.is_dynamic:
                        x, img_cond, motion_cond = layer(
                            hidden_states=x,
                            encoder_hidden_states=img_cond,
                            encoder_hidden_states2=motion_cond,
                            temb=temb,
                            temb2=temb2
                            # image_rotary_emb=None,
                        )
                    else:
                        x, img_cond = layer(
                            hidden_states=x,
                            encoder_hidden_states=img_cond,
                            temb=temb,
                            # image_rotary_emb=None,
                        )

            x = self.norm(x)
        else:
            for layer in self.layers:
                x = self.forward_layer(layer, x, cond, mod)
            x = self.norm(x)

        return x

    def forward_dynamic(
        self,
        x: torch.Tensor, 
        img_cond: torch.Tensor = None,
        motion_cond: torch.Tensor = None,
        mod: torch.Tensor = None,
        temb: torch.Tensor = None,
        temb2: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass of the transformer model.
        Args:
            x (torch.Tensor): Input tensor of shape [N, L, D].
            cond (torch.Tensor, optional): Conditional tensor of shape [N, L_cond, D_cond] or None. Defaults to None.
            mod (torch.Tensor, optional): Modulation tensor of shape [N, D_mod] or None. Defaults to None.
            temb (torch.Tensor, optional): Modulation tensor of shape [N, D_mod] or None. Defaults to None.  # For SD3_MM_Cond, temb means MotionCLIP
        Returns:
            torch.Tensor: Output tensor of shape [N, L, D].
        """
        # qw00n;
        # x: [B, N, 1024] => query points
        # img_cond: [B, num_merged_tokens, 1536] => image merged tokens
        # motion_cond: [B, T, 1024]
        # mod: None
        # temb: [B, 1024] => target pose token
        #self.assert_runtime_integrity(x, cond, mod)

        if self.block_type in [
            "cogvideo_cond",
            "sd3_cond",
            "sd3_mm_cond",
            "sd3_mm_bh_cond",
            "sd3_mm_dynamic_cond"
        ]:
            img_cond = self.linear_cond_proj_dynamic(img_cond) # qw00n; another projection layer
            for layer in self.layers:
                # qw00n; training setup
                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = (
                        {"use_reentrant": False}
                        if is_torch_version(">=", "1.11.0")
                        else {}
                    )

                    if self.block_type not in self.motion_embed_type:
                        x, cond = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(layer),
                            x,
                            cond,
                            **ckpt_kwargs,
                        )
                    else: ##
                        x, img_cond, motion_cond = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(layer),
                            x,
                            img_cond,
                            motion_cond,
                            temb,
                            temb2,
                            **ckpt_kwargs,
                        )
                else:
                    x, img_cond, motion_cond = layer(
                        hidden_states=x,
                        encoder_hidden_states=img_cond,
                        encoder_hidden_states2=motion_cond,
                        temb=temb,
                        temb2=temb2
                        # image_rotary_emb=None,
                    )

            x = self.norm(x)
        else:
            for layer in self.layers:
                #x = self.forward_layer(layer, x, cond, mod)
                x, img_cond, motion_cond = layer(
                        hidden_states=x,
                        encoder_hidden_states=img_cond,
                        encoder_hidden_states2=motion_cond,
                        temb=temb,
                        temb2=temb2
                        # image_rotary_emb=None,
                    )
            x = self.norm(x)

        return x
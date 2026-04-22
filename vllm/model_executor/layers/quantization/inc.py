# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fractions import Fraction
from typing import TYPE_CHECKING, Any

import regex as re
import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    MPLinearKernel,
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig,
    QuantizationMethods,
)
from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Method
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    PackedvLLMParameter,
    RowvLLMParameter,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)


class INCConfig(QuantizationConfig):
    """Config class for Intel Neural Compressor (INC).
    Repo: https://github.com/intel/neural-compressor
    """

    SUPPORTED_BITS = {2, 3, 4, 8}
    SUPPORTED_DTYPES = {"int"}
    SUPPORTED_FORMATS = {"auto_round:auto_gptq", "auto_round:auto_awq"}
    SUPPORTED_BACKENDS = {
        "auto",
        "gptq",
        "gptq:marlin",
        "awq",
        "awq:marlin",
        "marlin",
    }

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        sym: bool = True,
        packing_format: str = "auto_round:auto_gptq",
        block_name_to_quantize: str | list[str] | None = None,
        extra_config: dict[str, Any] | None = None,
        data_type: str = "int",
        backend: str = "auto",
    ) -> None:
        super().__init__()
        if weight_bits not in self.SUPPORTED_BITS:
            raise ValueError(
                f"Unsupported weight_bits: {weight_bits}, "
                f"currently only support {self.SUPPORTED_BITS}."
            )
        if data_type not in self.SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported data_type: {data_type},"
                f" currently only support  {self.SUPPORTED_DTYPES}."
            )
        if packing_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported packing_format: {packing_format}, "
                f"currently only support {self.SUPPORTED_FORMATS}."
            )
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend: {backend},  "
                f"currently only support {self.SUPPORTED_BACKENDS}."
            )

        self.weight_bits = weight_bits
        self.group_size = group_size
        self.sym = sym
        self.packing_format = packing_format
        self.block_name_to_quantize = (
            block_name_to_quantize.split(",")
            if isinstance(block_name_to_quantize, str)
            else block_name_to_quantize
        )
        self.extra_config = extra_config
        self.data_type = data_type
        self.backend = backend
        self.pack_factor = Fraction(32, weight_bits)

    def __repr__(self) -> str:
        return (
            f"INCConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, sym={self.sym})"
        )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "inc"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "INCConfig":
        return cls(
            weight_bits=cls.get_from_keys(config, ["bits"]),
            group_size=cls.get_from_keys(config, ["group_size"]),
            sym=cls.get_from_keys(config, ["sym"]),
            packing_format=cls.get_from_keys_or(
                config, ["packing_format"], "auto_round:auto_gptq"
            ),
            block_name_to_quantize=cls.get_from_keys_or(
                config, ["block_name_to_quantize", "to_quant_block_names"], None
            ),
            extra_config=cls.get_from_keys_or(config, ["extra_config"], None),
            data_type=cls.get_from_keys_or(config, ["data_type"], "int"),
            backend=cls.get_from_keys_or(config, ["backend", "vllm_backend"], "auto"),
        )

    def get_layer_config(self, layer, layer_name: str):
        def get_config(name: str, quantized: bool = True):
            if not self.extra_config:
                return (
                    self.weight_bits if quantized else 16,
                    self.group_size if quantized else -1,
                    self.sym if quantized else True,
                )

            # exact match first
            if name in self.extra_config:
                cfg = self.extra_config[name]
                return (
                    cfg.get("bits", self.weight_bits if quantized else 16),
                    cfg.get("group_size", self.group_size if quantized else -1),
                    cfg.get("sym", self.sym if quantized else True),
                )

            REGEX_SPECIAL_CHARS = set(r"*+?^$()[]{}|\\")
            for pattern, cfg in self.extra_config.items():
                if not isinstance(pattern, str) or not any(
                    c in REGEX_SPECIAL_CHARS for c in pattern
                ):
                    continue

                try:
                    if re.search(re.compile(pattern), name) is not None:
                        return (
                            cfg.get("bits", self.weight_bits if quantized else 16),
                            cfg.get("group_size", self.group_size if quantized else -1),
                            cfg.get("sym", self.sym if quantized else True),
                        )
                except re.error:
                    # Invalid regex, ignore.
                    continue

            return (
                self.weight_bits if quantized else 16,
                self.group_size if quantized else -1,
                self.sym if quantized else True,
            )

        # 1. Exact match from config
        if self.extra_config and layer_name in self.extra_config:
            return get_config(layer_name)

        # 2. Determine whether layer should be quantized
        quantized = not isinstance(layer, ParallelLMHead)
        if self.block_name_to_quantize:
            quantized = any(
                layer_name.startswith(name) for name in self.block_name_to_quantize
            )

        # 3. Handle fused MoE
        if self.extra_config and "fusedmoe" in layer.__class__.__name__.lower():
            moe_configs = [
                get_config(name, quantized)
                for name in self.extra_config
                if name.startswith(layer_name)
            ]
            if moe_configs:
                if len(set(moe_configs)) == 1:
                    return moe_configs[0]
                raise ValueError(
                    f"Fused MoE layer '{layer_name}' requires "
                    f"consistent quant config for all sub-layers"
                )

        # 4. Handle fused QKV or other patterns
        if self.extra_config:
            for fusion_key, sub_keys in self.packed_modules_mapping.items():
                if fusion_key in layer_name and layer_name.count(fusion_key) == 1:
                    sub_names = [
                        layer_name.replace(fusion_key, sub_key) for sub_key in sub_keys
                    ]
                    sub_configs = [get_config(name, quantized) for name in sub_names]
                    if len(set(sub_configs)) == 1:
                        return sub_configs[0]
                    raise ValueError(
                        f"Fused module '{layer_name}' requires "
                        f"consistent quant config for {sub_names}"
                    )

        # 5. Fallback or try a regular expression match
        return get_config(layer_name, quantized)

    def check_quantized(self, weight_bits: int) -> bool:
        return weight_bits < 16

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.block_name_to_quantize is not None:
            self.block_name_to_quantize = hf_to_vllm_mapper.apply_list(
                self.block_name_to_quantize
            )
        if self.extra_config is not None:
            self.extra_config = hf_to_vllm_mapper.apply_dict(self.extra_config)

    def apply_awq_quant_layer(self, layer, prefix: str, backend: str = "auto"):
        from vllm.model_executor.layers.fused_moe import FusedMoE
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            check_marlin_supported,
            check_moe_marlin_supports_layer,
        )

        weight_bits, group_size, sym = self.get_layer_config(layer, prefix)
        if not self.check_quantized(weight_bits):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return UnquantizedLinearMethod()
            else:
                return None

        logger.debug(
            "[%s] Type: %s, Bits: %s, Group Size: %s, Sym: %s",
            prefix,
            layer.__class__.__name__,
            weight_bits,
            group_size,
            sym,
        )
        if backend == "auto" or "marlin" in backend:
            AWQ_TYPE_MAP = {
                4: scalar_types.uint4,
                8: scalar_types.uint8,
            }
            use_marlin = (weight_bits in AWQ_TYPE_MAP) and check_marlin_supported(
                AWQ_TYPE_MAP[weight_bits], group_size, not sym
            )

            if isinstance(layer, FusedMoE):
                use_marlin = use_marlin and check_moe_marlin_supports_layer(
                    layer, group_size
                )

        else:
            use_marlin = False
        if use_marlin:
            from vllm.model_executor.layers.quantization.awq_marlin import (
                AWQMarlinConfig,
                AWQMarlinLinearMethod,
                AWQMarlinMoEMethod,
            )

            quant_args_marlin = AWQMarlinConfig(
                weight_bits=weight_bits,
                group_size=group_size,
                zero_point=not sym,
                lm_head_quantized=False,
                full_config={},
                modules_to_not_convert=[],
            )
        else:
            from vllm.model_executor.layers.quantization.awq import (
                AWQConfig,
                AWQLinearMethod,
            )

            quant_args = AWQConfig(
                weight_bits=weight_bits,
                group_size=group_size,
                zero_point=not sym,
            )

        if isinstance(layer, FusedMoE):
            if use_marlin:
                return AWQMarlinMoEMethod(quant_args_marlin, layer.moe_config)
            from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Config

            config = {
                "quant_method": "awq",
                "bits": weight_bits,
                "group_size": group_size,
                "zero_point": not sym,
                "lm_head": False,
            }
            return MoeWNA16Config.from_config(config).get_quant_method(layer, prefix)

        if isinstance(layer, (LinearBase, ParallelLMHead)):
            if use_marlin:
                return AWQMarlinLinearMethod(quant_args_marlin)
            else:
                return AWQLinearMethod(quant_args)
        return None

    def apply_gptq_quant_layer(self, layer, prefix: str, backend: str = "auto"):
        from vllm.model_executor.layers.fused_moe import FusedMoE
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            check_marlin_supported,
            check_moe_marlin_supports_layer,
        )

        weight_bits, group_size, sym = self.get_layer_config(layer, prefix)
        if not self.check_quantized(weight_bits):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return UnquantizedLinearMethod()
            else:
                return None

        logger.debug(
            "[%s] Type: %s, Bits: %s, Group Size: %s, Sym: %s",
            prefix,
            layer.__class__.__name__,
            weight_bits,
            group_size,
            sym,
        )
        if backend == "auto" or "marlin" in backend:
            GPTQ_TYPE_MAP = {
                (4, True): scalar_types.uint4b8,
                (8, True): scalar_types.uint8b128,
            }
            use_marlin = (weight_bits, sym) in GPTQ_TYPE_MAP and check_marlin_supported(
                GPTQ_TYPE_MAP[(weight_bits, sym)], group_size, has_zp=not sym
            )
            if isinstance(layer, FusedMoE):
                use_marlin = use_marlin and check_moe_marlin_supports_layer(
                    layer, group_size
                )
        else:
            use_marlin = False
        if use_marlin:
            from vllm.model_executor.layers.quantization.gptq_marlin import (
                GPTQMarlinConfig,
                GPTQMarlinLinearMethod,
                GPTQMarlinMoEMethod,
            )

            quant_args_marlin = GPTQMarlinConfig(
                weight_bits=weight_bits,
                group_size=group_size,
                is_sym=sym,
                lm_head_quantized=False,
                desc_act=False,
                dynamic={},
                full_config={},
            )
        else:
            from vllm.model_executor.layers.quantization.gptq import (
                GPTQConfig,
                GPTQLinearMethod,
            )

            quant_args = GPTQConfig(
                weight_bits=weight_bits,
                group_size=group_size,
                lm_head_quantized=False,
                desc_act=False,
                dynamic={},
            )

        if isinstance(layer, FusedMoE):
            if use_marlin:
                return GPTQMarlinMoEMethod(quant_args_marlin, layer.moe_config)
            else:
                from vllm.model_executor.layers.quantization.moe_wna16 import (
                    MoeWNA16Config,
                )

                config = {
                    "quant_method": "gptq",
                    "bits": weight_bits,
                    "group_size": group_size,
                    "sym": sym,
                    "lm_head": False,
                }
                return MoeWNA16Config.from_config(config).get_quant_method(
                    layer, prefix
                )

        if isinstance(layer, (LinearBase, ParallelLMHead)):
            if use_marlin:
                return GPTQMarlinLinearMethod(quant_args_marlin)
            else:
                return GPTQLinearMethod(quant_args)

        return None

    def apply_xpu_w4a16_quant_layer(self, layer, prefix: str):
        from vllm.model_executor.layers.fused_moe import FusedMoE

        weight_bits, group_size, sym = self.get_layer_config(layer, prefix)

        if not self.check_quantized(weight_bits):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return UnquantizedLinearMethod()
            elif isinstance(layer, FusedMoE):
                from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (  # noqa: E501
                    UnquantizedFusedMoEMethod,
                )

                return UnquantizedFusedMoEMethod(layer.moe_config)
            else:
                return None

        if weight_bits != 4:
            raise NotImplementedError(
                f"INC on XPU only supports 4-bit quantization, "
                f"got weight_bits={weight_bits}."
            )
        if not sym:
            raise NotImplementedError(
                "INC W4A16 on XPU only supports symmetric quantization for now."
            )

        if isinstance(layer, FusedMoE):
            from vllm.model_executor.layers.quantization.moe_wna16 import (
                MoeWNA16Config,
            )

            config = {
                "quant_method": "gptq",
                "bits": weight_bits,
                "group_size": group_size,
                "sym": sym,
                "lm_head": False,
            }
            moe_config = MoeWNA16Config.from_config(config)
            return INCXPUMoEMethod(moe_config, layer.moe_config)

        if isinstance(layer, (LinearBase, ParallelLMHead)):
            return INCXPULinearMethod(
                weight_bits=weight_bits,
                group_size=group_size,
                sym=sym,
            )
        return None

    def apply_cpu_w4a16_quant_layer(self, layer, prefix: str):
        weight_bits, group_size, sym = self.get_layer_config(layer, prefix)
        if not self.check_quantized(weight_bits):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return UnquantizedLinearMethod()
            else:
                return None

        if weight_bits != 4:
            raise NotImplementedError(
                f"INC on CPU only supports 4-bit quantization, "
                f"got weight_bits={weight_bits}."
            )
        if not sym:
            raise NotImplementedError(
                "INC W4A16 on CPU only supports symmetric quantization for now."
            )
        if isinstance(layer, (LinearBase, ParallelLMHead)):
            return self.apply_gptq_quant_layer(layer, prefix)
        return None

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if prefix and self.extra_config:
            for layer_name in self.extra_config:
                if (
                    layer_name == prefix or layer_name == f"model.{prefix}"
                ) and self.extra_config[layer_name].get("bits", 16) >= 16:
                    return UnquantizedLinearMethod()
        if current_platform.is_xpu():
            return self.apply_xpu_w4a16_quant_layer(layer, prefix)
        is_gptq = "gptq" in self.packing_format or "gptq" in self.backend
        if current_platform.is_cpu() and is_gptq:
            return self.apply_cpu_w4a16_quant_layer(layer, prefix)
        if is_gptq:
            return self.apply_gptq_quant_layer(layer, prefix)
        if "awq" in self.packing_format or "awq" in self.backend:
            return self.apply_awq_quant_layer(layer, prefix)

        raise NotImplementedError(
            f"Unsupported quantization configuration for layer '{prefix}'. "
            f"Platform: CPU={current_platform.is_cpu()}. "
            f"Platform: XPU={current_platform.is_xpu()}. "
            f"Format: {self.packing_format}, Backend: {self.backend}."
        )

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> "QuantizationMethods | None":
        """Override the `auto-round` method to `inc`."""
        is_auto_round_format = hf_quant_cfg.get("quant_method", None) == "auto-round"
        if is_auto_round_format:
            return cls.get_name()
        return None


class INCXPULinearMethod(LinearMethodBase):
    """XPU linear method for INC W4A16 quantization (symmetric only).

    Loads GPTQ-format checkpoint weights and delegates compute to the
    XPUwNa16LinearKernel via the MPLinearKernel framework.  The GPTQ
    parameter names and [K_packed, N] layout are remapped to the
    kernel-expected names (weight_packed, weight_scale, …) and
    [N, K_packed] layout during process_weights_after_loading.
    """

    _kernel_backends_being_used: set[str] = set()

    def __init__(self, weight_bits: int, group_size: int, sym: bool):
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.sym = sym
        self.pack_factor = 32 // weight_bits
        self.kernel: MPLinearKernel | None = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        scales_and_zp_size = input_size_per_partition // self.group_size

        # --- Register GPTQ checkpoint parameters ---
        # GPTQ: qweight [in // pack_factor, out] packed along input dim
        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition // self.pack_factor,
                output_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=0,
            packed_factor=self.pack_factor,
            weight_loader=weight_loader,
        )
        # scales: [num_groups, out] params_dtype
        scales = GroupQuantScaleParameter(
            data=torch.empty(
                scales_and_zp_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_loader,
        )
        # qzeros: [num_groups, out // pack_factor] int32
        qzeros = PackedvLLMParameter(
            data=torch.empty(
                scales_and_zp_size,
                output_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.pack_factor,
            weight_loader=weight_loader,
        )
        # GPTQ checkpoints may include g_idx for activation reordering.
        g_idx = RowvLLMParameter(
            data=torch.tensor(
                [i // self.group_size for i in range(input_size_per_partition)],
                dtype=torch.int32,
            ),
            input_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("scales", scales)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("g_idx", g_idx)

        # --- Select and instantiate kernel ---
        quant_type = scalar_types.uint4b8
        mp_config = MPLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=(
                input_size_per_partition,
                output_size_per_partition,
            ),
            weight_type=quant_type,
            act_type=params_dtype,
            group_size=self.group_size,
            zero_points=not self.sym,
            has_g_idx=False,
        )

        kernel_type = choose_mp_linear_kernel(mp_config)
        if kernel_type.__name__ not in self._kernel_backends_being_used:
            logger.info("Using %s for INC XPU W4A16", kernel_type.__name__)
            self._kernel_backends_being_used.add(kernel_type.__name__)

        # Use the hardcoded names that XPUwNa16LinearKernel expects
        self.kernel = kernel_type(
            mp_config,
            w_q_param_name="weight_packed",
            w_s_param_name="weight_scale",
            w_zp_param_name="weight_zero_point",
            w_gidx_param_name="g_idx",
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Remap GPTQ params to kernel-expected names/layout and delegate."""
        # qweight [K_packed, N] → weight_packed [N, K_packed]
        layer.weight_packed = Parameter(
            layer.qweight.data.t().contiguous(), requires_grad=False
        )
        # scales [num_groups, N] → weight_scale [N, num_groups]
        # (kernel transposes back to [num_groups, N] internally)
        layer.weight_scale = Parameter(
            layer.scales.data.t().contiguous(), requires_grad=False
        )

        # Remove GPTQ params that have been remapped
        layer.register_parameter("qweight", None)
        layer.register_parameter("scales", None)
        layer.register_parameter("qzeros", None)

        # Delegate to kernel (creates scalar zp, clears g_idx)
        self.kernel.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out_shape = x.shape[:-1] + (layer.weight_packed.shape[0],)
        out = self.kernel.apply_weights(layer, x, bias)
        return out.reshape(out_shape)


# TopK values supported by vllm_xpu_kernels._moe_C.remap_hidden_states.
_XPU_SUPPORTED_MOE_TOPK = frozenset({1, 2, 4, 6, 8, 10})


class INCXPUMoEMethod(MoeWNA16Method):
    """INC W4A16 MoE on XPU using native XPU fused MoE kernel.

    Inherits GPTQ-format weight creation/loading from MoeWNA16Method.
    Overrides apply to use the native XPU fused MoE kernel for supported
    TopK values, falling back to the Triton path otherwise.
    """

    def select_gemm_impl(
        self,
        prepare_finalize,
        layer: torch.nn.Module,
    ):
        if self.moe.experts_per_token in _XPU_SUPPORTED_MOE_TOPK:
            from vllm.model_executor.layers.fused_moe import XPUExpertsWNA16

            assert self.moe_quant_config is not None
            return XPUExpertsWNA16(
                moe_config=self.moe, quant_config=self.moe_quant_config
            )

        # Unsupported TopK — fall back to Triton WNA16
        logger.info_once(
            "XPU native MoE kernel does not support TopK=%d. "
            "Falling back to Triton WNA16 MoE.",
            self.moe.experts_per_token,
        )
        assert self.moe_quant_config is not None
        from vllm.triton_utils import HAS_TRITON

        if HAS_TRITON:
            from vllm.model_executor.layers.fused_moe import TritonWNA16Experts

            layer.w13_weight = layer.w13_qweight
            layer.w2_weight = layer.w2_qweight
            return TritonWNA16Experts(
                moe_config=self.moe, quant_config=self.moe_quant_config
            )

        raise NotImplementedError(
            "TritonExperts requires Triton for INC XPU WNA16 MoE "
            f"with TopK={self.moe.experts_per_token}."
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        topk = topk_ids.size(1)
        if topk in _XPU_SUPPORTED_MOE_TOPK:
            return self._apply_xpu(layer, x, topk_weights, topk_ids)
        # Fallback to generic fused_experts (Triton)
        return super().apply(layer, x, topk_weights, topk_ids,
                             shared_experts_input)

    def _apply_xpu(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        from vllm_xpu_kernels.fused_moe_interface import xpu_fused_moe

        output = torch.empty_like(x)
        xpu_fused_moe(
            hidden_states=x,
            w13=layer.w13_qweight,
            w13_scales=layer.w13_scales,
            w13_bias=None,
            w2=layer.w2_qweight,
            w2_scales=layer.w2_scales,
            w2_bias=None,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            n_experts_per_token=topk_ids.size(1),
            activation=layer.activation.value,
            num_experts=layer.local_num_experts,
            ep_rank=layer.moe_parallel_config.ep_rank,
            ep_size=layer.moe_parallel_config.ep_size,
            expert_map=layer.expert_map,
            output=output,
            is_int4=True,
        )
        return output

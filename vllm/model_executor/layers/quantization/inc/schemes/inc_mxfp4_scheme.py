# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.platforms import current_platform

from ..inc_linear import INCLinearMethod
from .inc_scheme import INCScheme

if TYPE_CHECKING:
    import torch

    from ..config_parser import INCLayerConfig
    from ..inc import INCConfig

logger = init_logger(__name__)


class INCMxfp4Scheme(INCScheme):
    """MXFP4 (W4A4) scheme for AutoRound checkpoints.

    Linear is supported on XPU only, reusing the shared MXFP4 linear kernel.
    MoE is not implemented yet.
    """

    @staticmethod
    def can_handle(layer_config: "INCLayerConfig") -> bool:
        return layer_config.is_mxfp4

    def get_linear_method(
        self,
        config: "INCConfig",
        layer: "torch.nn.Module",
        prefix: str,
        layer_config: "INCLayerConfig",
    ):
        del config, layer, prefix
        if not current_platform.is_xpu():
            raise NotImplementedError(
                f"INC MXFP4: linear only supported on XPU, config {layer_config}"
            )
        from .inc_mxfp4_linear import INCMxfp4LinearMethod

        return INCLinearMethod(INCMxfp4LinearMethod(layer_config))

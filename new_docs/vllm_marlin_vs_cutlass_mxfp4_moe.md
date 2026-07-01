# MiniMax-M3 MXFP4 MoE：为什么走不到 Marlin、以及如何强制走 Marlin

> 场景：NVIDIA B200 (SM100/Blackwell) 上部署自量化 MiniMax-M3-MXFP4，想改测 **Marlin MXFP4 MoE** 内核，
> 设置 `VLLM_DISABLED_KERNELS=FlashInferMxFp4LinearKernel` 却没有任何效果，MoE 仍然走 cutlass。
> 本文说明根因、两套 MXFP4 内核体系的区别，以及在 B200 上强制走 Marlin 的唯一办法。

---

## 1. 结论速览（TL;DR）

1. **你禁错对象了。** `VLLM_DISABLED_KERNELS=FlashInferMxFp4LinearKernel` 只作用于 **普通 Linear 层** 的
   MXFP4 内核选择，**完全不影响 MoE**。MiniMax-M3 的 MXFP4 全部在 MoE 里，根本没用到这个 linear kernel。
2. **MoE 走 cutlass 还是 Marlin 是硬判定，没有任何环境变量开关。** 在 B200/SM100 上
   `CutlassExpertsMxfp4._supports_current_device()` 恒为 `True`，于是 `use_cutlass_mxfp4=True`，
   永远进不了 `MarlinExperts` 分支。
3. **Marlin 确实支持 MXFP4 MoE**（`MarlinExperts`，官方 fallback 路径，功能完整），只是 B200 因 cutlass
   可用而被优先选中，自然走不到 Marlin。
4. 要在 B200 上测 Marlin，**唯一办法是改代码**把 `use_cutlass_mxfp4` 强制成 `False`（cutlass 不可用时
   代码本来就会自动落到 `MarlinExperts`）。

---

## 2. 两套完全独立的 MXFP4 内核体系

vLLM 里存在两套名字都带 "MxFp4" 的内核，很容易混淆：

| | 作用对象 | 选择入口 | 受 `VLLM_DISABLED_KERNELS` 控制？ |
|---|---|---|---|
| `FlashInferMxFp4LinearKernel` / `MarlinMxFp4LinearKernel` | **普通 Linear 层**（如 GPT-OSS 的 mxfp4 linear） | `choose_mxfp4_linear_kernel()`（`model_executor/kernels/linear/__init__.py:773`） | ✅ 是 |
| `CutlassExpertsMxfp4` / `MarlinExperts` | **MoE 专家 FFN**（MiniMax-M3 走这条） | `CompressedTensorsW4A4Mxfp4MoEMethod.__init__`（`compressed_tensors_moe_w4a4_mxfp4.py:50`） | ❌ 否，完全不读该变量 |

### 2.1 `VLLM_DISABLED_KERNELS` 到底控制什么

- 定义：`envs.py:1110`，把逗号分隔的字符串 split 成 list（元素是**内核类名 `__name__`**）。
- 唯一消费点：`model_executor/kernels/linear/__init__.py`。在选 Linear 层内核时逐个检查：

  ```python
  # __init__.py:445 / 691 / 742 / 787 / 911
  if kernel.__name__ in envs.VLLM_DISABLED_KERNELS:
      return False, f" {kernel.__name__} is disabled by environment variable"
  ```

- 也就是说它只对 `_POSSIBLE_*_KERNELS` 注册表里的 **Linear 内核**生效（int8 / fp8 / mxfp8 / **mxfp4 linear** / nvfp4 等）。
- **MoE 的 cutlass/marlin 选择完全不经过这个注册表**，所以设了对 MiniMax-M3 毫无影响 ——
  而且 MiniMax-M3 的 MoE 甚至没有用到 `FlashInferMxFp4LinearKernel` 这个 linear kernel。

---

## 3. MoE 侧真正的判定逻辑（硬编码，无 env 开关）

文件：`model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_mxfp4.py`

```python
class CompressedTensorsW4A4Mxfp4MoEMethod(CompressedTensorsMoEMethod):
    def __init__(self, moe):
        super().__init__(moe)
        self.group_size = 32
        self.mxfp4_backend = Mxfp4MoeBackend.MARLIN
        # use cutlass if supported, otherwise fallback to marlin for weight-only FP4
        self.use_cutlass_mxfp4 = CutlassExpertsMxfp4._supports_current_device()   # ← 关键，行50
        self.experts_cls: type[mk.FusedMoEExperts]
        if self.use_cutlass_mxfp4:                       # B200/SM100 → True，走这里
            logger.info_once("Using CutlassExpertsMxfp4 for MXFP4 MoE")
            self.experts_cls = CutlassExpertsMxfp4
        elif current_platform.is_xpu():
            self.mxfp4_backend = Mxfp4MoeBackend.XPU
            self.experts_cls = XPUExpertsMxFp4
        else:                                            # ← MarlinExperts 在这个分支
            logger.info_once("Using MarlinExperts for MXFP4 MoE")
            self.experts_cls = MarlinExperts
```

`_supports_current_device()`（`fused_moe/experts/cutlass_moe.py:1008`）：

```python
@staticmethod
def _supports_current_device() -> bool:
    p = current_platform
    capability = p.get_device_capability()
    return (
        p.is_cuda()
        and capability is not None
        and ops.mxfp4_experts_quant_supported(capability.to_int())   # SM100 → True
    )
```

在 B200/SM100（capability `(10,0)`）上 `ops.mxfp4_experts_quant_supported((10,0)) → True`，
所以 `use_cutlass_mxfp4` 恒为 `True`，**MoE 永远走 cutlass，进不了 Marlin 分支**。

> 已核对 `envs.py`：**没有任何 mxfp4 MoE backend 相关的环境变量**（既没有关 cutlass 的，也没有强制 marlin 的）。
> 所以靠环境变量无法切换。

---

## 4. cutlass 与 Marlin 两条 MoE 路径的区别

| | `CutlassExpertsMxfp4`（W4A4） | `MarlinExperts`（W4A16 weight-only） |
|---|---|---|
| 权重精度 | MXFP4 | MXFP4 |
| **激活精度** | 动态量化为 **MXFP4**（低精度） | 保持 **FP16/BF16**（不量化激活） |
| quant config | `mxfp4_moe_quant_config`（行139） | `make_mxfp4_moe_quant_config`（行148） |
| 权重预处理 | cutlass 布局 | `prepare_moe_fp4_layer_for_marlin(layer)`（行205，重排成 Marlin 布局） |
| 激活函数 | cutlass 内核可做 `silu_and_mul_with_clamp` | Marlin 走通用 activation |
| 适用设备 | 仅 cutlass 支持的设备（SM100 等） | 通用 fallback（cutlass 不可用时） |

要点：Marlin 是 **weight-only** —— 只有权重是 FP4，激活保持高精度，精度通常更好、但吞吐低于 W4A4 cutlass。
两者都是官方支持的完整实现，只是量化语义不同。

---

## 5. 如何在 B200 上强制走 Marlin

因为没有现成环境变量，唯一办法是改代码把 `use_cutlass_mxfp4` 强制成 `False`。
推荐加一个**可回退的环境变量开关**，方便对比测试。

### 5.1 最小改动（读环境变量，不需改 envs.py）

在 `compressed_tensors_moe_w4a4_mxfp4.py` 的 `__init__` 里，把行50 改为：

```python
import os  # 文件顶部若无则补

...
        # use cutlass if supported, otherwise fallback to marlin for weight-only FP4
        self.use_cutlass_mxfp4 = CutlassExpertsMxfp4._supports_current_device()
        # 允许通过环境变量强制走 Marlin（W4A16 weight-only），用于对比测试
        if os.environ.get("VLLM_MXFP4_MOE_FORCE_MARLIN", "0") == "1":
            self.use_cutlass_mxfp4 = False
            logger.info_once(
                "VLLM_MXFP4_MOE_FORCE_MARLIN=1 → forcing MarlinExperts for MXFP4 MoE"
            )
```

启动时设置：

```bash
export VLLM_MXFP4_MOE_FORCE_MARLIN=1
```

置位后 `use_cutlass_mxfp4=False`，代码自动落到 `else` 分支使用 `MarlinExperts`。
不设或设 `0` 则保持原行为（cutlass）。

> 该改动需要同步到你实际运行的库 `/storage/lkk/xpu_vllm/vllm`（eval_vllm.sh 用它），
> 若要三库一致，再迁移到 `/storage/lkk/m3/commit/vllm` 和 `/storage/lkk/m3/vllm_m3`。

### 5.2 验证是否真的走了 Marlin

启动日志里应出现：

```
Using MarlinExperts for MXFP4 MoE
```

而不是 `Using CutlassExpertsMxfp4 for MXFP4 MoE`。

---

## 6. 注意事项

- **Marlin 是 weight-only（W4A16）**：它读的是权重 FP4 + FP16 激活。你的模型权重本来就是 MXFP4，
  Marlin 会在加载时 `prepare_moe_fp4_layer_for_marlin` 把权重重排，通常可直接用；无需重量化。
- **Marlin/cutlass 只是 MoE FFN 的权重内核，与 attention 完全无关**。切它不会影响此前报错 6/7
  （flashinfer allreduce weight_bias / trtllm-gen page-128 decode）——那些属于注意力/通信融合路径，
  仍需靠升级 flashinfer 解决（见 `vllm_allreduce_weight_bias_fix.md`）。
- 因此**切到 Marlin 不能绕过报错 7**；它只是让你能对比 MoE 两条内核路径的精度/性能。

---

## 7. 相关文件索引

- `model_executor/kernels/linear/__init__.py`：Linear 层内核选择 + `VLLM_DISABLED_KERNELS` 消费点
  （`:445/691/742/787/911`；mxfp4 linear 选择 `:773`）。
- `model_executor/kernels/linear/mxfp4/flashinfer.py:18`：`FlashInferMxFp4LinearKernel`（Linear 层，非 MoE）。
- `model_executor/kernels/linear/mxfp4/marlin.py:9`：`MarlinMxFp4LinearKernel`（Linear 层，非 MoE）。
- `.../compressed_tensors_moe/compressed_tensors_moe_w4a4_mxfp4.py:50`：MoE cutlass/marlin 硬判定（**改这里**）。
- `fused_moe/experts/cutlass_moe.py:1008`：`CutlassExpertsMxfp4._supports_current_device()`。
- `fused_moe/experts/marlin_moe.py`：`MarlinExperts`（MoE 的 Marlin 实现）。
- `envs.py:1110`：`VLLM_DISABLED_KERNELS` 定义（确认无 mxfp4 MoE 相关 env）。

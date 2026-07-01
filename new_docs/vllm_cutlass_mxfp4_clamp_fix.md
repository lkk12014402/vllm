# 报错4分析与修复：CUTLASS MXFP4 MoE 不支持钳位 SwiGLU-OAI 激活

> 对应日志：`/storage/lkk/m3/test_4.log`
> 结论：**是 vLLM 的支持缺陷**（不是量化产物问题，也不是 nvidia fork 特有的 bug）。
> 涉及的两份 vLLM 代码（运行时的 `/storage/lkk/xpu_vllm/vllm` 与 docker 拷贝
> `/storage/lkk/m3/vllm_m3`）在这一块**完全相同**，都有同样的缺口。

---

## 1. 报错现象

模型此时已经**完全加载成功**（视觉塔、注意力、所有权重加载都过了——这说明前 3 个
报错的修复都生效了）。崩溃发生在**第一次前向计算的 MoE 阶段**：

```
File ".../fused_moe/experts/cutlass_moe.py", line 1083, in apply
    run_cutlass_moe_mxfp4(
File ".../fused_moe/experts/cutlass_moe.py", line 913, in run_cutlass_moe_mxfp4
    apply_moe_activation(activation, c2, c1)
File ".../fused_moe/activation.py", line 153, in apply_moe_activation
    assert clamp_limit is not None, "SWIGLUOAI_UNINTERLEAVE requires clamp_limit"
AssertionError: SWIGLUOAI_UNINTERLEAVE requires clamp_limit
```

一句话：MoE 专家用的是 **CUTLASS MXFP4** 内核，模型要求的激活是
**钳位版 SwiGLU-OAI（`swigluoai_uninterleave`）**，但 CUTLASS 这条路径在调用激活函数
时**没有把 clamp 参数传进去**，于是断言失败。

---

## 2. 为什么 MiniMax-M3 需要 `clamp_limit`

MiniMax-M3 的 MoE 用的是一种**带钳位的 SwiGLU-OAI 激活**（和 GPT-OSS 同类），模型
构造时会传三个参数：

`vllm/models/minimax_m3/nvidia/model.py`（`MiniMaxM3MoE`，约 256 行）：

```python
FusedMoE(
    ...
    activation="swigluoai_uninterleave",
    swiglu_limit=config.swiglu_limit,   # 钳位上限 clamp_limit
    swiglu_alpha=config.swiglu_alpha,   # SwiGLU 的 alpha
    swiglu_beta=config.swiglu_beta,     # SwiGLU 的 beta
)
```

这三个参数是**激活函数的一部分**，缺一不可。激活内核 `apply_moe_activation` 里，
`SWIGLUOAI_UNINTERLEAVE` 会走 `silu_and_mul_with_clamp(output, input, clamp_limit, alpha, beta)`：

`vllm/model_executor/layers/fused_moe/activation.py`（150-154 行）：

```python
elif activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE:
    # SwiGLU-OAI on packed w13 (gate = first half, up = second half).
    assert clamp_limit is not None, "SWIGLUOAI_UNINTERLEAVE requires clamp_limit"
    torch.ops._C.silu_and_mul_with_clamp(output, input, clamp_limit, alpha, beta)
```

所以只要 `clamp_limit` 是 `None`，这个激活就一定失败——这是正确的防御性断言，
**问题不在这里，而在于上游没把参数喂进来**。

---

## 3. 参数是怎么“丢”的：三层断链

模型明明传了 `swiglu_limit`，为什么到激活函数就变 `None` 了？因为
**compressed-tensors 的 MXFP4 MoE 这条链路上有三处没有把 clamp 参数接起来**。

### 断点 ①：后端选择器只看设备，不看激活是否支持

`.../compressed_tensors_moe/compressed_tensors_moe_w4a4_mxfp4.py`（原第 50 行）：

```python
# 只判断“当前 GPU 是否支持 cutlass mxfp4”，从不判断“cutlass 是否支持这个激活”
self.use_cutlass_mxfp4 = CutlassExpertsMxfp4._supports_current_device()
```

而 `CutlassExpertsMxfp4` 自己其实声明了一张“支持的激活白名单”
（`cutlass_moe.py`，`_supports_activation`），**原本并不包含 `SWIGLUOAI_UNINTERLEAVE`**：

```python
@staticmethod
def _supports_activation(activation: MoEActivation) -> bool:
    return activation in [
        MoEActivation.SILU, MoEActivation.GELU,
        MoEActivation.SWIGLUOAI,          # 注意：是“交错版”，走 swigluoai_and_mul，不需要 clamp
        MoEActivation.SWIGLUSTEP,
        MoEActivation.SILU_NO_MUL, MoEActivation.GELU_NO_MUL,
        MoEActivation.RELU2_NO_MUL,
        # ← 没有 SWIGLUOAI_UNINTERLEAVE
    ]
```

选择器**从来没调用过 `_supports_activation`**，于是即便激活不被支持，也硬选了
CUTLASS，没有回退到别的后端。这是缺陷的第一层。

### 断点 ②：MXFP4 的 quant config 根本装不下 clamp 参数

即便选了 CUTLASS，激活参数本应通过 `FusedMoEQuantConfig` 携带到内核里
（其它后端如 triton 就是这样：`self.quant_config.gemm1_clamp_limit`）。
`FusedMoEQuantConfig` 是有这三个字段的：

`config.py`（258-260 行）：

```python
gemm1_alpha: float | None = None
gemm1_beta: float | None = None
gemm1_clamp_limit: float | None = None
```

但 MXFP4 专用的构造函数 `mxfp4_moe_quant_config`（原第 840 行）**根本没有这几个入参**，
自然也不会填进去——对比隔壁 `nvfp4_w4a16_moe_quant_config` 就带了 `gemm1_clamp_limit`：

```python
def mxfp4_moe_quant_config(w1_scale, w2_scale):      # ← 原来只有两个权重 scale
    return FusedMoEQuantConfig.make("mxfp4", w1_scale=..., w2_scale=..., ...)
    # 没有 gemm1_clamp_limit / alpha / beta
```

而且调用它的地方（`get_fused_moe_quant_config`）也没打算传：

```python
if self.use_cutlass_mxfp4:
    return mxfp4_moe_quant_config(
        w1_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
    )   # ← layer.swiglu_limit 就在手边，却没有被使用
```

这是缺陷的第二层：**clamp 参数在 config 层就被丢弃了**。

### 断点 ③：CUTLASS 前向调用激活时写死不传 clamp

最后即便前两层补上了，CUTLASS 前向里对激活的调用**本身也是硬编码不传 clamp 的**。

`cutlass_moe.py`，`run_cutlass_moe_mxfp4`（原第 908-916 行）：

```python
if activation == MoEActivation.SILU:
    # SiLU 走融合快路，不需要 clamp
    int_fp4, int_blockscale = ops.silu_and_mul_mxfp4_experts_quant(...)
else:
    apply_moe_activation(activation, c2, c1)   # ← 只有位置参数，clamp_limit 默认 None！
    int_fp4, int_blockscale = ops.mxfp4_experts_quant(c2, ...)
```

`SWIGLUOAI_UNINTERLEAVE` 走 `else` 分支，进 `apply_moe_activation`，但没有把
`clamp_limit/alpha/beta` 带进去 → 断言炸掉。这是缺陷的第三层，也是**堆栈最终报错的那一行**。

> 小结：模型正确地把 `swiglu_limit` 交给了 vLLM，但 compressed-tensors 的 MXFP4 MoE
> 后端在 **选择器 → quant config → 内核激活调用** 三个环节都没把它接住。

---

## 4. 为什么这是“vLLM 没支持好”，而不是量化/模型问题

- **激活是模型架构决定的**：MiniMax-M3 本来就用钳位 SwiGLU-OAI，不是量化引入的。
- **参数在运行时是齐全的**：`layer.swiglu_limit/alpha/beta` 一直存在，只是没被这条
  后端链路读取。
- **别的后端是对的**：triton (`triton_moe.py` 155-179)、marlin、gpt_oss_triton、
  trtllm_mxfp4 都正确地从 `quant_config.gemm1_clamp_limit` 读取并传给激活。
  唯独 **compressed-tensors 的 cutlass MXFP4** 这条路漏了。
- **两份 vLLM 代码都一样**：`/storage/lkk/xpu_vllm/vllm` 与 docker 拷贝
  `/storage/lkk/m3/vllm_m3` 的这两个文件逐行一致，都有这个缺口。

所以这是 vLLM「compressed-tensors MXFP4(W4A4) MoE + 钳位 SwiGLU-OAI」组合的**支持缺陷**。

---

## 5. 修复方案（已实施）

把断掉的三层接起来，让 clamp 参数从模型一路流到 CUTLASS 激活内核。**改的都是通用
的 fused_moe 代码，不是 minimax 模型代码**，对其它模型是向后兼容的（非钳位激活时
这些参数为 `None`，行为不变）。

### 改动 1：`fused_moe/config.py` — 让 `mxfp4_moe_quant_config` 能携带 clamp

```python
def mxfp4_moe_quant_config(
    w1_scale, w2_scale,
    gemm1_alpha: float | None = None,
    gemm1_beta: float | None = None,
    gemm1_clamp_limit: float | None = None,
) -> FusedMoEQuantConfig:
    return FusedMoEQuantConfig.make(
        "mxfp4",
        w1_scale=w1_scale, w2_scale=w2_scale,
        per_act_token_quant=False, per_out_ch_quant=False, block_shape=None,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
    )
```

### 改动 2：`compressed_tensors_moe_w4a4_mxfp4.py` — 把 layer 的 swiglu 参数注入 config

```python
if self.use_cutlass_mxfp4:
    return mxfp4_moe_quant_config(
        w1_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
        gemm1_alpha=getattr(layer, "swiglu_alpha", None),
        gemm1_beta=getattr(layer, "swiglu_beta", None),
        gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
    )
```

用 `getattr(..., None)` 是为了对没有这些属性的老模型保持兼容。

### 改动 3：`cutlass_moe.py` — `run_cutlass_moe_mxfp4` 接收并透传 clamp

签名新增三个参数：

```python
def run_cutlass_moe_mxfp4(
    ...,
    apply_router_weight_on_input: bool = False,
    gemm1_clamp_limit: float | None = None,
    gemm1_alpha: float = 1.0,
    gemm1_beta: float = 0.0,
) -> None:
```

激活调用处（原第 913 行）改为传参：

```python
else:
    apply_moe_activation(
        activation, c2, c1,
        clamp_limit=gemm1_clamp_limit,
        alpha=gemm1_alpha,
        beta=gemm1_beta,
    )
    int_fp4, int_blockscale = ops.mxfp4_experts_quant(c2, ...)
```

`CutlassExpertsMxfp4.apply`（原第 1083 行）从 `self.quant_config` 取出并下传：

```python
run_cutlass_moe_mxfp4(
    ...,
    gemm1_clamp_limit=self.quant_config.gemm1_clamp_limit,
    gemm1_alpha=(self.quant_config.gemm1_alpha
                if self.quant_config.gemm1_alpha is not None else 1.0),
    gemm1_beta=(self.quant_config.gemm1_beta
                if self.quant_config.gemm1_beta is not None else 0.0),
)
```

### 改动 4：`cutlass_moe.py` — `_supports_activation` 如实登记

修复后 CUTLASS 已能处理钳位 SwiGLU-OAI，把它加进白名单，保持声明与实现一致：

```python
MoEActivation.SWIGLUOAI,
MoEActivation.SWIGLUOAI_UNINTERLEAVE,   # ← 新增
MoEActivation.SWIGLUSTEP,
```

> 数据流走通后（SILU 之外的分支）：`c1`(shape `[m*topk, 2n]`) 经
> `silu_and_mul_with_clamp` 得到 `c2`(shape `[m*topk, n]`)，正好满足
> `apply_moe_activation` 对 gated 激活“输入是输出 2 倍”的断言，尺寸对得上。

---

## 6. 为什么不采用“回退到 Marlin”这种更省事的方案

一个直觉方案是：断点①里加激活检查，不支持就回退 Marlin。但**这解决不了问题**，因为
Marlin 那条路的 quant config 构造 `make_mxfp4_moe_quant_config` **同样没有传
`gemm1_clamp_limit`**（第二层断链在 Marlin 路径依旧存在），回退过去还是会在
`gemm1_clamp_limit is None` 上失败。而且回退 Marlin 是 weight-only（激活不量化），
会**牺牲你 MXFP4 W4A4 想要的性能**。所以正解是把 CUTLASS 这条路打通，而不是绕开。

---

## 7. AMD 版对照

AMD 版（`vllm/models/minimax_m3/amd/model.py`）走的 MoE 后端与 NVIDIA 的
compressed-tensors cutlass MXFP4 不同（AMD 无此 cutlass 内核，通常走 triton/其它路径，
而那些路径本来就正确读取 `gemm1_clamp_limit`），因此 AMD 版不触发这个断言。
换句话说，这个缺口是 **compressed-tensors 的 cutlass MXFP4 MoE 路径独有**的，与前两个
报错（nvidia fork 缺 `packed_modules_mapping`、mapper 键带结尾点）性质不同——那两个是
模型定义 fork 时漏抄，这个是**通用量化后端的功能缺口**。

---

## 8. 与前三个报错的关系（整体回顾）

| # | 报错 | 阶段 | 性质 | 处理 |
|---|------|------|------|------|
| 1 | `KeyError: vision...fc1.weight` | 加载 | nvidia fork mapper 键带结尾点 + 视觉层误量化 | 改 `model.py` mapper |
| 2 | `KeyError: vision...qkv_proj.weight` | 加载 | nvidia fork 缺 `packed_modules_mapping` | 给两个类补 `packed_modules_mapping` |
| 3 | `different quantization schemes [gate_proj, up_proj]` | 加载 | 量化命令 `--ignore_layers mlp.gate` 子串误伤 dense 层 gate_proj | **需重量化**，改成 `block_sparse_moe.gate` |
| 4 | `SWIGLUOAI_UNINTERLEAVE requires clamp_limit` | **前向** | vLLM cutlass MXFP4 后端漏传 clamp | 本文档四处改动 |

前 3 个是加载期问题（KeyError=命名/映射，可改代码/config；ValueError=融合层两半状态
矛盾，需重量化）。第 4 个是**前向计算期**的功能缺口，说明加载已全部走通。

---

## 9. 需要你确认的一点

报错 4 能跑到**前向阶段**，意味着报错 3（`gate_up_proj` 方案不一致，那是加载期的检查）
**已经不再触发**。这通常说明你已经按报错 3 文档的建议**用
`--ignore_layers block_sparse_moe.gate` 重新量化过模型**了。请确认当前
`/storage/lkk/m3/MiniMax-M3-MXFP4` 是否为重量化后的版本；如果是，报错 3 已闭环。

---

## 10. 改动文件清单（本次报错4）

- `vllm/model_executor/layers/fused_moe/config.py`
  —`mxfp4_moe_quant_config` 增加 `gemm1_alpha/beta/clamp_limit` 三个入参并转发。
- `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_mxfp4.py`
  —`get_fused_moe_quant_config` 从 `layer.swiglu_*` 注入 clamp 参数。
- `vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py`
  —`run_cutlass_moe_mxfp4` 增加并透传 clamp 参数；激活调用改为传参；
  `CutlassExpertsMxfp4.apply` 从 `self.quant_config` 取参下传；
  `CutlassExpertsMxfp4._supports_activation` 增加 `SWIGLUOAI_UNINTERLEAVE`。

> 全部改在通用 fused_moe / compressed-tensors 代码，未改 minimax 模型定义；
> 非钳位激活的模型这些参数为 `None`，行为与修改前一致。

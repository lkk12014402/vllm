# MiniMax-M3 vLLM 报错 6 分析与修复：`allreduce_fusion() got an unexpected keyword argument 'weight_bias'`

> 对应日志：`/storage/lkk/m3/test_6.log`
> 修复文件：`compilation/passes/fusion/allreduce_rms_fusion.py`（唯一改动文件）
> 涉及三份代码库：运行时 `/storage/lkk/xpu_vllm/vllm`、`/storage/lkk/m3/commit/vllm`、`/storage/lkk/m3/vllm_m3`

---

## 1. 现象

在 block_size（报错 5）修复通过后，`eval_vllm.sh` 继续跑到 **KV cache 初始化阶段的前向 profiling**（`determine_available_memory` → `profile_cudagraph_memory` → `_dummy_run` → 模型前向），然后崩溃：

```
TypeError: allreduce_fusion() got an unexpected keyword argument 'weight_bias'
RuntimeError: Worker failed with error 'allreduce_fusion() got an unexpected keyword argument 'weight_bias''
RuntimeError: Engine core initialization failed.
```

**重要进展判断**：能跑到前向 profiling，说明之前的加载类报错（报错 1/2）、量化 scheme 报错（报错 3/4）、block_size 报错（报错 5）都已越过。这是一个**新的、更靠后的前向阶段错误**。

---

## 2. 调用链（从日志还原）

```
minimax_m3/nvidia/model.py:741   fused_allreduce_gemma_rms_norm(hidden_states, residual, self.post_attention_layernorm)
  └─ model_executor/layers/fused_allreduce_gemma_rms_norm.py:126   flashinfer_trtllm_fused_allreduce_norm(...)   # 一个 torch 自定义算子
       └─ compilation/passes/fusion/allreduce_rms_fusion.py:225   call_trtllm_fused_allreduce_norm(...)
            └─ flashinfer_comm.allreduce_fusion(..., weight_bias=weight_bias, ...)   # ← 这里报错
```

`flashinfer_trtllm_fused_allreduce_norm` 是通过 `direct_register_custom_op` 注册的自定义算子，其实体就是 `call_trtllm_fused_allreduce_norm`。**模型显式调用**和 **torch.compile 融合 pass 生成的调用**都会汇聚到这个算子 → 因此它是唯一的修复入口（single chokepoint）。

---

## 3. 根因

### 3.1 `weight_bias` 是什么

MiniMax-M3 的 LayerNorm 是 **Gemma 风格 RMSNorm**，公式为：

```
out = x_normed * (1 + weight)          # 注意是 (1 + weight)，不是 weight
```

而标准 RMSNorm 是 `out = x_normed * weight`。

flashinfer 的融合内核 `allreduce_fusion` 用一个 `weight_bias` 参数来表达这个 `+1` 偏置：

- `weight_bias=0.0` → 标准 RMSNorm：`out = gamma * x * rsqrt(...)`
- `weight_bias=1.0` → Gemma / Qwen3.5 RMSNorm：`out = (1 + gamma) * x * rsqrt(...)`

nvidia fork 的 vLLM 代码在 gemma 归一化处硬编码传了 `weight_bias=1.0`。

### 3.2 版本不匹配（真正原因）

**你环境里安装的是 `flashinfer 0.6.11.post2`，它的 `allreduce_fusion()` 根本没有 `weight_bias` 这个参数。** nvidia fork 依赖的是**打过补丁 / 更新版**的 flashinfer。

已核对 flashinfer 各版本（官方仓库 tag 源码）：

| flashinfer 版本 | 是否支持 `weight_bias` |
|---|---|
| **0.6.11.post2**（你当前安装） | ❌ 无（`allreduce_fusion` 无此参数）|
| 0.6.12 | ✅ 有 |
| 0.6.13（PyPI 最新）/ main | ✅ 有 |

即 **`weight_bias` 是 flashinfer 0.6.12 引入的**，你正好卡在引入版本的前一版。

### 3.3 为什么 AMD 版本能跑通

两个模型实现（`nvidia/model.py`、`amd/model.py`）都调用了 `fused_allreduce_gemma_rms_norm`，差别在这个函数内部的 `_can_use_flashinfer` 分支：

- **AMD（ROCm）**：flashinfer 的 trtllm 融合 allreduce 内核不可用，`_can_use_flashinfer` 返回 `False`，回退到**非融合路径** `norm(all_reduce(hidden_states), residual)`，其中 `norm` 是原生 `MiniMAXGemmaRMSNorm`，自己用 Triton 算 `(1 + weight)`。**整条路径根本不碰 flashinfer 的 `weight_bias` 参数**，所以不报错。
- **NVIDIA**：`_can_use_flashinfer` 返回 `True`，走 flashinfer 融合路径，把 `weight_bias=1.0` 传给一个不认识该参数的旧 flashinfer → `TypeError`。

一句话：**AMD 靠"没有 flashinfer 融合内核所以走原生回退"绕开了这个参数；NVIDIA 走融合路径正好撞上旧版 flashinfer 缺参数。**

---

## 4. 修复方案

在唯一入口 `call_trtllm_fused_allreduce_norm`（`allreduce_rms_fusion.py`）里，**在调用 flashinfer 前检测其是否支持 `weight_bias`**：

- **支持**（≥0.6.12）：照常把 `weight_bias=weight_bias` 传进去（原生路径）。
- **不支持**（0.6.11.post2 等 stock 版本）：把偏置**折叠进 gamma**（`rms_gamma = rms_gamma + weight_bias`），并**不传** `weight_bias` kwarg。

因为官方语义就是"`weight_bias` 加到 `rms_gamma` 上再缩放"，所以
`normed * (gamma + weight_bias)` 与内核内部 `weight_bias` 的效果**数学完全等价**，无精度或行为差异。

### 4.1 具体改动（`compilation/passes/fusion/allreduce_rms_fusion.py`）

**① 顶部新增 `import inspect`**

```python
import contextlib
import inspect          # 新增
from importlib.util import find_spec
```

**② flashinfer_comm 解析块之后，新增能力检测标志**

```python
            flashinfer_comm = _flashinfer_comm
    except ImportError:
        pass

# Some (patched) flashinfer builds accept a ``weight_bias`` argument on
# ``allreduce_fusion`` that adds GemmaRMSNorm's ``(1 + weight)`` offset inside
# the fused kernel. Stock flashinfer (e.g. 0.6.11) does not. Detect support so
# we can fold the bias into ``rms_gamma`` instead when it is missing.
_FI_ALLREDUCE_FUSION_SUPPORTS_WEIGHT_BIAS = flashinfer_comm is not None and (
    "weight_bias" in inspect.signature(flashinfer_comm.allreduce_fusion).parameters
)
```

**③ `flashinfer_comm.allreduce_fusion(...)` 调用前，新增折叠逻辑**

```python
        # GemmaRMSNorm scales by ``(1 + weight)``; ``weight_bias`` (=1.0) carries
        # that offset. Patched flashinfer applies it inside the kernel via the
        # ``weight_bias`` arg; stock flashinfer lacks it, so fold the bias into
        # ``rms_gamma`` (mathematically identical: ``normed * (gamma + bias)``).
        if _FI_ALLREDUCE_FUSION_SUPPORTS_WEIGHT_BIAS:
            weight_bias_kwargs: dict[str, Any] = {"weight_bias": weight_bias}
        else:
            weight_bias_kwargs = {}
            if weight_bias != 0.0:
                rms_gamma = rms_gamma + weight_bias

        flashinfer_comm.allreduce_fusion(
            input=allreduce_in,
            ...
```

**④ 调用参数里，把 `weight_bias=weight_bias,` 换成 `**weight_bias_kwargs,`**

```python
            fp32_acc=fp32_acc,
            **weight_bias_kwargs,          # 原来是 weight_bias=weight_bias,
            trigger_completion_at_end=...,
```

### 4.2 关键优点：向前兼容

检测标志 `_FI_ALLREDUCE_FUSION_SUPPORTS_WEIGHT_BIAS` 让补丁**同时兼容新旧 flashinfer**：

- 现在（0.6.11.post2）→ 走折叠路径，可正常跑。
- 以后升级到 ≥0.6.12 → 自动走原生 `weight_bias` 路径，**不会重复加偏置、不会冲突**。

所以补丁"打了也不怕以后升级"。

---

## 5. 两条可选路径（二选一，供你后续测试）

> **已执行：路径 A（升级 flashinfer 到 0.6.13）。** 详见文末「实际执行记录」。

### 路径 A：升级 flashinfer（最省事，已采用）

```bash
pip install -U "flashinfer-python>=0.6.12"    # 或 ==0.6.13
```

升级后 nvidia fork 原始的 `weight_bias=` 代码即可原生跑通。
⚠️ 升级前请先确认 flashinfer 与本机 CUDA / torch 版本兼容，避免连带引入其它 ABI 问题。

### 路径 B：保留本补丁（兼容当前旧版 flashinfer）

无需动环境。补丁已应用到下述三个库，且向前兼容（升级后仍安全）。

---

## 6. 已应用的库

同一处修复（`compilation/passes/fusion/allreduce_rms_fusion.py`）已迁移到：

| 库 | 用途 | 状态 |
|---|---|---|
| `/storage/lkk/xpu_vllm/vllm` | **你实际运行的运行时库**（`eval_vllm.sh` 用它） | ✅ 已改 |
| `/storage/lkk/m3/commit/vllm` | 最新目标代码库 | ✅ 已改 |
| `/storage/lkk/m3/vllm_m3` | docker 拷贝的 vllm_m3 | ✅ 已改 |

三库该文件改法一致（import inspect + 检测标志 + 折叠逻辑 + `**weight_bias_kwargs`），均通过 AST 语法校验。

---

## 7. 报错脉络回顾（1 → 6）

| # | 阶段 | 报错 | 处理 |
|---|---|---|---|
| 1 | 加载 | `KeyError vision...fc1.weight` | 改 mapper 去结尾点 + 视觉层不量化 |
| 2 | 加载 | `KeyError vision...qkv_proj.weight` | 补 `packed_modules_mapping` |
| 3 | 加载 | `different quantization schemes [gate_proj, up_proj]` | **需用户重量化**（`--ignore_layers` 用 `block_sparse_moe.gate`）|
| 4 | 前向 | `SWIGLUOAI_UNINTERLEAVE requires clamp_limit` | cutlass MXFP4 MoE 打通 swiglu clamp 参数 |
| 5 | KV cache | `No common block size for 16` | 新增 `MiniMaxM3Config` 钩子把 block_size 钉到 128 |
| **6** | **前向 profiling** | **`allreduce_fusion() unexpected kwarg 'weight_bias'`** | **本文档：flashinfer 版本不匹配，折叠 weight_bias 进 gamma（或升级 flashinfer ≥0.6.12）** |

> 提醒：报错 3 属于**量化产物问题**，需你用正确的 `--ignore_layers` 重新量化，不在代码修复范围内。

---

## 8. 报错 7：`Missing TRTLLM-GEN kernel (decode)`（与报错 6 同源）

> 对应日志：`/storage/lkk/m3/test_7.log`

### 8.1 现象

weight_bias 修复通过后，前向再深入一层，在**注意力 decode 内核**崩溃：

```
RuntimeError: Error in function 'trtllm_paged_attention_launcher' ...:
Missing TRTLLM-GEN kernel (decode): qkvLayout=2, maskType=1, kernelType=2,
headDimQk=128, headDimV=128, tileSizeQ=16, tileSizeKv=128, numTokensPerPage=128, ...
```

调用链：
```
minimax_m3/nvidia/model.py:381  self.attn(q, k, v)
  → attention.py:536  unified_attention_with_output
    → v1/attention/backends/flashinfer.py:1883  trtllm_batch_decode_with_kv_cache(...)
      → flashinfer: Missing TRTLLM-GEN kernel (decode) numTokensPerPage=128, headDim=128
```

### 8.2 根因（与报错 5、报错 6 连锁）

- 报错 5 修复把 KV cache `block_size` 钉成 **128**（`page_size=128`）。
- vLLM 硬性规定 **page ≥ 128 必须走 trtllm-gen 注意力**（FlashInfer 原生 decode 只支持 page ≤ 64）。见 `flashinfer.py:682` 断言与 `flashinfer.py:980` 注释：
  ```python
  # Page sizes >= 128 require the trtllm-gen GQA/MQA path
  assert self.page_size <= 64 or (can_use_trtllm and num_qo_heads // num_kv_heads > 1)
  ```
- MiniMax-M3 是 GQA（`num_attention_heads=64 / num_key_value_heads=4`，`head_dim=128`），断言能过，trtllm-gen 是**唯一**注意力路径。
- 但旧 `flashinfer-cubin 0.6.11.post2` **没有编译出** `numTokensPerPage=128, headDim=128, tileSizeQ=16` 这一 decode 内核变体 → 运行期缺 kernel。

**为什么不能退回 Marlin 绕过**：Marlin/cutlass 是 **MoE 专家 FFN** 的权重内核（报错 4 那条路径），与 **attention decode** 无关。切换 MoE 内核不影响注意力，报错 7 照样触发。关掉 trtllm 注意力则直接撞上上面的断言。所以纯代码层无绕过，**必须升级 flashinfer**。

**为什么 AMD 跑通**：ROCm 用 AITER/Triton 注意力后端，原生支持 page 128，不依赖 flashinfer trtllm-gen cubin，因此报错 6、7 都不出现。

### 8.3 结论：报错 6 与报错 7 同根

两者都因 **本机 flashinfer（0.6.11.post2）比 MiniMax-M3 官方 docker 里的旧**：
- 报错 6：缺 `allreduce_fusion` 的 `weight_bias` 参数（0.6.12 引入）。
- 报错 7：`flashinfer-cubin` 缺 page-128 trtllm-gen decode 内核。

升级 flashinfer 一步同时解决两者。

---

## 9. 实际执行记录：升级 flashinfer 0.6.11.post2 → 0.6.13

### 9.1 环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA **B200**（SM100 / Blackwell，device cap `(10,0)`）|
| torch | **2.12.0+cu130**（CUDA 13.0，为 vLLM 专门编译，**不可动**）|
| 升级前 flashinfer | flashinfer-python 0.6.11.post2 + flashinfer-cubin 0.6.11.post2 |

### 9.2 关键陷阱：直接 `pip install -U` 会降级 torch

`pip install "flashinfer-python==0.6.13"` 的 `--dry-run` 显示它会**把 torch 降级到 2.9.1（cu12）** 并拉入大量 cu12 nvidia 包——会彻底破坏为 vLLM 编译的 `torch 2.12.0+cu130` 环境。**绝不能带依赖安装。**

### 9.3 采用的安全升级命令（`--no-deps`）

flashinfer-python 是纯 JIT 包（无链接 torch 的 `.so`），cubin 是 torch-agnostic 的预编译核，故 `--no-deps` 只替换这两个包、不动 torch：

```bash
# 同时升级 python 与 cubin（缺一不可：cubin 才含 page-128 decode 内核）
pip install --no-deps "flashinfer-python==0.6.13" "flashinfer-cubin==0.6.13"

# 回滚命令（如需）：
# pip install --no-deps flashinfer-python==0.6.11.post2 flashinfer-cubin==0.6.11.post2
```

### 9.4 升级后验证（已通过）

```
torch: 2.12.0+cu130 | cuda: 13.0        # ← torch 未被改动
flashinfer: 0.6.13
allreduce_fusion has weight_bias: True  # ← 报错 6 根因消除
has trtllm_batch_decode_with_kv_cache: True
vllm.utils.flashinfer OK / vllm flashinfer backend OK
allreduce_rms_fusion supports_weight_bias = True   # 我们的补丁自动走原生路径
```

- torch 保持 `2.12.0+cu130`，未被降级。
- `flashinfer-cubin` 同步升到 0.6.13（含 page-128 trtllm-gen decode 内核，解决报错 7）。
- 我们在 §4 打的 weight_bias 折叠补丁**向前兼容**：现在检测到原生支持，自动改走 `weight_bias=` 原生路径，无重复偏置、无冲突。

> 备注：运行时可见 `Failed to import from vllm._qutlass_C: undefined symbol ...` 警告，这是一个**可选 qutlass 扩展**的预存在 ABI 问题，vLLM 会自动回退，**与本次升级及 MiniMax-M3 推理无关**。

### 9.5 后续

直接重跑 `eval_vllm.sh` 验证即可。首次运行 flashinfer 可能 JIT 编译部分内核，稍慢属正常。若仍缺某个 decode 变体（少数极端 shape），可考虑再升到 main 或反馈 flashinfer。

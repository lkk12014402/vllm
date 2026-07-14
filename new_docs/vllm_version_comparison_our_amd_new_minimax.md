# MiniMax-M3 vLLM 多版本详细对比：我们的改动 × AMD-Quark × 上游 NEW × 官方 MiniMax Docker

> 本文档目标：把四个 vLLM 代码状态的 **逐文件、逐行差异** 讲清楚，回答：
> 1. 我们到底改了什么、为什么改（含精确 before/after 代码）；
> 2. 我们的 NVIDIA 修复与 **AMD-Quark 版实现** 的对照；
> 3. 最新上游 **NEW** 代码库是否已经支持我们的修改；
> 4. 我们的库与 **官方 MiniMax 发布的 docker image 里的 vllm** 的区别。
>
> ⚠️ 本文档所有结论均经**逐文件 grep/diff 亲自核验**（不采信任何单一工具的模糊匹配）。

---

## 0. 四个对象一览

| 代号 | 路径 | 版本 | 说明 |
|---|---|---|---|
| **OUR** | `/storage/lkk/m3/latest/vllm/` | `v0.19.1rc0-2813-g41414c449` | 基线 `cc56379e2` + 我们的修复（commit `1cb8f8b27`）+ 文档（`41414c449`） |
| **BASE** | （OUR 的基线提交） | `cc56379e2` | 我们修改前的起点 |
| **NEW** | `/storage/lkk/m3/latest/new/vllm/` | `v0.23.1rc0-1091-g382bbd514` | 更新的上游 vLLM（比 OUR 基线晚很多） |
| **MINIMAX** | `/storage/lkk/m3/latest/minimax_vllm/vllm/` | `0.1.dev17492+g454b47db8`（无 .git，纯发布包） | 官方 MiniMax docker image 内的 vllm |

> 说明：三个库的 MiniMax-M3 模型都在 **`vllm/models/minimax_m3/`**（非标准的
> `model_executor/models/`），分 `nvidia/`（CUDA/Blackwell）、`amd/`（ROCm）、`common/` 三套。

---

## 1. 我们改了什么（OUR vs BASE，逐文件精确 diff）

我们的全部代码改动 = 单个提交 **`1cb8f8b27 support mxfp4 for m3`**，共 **6 个代码文件**
（第二个提交 `41414c449` 只加 `new_docs/`，不改代码）。核心目标：让 vLLM 在
Blackwell(SM100) 上正确加载并推理 MiniMax-M3 的 **compressed-tensors MXFP4 / 混合精度** 模型。

### 1.1 `vllm/models/minimax_m3/nvidia/model.py`（+14 行）

**改动 A：给两个顶层类加 `packed_modules_mapping`**
```python
# 新增（MiniMaxM3SparseForCausalLM 与 ...ForConditionalGeneration 各加一份）
packed_modules_mapping = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}
```
**改动 B：视觉塔 fc1/fc2 子串映射，去掉尾点**
```python
# before (BASE)
orig_to_new_substr={
    ".mlp.fc1.": ".fc1.",
    ".mlp.fc2.": ".fc2.",
}
# after (OUR)
orig_to_new_substr={
    ".mlp.fc1": ".fc1",
    ".mlp.fc2": ".fc2",
}
```

**为什么**：
- `packed_modules_mapping` 让 compressed-tensors 加载器知道 checkpoint 里分开的
  `q/k/v_proj`、`gate/up_proj` 要融合进 vLLM 的 `qkv_proj` / `gate_up_proj`，并据此为每个
  分片推断量化 scheme。缺它 → 融合层加载命名/分片不匹配。
- 量化后视觉 MLP 张量名尾缀不再固定是 `.fc1.`（变成 `.fc1.weight_packed`/`.fc1.weight_scale`
  等），带尾点的旧映射匹配不到 → 视觉层 KeyError。去尾点后能覆盖 `.weight` /
  `.weight_packed` / `.weight_scale` 等各种后缀。

> **重要对照（见 §2）**：这两处**恰好就是 AMD 团队早已写进 `amd/model.py` 的做法**——
> AMD 侧基线就有 `packed_modules_mapping`（行 1054/1129）和无尾点的 `.mlp.fc1`（行 1140/1141），
> 但一直没同步到 `nvidia/model.py`。我们本质是**把 AMD 侧已修好的东西补进 NVIDIA 侧**。

### 1.2 `vllm/model_executor/models/config.py`（+47 行）

**新增 `MiniMaxM3Config`**，并注册给 `MiniMaxM3SparseForCausalLM` /
`MiniMaxM3SparseForConditionalGeneration`：
```python
class MiniMaxM3Config(VerifyAndUpdateConfig):
    @classmethod
    def verify_and_update_config(cls, vllm_config):
        ...
        sparse_block_size = sparse_cfg.get("sparse_block_size")  # 128
        ...
        cache_config.block_size = sparse_block_size
        cache_config.user_specified_block_size = True
```

**为什么**：MiniMax-M3 前几层是 dense full-attention（接受默认 block_size=16），其余层是
block-sparse（indexer/稀疏内核只支持 `block_size == sparse_block_size == 128`）。vLLM 的通用
block-size 自动选择只看**第一个**非 SSM 注意力后端（正好是 dense 层），于是 block_size 停在
16，KV-cache 初始化在 `select_common_block_size` 报 `No common block size for 16.`。
把 block_size 钉成 128 并标 user-specified，dense（MultipleOf(16)）与 sparse（需 128）才能一致。

### 1.3~1.5 MXFP4 MoE 的 clamped SwiGLU-OAI 激活（三处联动）

MiniMax-M3 的 MoE 激活是 **clamped SwiGLU-OAI**（`swigluoai_uninterleave`），需要
`alpha/beta/clamp_limit` 做 `silu_and_mul_with_clamp`。BASE 的 MXFP4 W4A4 cutlass 路径按普通
SiLU/GELU 处理，**丢掉 clamp** → 激活数值错、精度崩。三处把参数从模型层透传到 cutlass kernel：

**1.3 `compressed_tensors_moe_w4a4_mxfp4.py`（+8 行）**
```python
# before
return mxfp4_moe_quant_config(
    w1_scale=layer.w13_weight_scale,
    w2_scale=layer.w2_weight_scale,
)
# after
return mxfp4_moe_quant_config(
    w1_scale=layer.w13_weight_scale,
    w2_scale=layer.w2_weight_scale,
    gemm1_alpha=getattr(layer, "swiglu_alpha", None),
    gemm1_beta=getattr(layer, "swiglu_beta", None),
    gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
)
```

**1.4 `fused_moe/config.py`（+11 行）**：`mxfp4_moe_quant_config` 新增三个可选参数
`gemm1_alpha/gemm1_beta/gemm1_clamp_limit`，塞进 `FusedMoEQuantConfig.make(...)`。普通激活时为
`None`。

**1.5 `fused_moe/experts/cutlass_moe.py`（+24 行）**：
- `run_cutlass_moe_mxfp4` 新增 `gemm1_clamp_limit/gemm1_alpha/gemm1_beta`；
- gated 分支从 `apply_moe_activation(activation, c2, c1)` 改成带 `clamp_limit/alpha/beta`；
- `CutlassExpertsMxfp4` 支持列表新增 `MoEActivation.SWIGLUOAI_UNINTERLEAVE`。

> 注：`SWIGLUOAI_UNINTERLEAVE` 这个 enum 和 `apply_moe_activation` 的 clamp 能力在 BASE 的
> `fused_moe/activation.py` **本就存在**（`SWIGLUOAI_UNINTERLEAVE → silu_and_mul_with_clamp`）。
> 我们做的是**把这条已有能力接进 MXFP4 cutlass 路径**（BASE 的 cutlass gated 分支没传 clamp）。

**设计要点**：三个参数一律 `None` 兜底 → 对非 M3、普通激活的 MXFP4 MoE **完全无影响**（向后兼容），
只有 M3 这类 clamped 激活才启用。

### 1.6 `vllm/compilation/passes/fusion/allreduce_rms_fusion.py`（+22 行）

**新增基于 `inspect.signature` 的 flashinfer 兼容检测**：
```python
_FI_ALLREDUCE_FUSION_SUPPORTS_WEIGHT_BIAS = flashinfer_comm is not None and (
    "weight_bias" in inspect.signature(flashinfer_comm.allreduce_fusion).parameters
)
...
if _FI_ALLREDUCE_FUSION_SUPPORTS_WEIGHT_BIAS:
    weight_bias_kwargs = {"weight_bias": weight_bias}
else:
    weight_bias_kwargs = {}
    if weight_bias != 0.0:
        rms_gamma = rms_gamma + weight_bias   # 折进 gamma，数学等价
flashinfer_comm.allreduce_fusion(..., **weight_bias_kwargs, ...)
```

**为什么**：GemmaRMSNorm 的 `(1+weight)` 偏移用 `weight_bias=1.0` 表达。**打过补丁的 flashinfer**
支持 kernel 内 `weight_bias` 参数；**stock flashinfer（如 0.6.11）没有**，直接传会 TypeError。
探测签名：有就传，没有就把偏移折进 `rms_gamma`（`normed*(gamma+bias)` 等价）。让本库在**有/无
补丁两种 flashinfer** 上都能跑。

---

## 2. 我们的 NVIDIA 修复 × AMD-Quark 版实现 对照

`amd/` 与 `nvidia/` 是**两套平台特定实现**（ROCm vs CUDA/Blackwell），不是同一份代码。核心结构：

| 能力 | `amd/`（ROCm） | `nvidia/`（CUDA/SM100） |
|---|---|---|
| SwiGLU-OAI 激活（dense MLP） | Triton `swiglu_oai_split`（fp32，`amd/ops/swiglu_oai.py`） | `SiluAndMulWithClamp`（CUDA op） |
| MoE 激活 | `activation="swigluoai_uninterleave"` + clamp | 同（`swigluoai_uninterleave`） |
| 稀疏注意力 / indexer | `amd/ops/index_topk.py` + `amd/ops/sparse_attn.py`（Triton） | `nvidia/indexer_msa.py` + `nvidia/sparse_attention_msa.py`（fmha_sm100 + Triton） |
| `packed_modules_mapping` | ✅ 基线就有（`amd/model.py:1054/1129`） | ❌ 基线无 → **我们补上** |
| 视觉 fc1/fc2 映射 | ✅ 基线即无尾点 `.mlp.fc1`（`amd/model.py:1140/1141`） | ❌ 基线带尾点 `.mlp.fc1.` → **我们改成无尾点** |

**核心结论**：AMD-Quark 路径（`amd/model.py`）的作者**早已把加载/命名相关的正确写法写好**
（packed_modules_mapping、无尾点 fc 映射）；但 NVIDIA 路径（`nvidia/model.py`）没同步。
**我们的 §1.1 修复，本质是把 AMD 侧已验证的做法搬到 NVIDIA 侧对齐**。

至于激活：两侧都用 `swigluoai_uninterleave`。差异只是 **kernel 实现选择**——AMD 特意保留
fp32 精度的 Triton kernel（注释解释：`silu_and_mul_with_clamp` 在 ROCm 上虽然更快但会 rounding，
而该激活喂给 MXFP8 quant+MoE，精度更重要）；NVIDIA 走 CUDA 的 `SiluAndMulWithClamp` +（我们接通的）
cutlass MXFP4 clamp 路径。

> AMD-Quark 版模型本身（权重）与我们自量化的 compressed-tensors 模型是**两条量化路径**，
> 之前的报告已述（quark 是 file-to-file、原始 schema）。此处对比的是 **vLLM 侧的两套 model 代码**。

---

## 3. OUR vs NEW（最新上游 `v0.23.1rc0-1091-g382bbd514`）—— NEW 是否已支持？

> ⚠️ 本节结论经**逐文件亲自 grep/diff 核验**。（初次用探索代理 grep 得到"NEW 已全含"的结论
> 是**错误**的——grep 命中了同名词的其它函数。以下为实测更正后的权威结果。）

### 3.1 六处修改在 NEW 的真实状态

| # | 我们的修改 | NEW 是否已有 | 证据 |
|---|---|---|---|
| 1 | `nvidia/model.py` `packed_modules_mapping` | ✅ **已有** | NEW `nvidia/model.py` 有 2 处 packed_modules_mapping |
| 1 | `nvidia/model.py` fc1/fc2 无尾点 | ❌ **NEW 仍是带尾点** `.mlp.fc1.` | NEW `nvidia/model.py:1078` |
| 2 | `MiniMaxM3Config`（block_size 钉 128） | ❌ **NEW 无** | NEW `model_executor/models/config.py` 中 `MiniMax`/`sparse_block_size` 命中 **0** |
| 3 | `compressed_tensors_moe_w4a4_mxfp4` 传 swiglu | ❌ **NEW 无** | NEW 该文件 `swiglu_alpha/limit` 命中 0 |
| 4 | `mxfp4_moe_quant_config` 加 gemm1_* | ❌ **NEW 无** | NEW `mxfp4_moe_quant_config`（config.py:840）函数体**只有 w1/w2_scale**，与 BASE 一致 |
| 5 | `run_cutlass_moe_mxfp4` clamp + SWIGLUOAI_UNINTERLEAVE | ❌ **NEW 无** | NEW `cutlass_moe.py` `gemm1_clamp_limit`/`SWIGLUOAI_UNINTERLEAVE` 命中 **0** |
| 6 | allreduce `inspect.signature` 兼容 | ❌ **NEW 无** | NEW `allreduce_rms_fusion.py` `inspect.signature` 命中 0（NEW 直接传 `weight_bias=weight_bias`，假定 flashinfer 支持） |

**结论：NEW 只重合 1 项（packed_modules_mapping），其余 5 项（含 fc 尾点、block_size 钉定、
MXFP4 clamp 三连、allreduce 兼容）NEW 都没有。** 即：
> **最新上游 NEW 并不能直接跑我们的 compressed-tensors MXFP4 M3 模型**——它缺 MXFP4 MoE 的
> clamp 透传（激活会算错）、缺 block_size 钉定（KV-cache 初始化会报 `No common block size`）、
> 视觉 fc 映射尾点问题依旧、且 allreduce 在 stock flashinfer 上可能 TypeError。

> 关于 NEW 里那些 `gemm1_clamp_limit`：NEW 的**其它** quant config（`nvfp4_*`/`fp8_*`/
> `mxfp4_w4a16`/`mxfp4_mxfp8`/`ocp_mx_*` 等）确实带 `gemm1_clamp_limit`，但**唯独 W4A4 的
> `mxfp4_moe_quant_config` 没有**——这正是我们补的那条 compressed-tensors MXFP4 路径。

### 3.2 `models/minimax_m3/` 目录级差异

- **OUR ⊂ NEW（文件层面）**：OUR 的 minimax_m3 文件 NEW 全都有（无缺失）。
- **NEW 多 2 个 AMD 文件**：`amd/ops/sparse_pa.py`、`amd/sparse_attention_msa.py`
  （NEW 对 AMD 稀疏注意力做了进一步拆分/重构，与 NVIDIA MXFP4 路径无关）。
- 但**文件同名 ≠ 内容相同**：`nvidia/model.py`、`common/*`、`amd/*` 多处内容有差异（NEW 更新）。
  关键是上面 §3.1 的量化/加载点——NEW 没有我们的修复。

---

## 4. OUR vs MINIMAX（官方 docker image `0.1.dev17492+g454b47db8`）

> 有意思的历史关联：这个 `g454b47db8` 正是之前 `test_9.log` 里出现的神秘版本——
> 也就是说**官方 MiniMax docker 用的就是这个 vllm 构建**。

### 4.1 minimax_m3 目录：MINIMAX 缺的文件（精确集）

OUR 有、**MINIMAX 缺**的只有 3 个：
```
amd/ops/index_topk.py        （AMD Triton top-k 索引 kernel）
amd/ops/sparse_attn.py       （AMD Triton 稀疏注意力）
nvidia/indexer_msa.py        （NVIDIA SM100 indexer：fmha_sm100 OnlyScore + Triton top-k）
```
> 注意：MINIMAX **有** `common/ops/index_topk.py`、`common/ops/sparse_attn.py`、
> `nvidia/sparse_attention_msa.py`（这些不缺）。缺的是 AMD 的两个 ops 和 NVIDIA 的独立 indexer 模块。
> 说明 MINIMAX 的 SM100 indexer/稀疏栈**拆分粒度更粗**（更多逻辑内联在 model.py / 共享模块里），
> 而 OUR/NEW 把 NVIDIA indexer 单独拆成 `nvidia/indexer_msa.py`。

### 4.2 权重加载机制：`stacked_params_mapping`（旧式） vs `packed_modules_mapping`（声明式）

**MINIMAX 的 `nvidia/model.py` 用旧式的手动 `stacked_params_mapping`**（`load_weights` 内循环），
**没有** `packed_modules_mapping`：
```python
# MINIMAX nvidia/model.py:845
stacked_params_mapping = [
    (".qkv_proj", ".q_proj", "q"),
    (".qkv_proj", ".k_proj", "k"),
    (".qkv_proj", ".v_proj", "v"),
    (".qkv_proj", ".index_q_proj", "index_q"),
    (".qkv_proj", ".index_k_proj", "index_k"),
    (".gate_up_proj", ".gate_proj", 0),
    (".gate_up_proj", ".up_proj", 1),
]
```
而 **OUR/NEW 用声明式 `packed_modules_mapping`**（交给通用加载器处理融合分片的量化 scheme 推断）。
两者对 bf16/简单量化都能加载，但**声明式对 compressed-tensors 分片量化 scheme 的推断更完整**，
这也是我们在 NVIDIA 侧改用它的原因。

MINIMAX 的 fc 映射仍是带尾点 `.mlp.fc1.`（`nvidia/model.py:1004`）。

### 4.3 量化路径：MINIMAX 不含 compressed-tensors MXFP4 的 clamp 支持

**决定性证据**：MINIMAX 全树 grep `gemm1_clamp_limit` / `gemm1_alpha` / `gemm1_beta` /
`SWIGLUOAI_UNINTERLEAVE` / `mxfp4_moe_quant_config` 传 swiglu —— **全部 0 命中**。
即 MINIMAX **没有** MXFP4 MoE 的 clamped SwiGLU-OAI 透传（§1.3~1.5），也没有
`MiniMaxM3Config` 的 block_size 钉定。

其 `load_weights` 里有 ModelOpt 风格的 block-scale 处理：
```python
# MINIMAX nvidia/model.py:889-892
# The checkpoint stores block scales as ``weight_scale_inv``; the
# ModelOpt MXFP8 layers expose them as ``weight_scale``.
if "weight_scale_inv" in name:
    name = name.replace("weight_scale_inv", "weight_scale")
```
> 注：`weight_scale_inv→weight_scale` 这段 **OUR 的 `nvidia/model.py` 也有**（同样 3 处，
> 行 889-892），并非 MINIMAX 独有——它只是说明 M3 的 model 代码本身兼容 ModelOpt/MXFP8 的
> block-scale 命名，**不能**单独用来证明"MINIMAX 只走 ModelOpt"。真正的区分证据是上面那组
> MXFP4-clamp 相关的 **0 命中**。

**结论：官方 docker 的 M3 缺少我们这条 compressed-tensors MXFP4（+clamped SwiGLU-OAI）路径的
关键支持**——MXFP4 MoE clamp 三连、`MiniMaxM3Config` block_size 钉定，MINIMAX **都没有**；
它面向的是 ModelOpt/MXFP8 量化。

### 4.4 逐点对照表（OUR vs MINIMAX）

| 能力 | OUR | MINIMAX |
|---|---|---|
| `packed_modules_mapping`（NVIDIA） | ✅ | ❌（用 `stacked_params_mapping`） |
| 视觉 fc1/fc2 映射 | 无尾点 | 带尾点 `.mlp.fc1.` |
| `MiniMaxM3Config`（block_size 钉 128） | ✅ | ❌（0 命中） |
| MXFP4 MoE clamped SwiGLU-OAI 三连 | ✅ | ❌（0 命中；面向 ModelOpt/MXFP8） |
| compressed-tensors MXFP4 支持 | ✅ | ❌ |
| allreduce `inspect.signature` 兼容 | ✅ | ❌（直接传 `weight_bias`） |
| 独立 `nvidia/indexer_msa.py` | ✅ | ❌（更粗粒度） |

---

## 5. 三方总览（一张表看清）

| 修改点 | OUR | BASE(cc56379e2) | NEW(g382bbd514) | MINIMAX(g454b47db8) |
|---|---|---|---|---|
| nvidia `packed_modules_mapping` | ✅ 加 | ❌ | ✅ 已有 | ❌（stacked_params） |
| nvidia fc1/fc2 去尾点 | ✅ | ❌带尾点 | ❌带尾点 | ❌带尾点 |
| `MiniMaxM3Config` block_size=128 | ✅ | ❌ | ❌ | ❌ |
| `mxfp4_moe_quant_config` gemm1_* | ✅ | ❌ | ❌ | ❌ |
| compressed_tensors 传 swiglu_* | ✅ | ❌ | ❌ | ❌ |
| cutlass MXFP4 clamp + UNINTERLEAVE 接线 | ✅ | ❌ | ❌ | ❌ |
| allreduce inspect.signature 兼容 | ✅ | ❌ | ❌ | ❌ |
| compressed-tensors **MXFP4** M3 可跑 | ✅ | ❌ | ❌（缺 clamp/block_size） | ❌（面向 ModelOpt/MXFP8） |

**一句话总结**：
- 我们的 6 处修改里，NVIDIA `packed_modules_mapping` 在 NEW 已合入（NEW 更新使然），但**其余
  5 项（尤其 MXFP4 MoE 的 clamped SwiGLU-OAI 三连 + block_size 钉定 + allreduce 兼容）在
  NEW / MINIMAX / BASE 都没有**。
- **NEW 更新但不支持我们的 compressed-tensors MXFP4 M3 路径**；官方 **MINIMAX docker 走的是
  ModelOpt MXFP8**，也不支持。
- 我们的 NVIDIA 加载修复（packed_modules + fc 去尾点）**本质是对齐 AMD 侧早已存在的写法**。

---

## 6. 结论与建议

1. **要跑 compressed-tensors MXFP4 / MoE-MXFP4+MXFP8 混合 的 M3，目前只有 OUR 可用**；
   NEW、MINIMAX、BASE 都缺关键的 MXFP4 clamp 透传与 block_size 钉定。
2. **若要迁到 NEW（为拿上游新特性）**：需把我们的 5 项修复（1.2~1.6 + fc 去尾点）**cherry-pick /
   重新移植**到 NEW；packed_modules_mapping 已在 NEW，可跳过。注意 NEW 的 minimax_m3 有额外
   AMD 重构文件，移植时以 §1 的 6 处为准逐一核对。
3. **官方 MINIMAX docker** 面向 ModelOpt MXFP8，与我们的量化路线不同；不建议直接混用。
4. 复核方式（可复现）：对任一库执行
   `grep -c "gemm1_clamp_limit"` 于 `mxfp4_moe_quant_config` 函数体、
   `grep "MiniMaxM3Config" model_executor/models/config.py`、
   `grep "\.mlp\.fc1" models/minimax_m3/nvidia/model.py`、
   `grep "inspect.signature" .../allreduce_rms_fusion.py` 即可快速判定是否含我们的修复。

---

## 7. 关联文档

- 改动总览与安装：`new_docs/README_minimax_m3_vllm_changes.md`
- 各问题专题：`vllm_gate_up_proj_quant_mismatch.md`、`vllm_cutlass_mxfp4_clamp_fix.md`、
  `vllm_block_size_sparse_attention_fix.md`、`vllm_allreduce_weight_bias_fix.md`、
  `vllm_qkv_fix_and_amd_comparison.md`、`vllm_vision_quant_bug_analysis.md`
- 量化路径：`minimax_m3_mixed_precision_quant.md`、`auto_round_model_free_vs_rtn_analysis.md`

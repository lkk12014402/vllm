# vLLM 对 MiniMax-M3 MXFP4/混合精度的改动说明与安装指南

> 本库：`/storage/lkk/m3/latest/vllm/`
> 版本：`v0.19.1rc0-2813-g41414c449`（基线 `cc56379e2` + 我们的改动）
> 目标：让 vLLM 能在 NVIDIA Blackwell（B200 / SM100）上正确加载并推理
> **MiniMax-M3** 的 **MXFP4**（以及 MoE-MXFP4 + 其余 MXFP8 混合精度）量化模型。

---

## 1. 背景

MiniMax-M3 是一个 **MoE + 视觉多模态** 模型，含若干特殊结构：
- **稀疏注意力 + lightning indexer**（前几层 dense full-attention，其余层 block-sparse）；
- **MoE 路由专家** `block_sparse_moe.experts.w1/w2/w3`（占全模型 96.7% 参数）；
- 激活函数是 **clamped SwiGLU-OAI**（`swigluoai`，含 alpha/beta/clamp_limit），不是普通 SiLU。

我们用 auto-round（model-free）把它量化成 **MXFP4**（MoE）/ **MXFP8**（共享专家等），
导出为 compressed-tensors `mxfp4-pack-quantized`。在 stock vLLM 上加载 / 推理会连续报错，
本库对这些问题做了针对性修复。

所有改动集中在一个提交 **`1cb8f8b27 support mxfp4 for m3`**，共 **6 个代码文件**
（另一个提交 `41414c449 add doc` 只增加 `new_docs/` 文档，不改代码）。

---

## 2. 代码改动清单（6 个文件）

| # | 文件 | 作用 | 对应问题 |
|---|---|---|---|
| 1 | `vllm/models/minimax_m3/nvidia/model.py` | 加 `packed_modules_mapping`；修视觉 `fc1/fc2` 映射 | 权重加载命名/融合 |
| 2 | `vllm/model_executor/models/config.py` | 新增 `MiniMaxM3Config`，锁定 KV-cache block_size | 稀疏注意力 block_size |
| 3 | `.../compressed_tensors_moe/compressed_tensors_moe_w4a4_mxfp4.py` | 向 MoE quant config 传 SwiGLU-OAI 参数 | MXFP4 MoE 激活 clamp |
| 4 | `vllm/model_executor/layers/fused_moe/config.py` | `mxfp4_moe_quant_config` 增加 gemm1_alpha/beta/clamp_limit | 同上 |
| 5 | `vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py` | Cutlass MXFP4 kernel 应用 clamped SwiGLU-OAI | 同上 |
| 6 | `vllm/compilation/passes/fusion/allreduce_rms_fusion.py` | 兼容有/无 `weight_bias` 的 flashinfer | GemmaRMSNorm allreduce 融合 |

---

## 3. 每处改动的原因与内容

### 3.1 `models/minimax_m3/nvidia/model.py` —— 权重加载命名 / 融合映射

**加 `packed_modules_mapping`**（给两个顶层类都加）：
```python
packed_modules_mapping = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}
```
**为什么**：compressed-tensors 加载器靠 `packed_modules_mapping` 把 checkpoint 里分开的
`q/k/v_proj`、`gate/up_proj` 权重映射到 vLLM 融合的 `qkv_proj` / `gate_up_proj` 层，并据此
推断每个分片的量化 scheme。缺了它，融合层加载会命名/分片不匹配。

**修视觉 `fc1/fc2` 子串映射**：
```python
# 原来: ".mlp.fc1." -> ".fc1."   （尾部多一个点）
# 改为: ".mlp.fc1"  -> ".fc1"
```
**为什么**：量化后视觉塔 MLP 的张量名后缀变成 `.fc1.weight_packed` / `.fc1.weight_scale`
等（不再一定是 `.fc1.`），旧的带尾点子串匹配不到，导致视觉层权重 KeyError。去掉尾点即可
覆盖 `.weight` / `.weight_packed` / `.weight_scale` 等各种后缀。

### 3.2 `models/config.py` —— 稀疏注意力 KV-cache block_size

新增 `MiniMaxM3Config`（注册给 `MiniMaxM3SparseForCausalLM` /
`MiniMaxM3SparseForConditionalGeneration`），把 KV-cache `block_size` 锁定为模型的
`sparse_attention_config.sparse_block_size`（128），并标记为 user-specified。

**为什么**：MiniMax-M3 的前几层是 dense full-attention（接受默认 block_size=16），其余是
block-sparse（indexer/稀疏内核只支持 block_size = sparse_block_size = 128）。vLLM 的通用
block-size 自动选择只看**第一个**非 SSM 注意力后端——正好是 dense 层，于是 block_size 停在
16，稀疏层无法满足，KV-cache 初始化在 `select_common_block_size` 报
`No common block size for 16.`。锁定为 128 后，dense（MultipleOf(16)）与 sparse（需要 128）
才能达成一致。

### 3.3 + 3.4 + 3.5 —— MXFP4 MoE 的 clamped SwiGLU-OAI 激活（三处联动）

MiniMax-M3 的 MoE 激活是 **clamped SwiGLU-OAI**（`swigluoai_uninterleave`），需要
`alpha / beta / clamp_limit` 三个参数做 `silu_and_mul_with_clamp`。stock vLLM 的 MXFP4
MoE 路径只按普通 SiLU/GELU 处理，**丢掉了 clamp**，导致激活数值错、精度崩。

三处改动把这些参数从模型层一路透传到 Cutlass kernel：
- **`compressed_tensors_moe_w4a4_mxfp4.py`**：从 layer 读
  `swiglu_alpha/beta/limit`（`getattr(..., None)`），传给 `mxfp4_moe_quant_config`。
- **`fused_moe/config.py`**：`mxfp4_moe_quant_config` 新增
  `gemm1_alpha/gemm1_beta/gemm1_clamp_limit` 三个可选参数，塞进 `FusedMoEQuantConfig`。
  普通 SiLU/GELU 时为 `None`。
- **`fused_moe/experts/cutlass_moe.py`**：
  - 支持列表新增 `MoEActivation.SWIGLUOAI_UNINTERLEAVE`；
  - `run_cutlass_moe_mxfp4` 增加 `gemm1_clamp_limit/alpha/beta`，在
    `apply_moe_activation(...)` 时带上 `clamp_limit/alpha/beta`，真正执行 clamped SwiGLU-OAI。

**为什么这么设计**：参数用 `None` 兜底 → 对非 M3、普通激活的 MXFP4 MoE 完全无影响
（向后兼容），只有 M3 这类带 clamp 的激活才会启用。

### 3.6 `allreduce_rms_fusion.py` —— 兼容 flashinfer 的 `weight_bias`

`allreduce_fusion`（GemmaRMSNorm + allreduce 融合）里，GemmaRMSNorm 的 `(1 + weight)` 偏移
用 `weight_bias=1.0` 表达。**打过补丁的 flashinfer** 支持在 kernel 内用 `weight_bias`
参数施加该偏移；**stock flashinfer（如 0.6.11）没有这个参数**，直接传会报 TypeError。

改动：用 `inspect.signature` 探测 flashinfer 的 `allreduce_fusion` 是否有 `weight_bias`：
- 有 → 照常传 `weight_bias`；
- 没有 → 把偏移折进 `rms_gamma`（`rms_gamma = rms_gamma + weight_bias`，数学等价
  `normed * (gamma + bias)`），不再传该参数。

**为什么**：让本库在**有/无补丁两种 flashinfer** 上都能跑，不强制要求特定 flashinfer 构建。

---

## 3b. MXFP4 MoE 后端选择与 GPU 要求

**要跑 MXFP4，不是只能走 Cutlass。** vLLM 对 compressed-tensors MXFP4 MoE 会**按 GPU 自动
在三条后端里选**（`compressed_tensors_moe/compressed_tensors_moe_w4a4_mxfp4.py` 的 `__init__`）：

```
优先 CutlassExpertsMxfp4  →  否则 XPUExpertsMxFp4(Intel GPU)  →  否则 MarlinExperts(兜底)
```

| 后端 | GPU 要求（compute capability） | 量化性质 | 说明 |
|---|---|---|---|
| **CutlassExpertsMxfp4** | **SM100 ~ SM119**（`capability >= 100 && < 120`） | 真 **W4A4**（权重+激活都 MXFP4） | Blackwell（B200/B100、RTX 50 系）。kernel 需 **CUDA ≥ 12.9** 编译。**我们验证过的路径**（gsm8k 95.2%） |
| **XPUExpertsMxFp4** | Intel XPU | — | 非 NVIDIA |
| **MarlinExperts**（兜底） | **SM75+**（`has_device_capability((7,5))`） | **W4A16 weight-only**（仅权重 MXFP4，激活高精度） | Turing 及以后（T4 / A100(SM80) / H100(SM90) / RTX20+…） |

判定来源（代码级证据）：
- Cutlass 门槛在 C++ kernel：`csrc/.../fp4/mxfp4_experts_quant.cu` 的
  `mxfp4_experts_quant_sm_supported()` → `return cuda_device_capability >= 100 && < 120;`
  （Python 侧 `CutlassExpertsMxfp4._supports_current_device()` 调
  `ops.mxfp4_experts_quant_supported(capability)`）。
- Marlin 门槛：`MarlinExperts._supports_current_device()` →
  `p.is_cuda() and p.has_device_capability((7, 5))`（SM75+）。

### 关键结论与注意

1. **Blackwell（SM100–119，如 B200）** → 走 **Cutlass**，真 W4A4，是我们验证过精度的路径。
2. **A100(SM80) / H100(SM90)** → **自动 fallback 到 Marlin**，能加载 MXFP4 模型，但为
   **W4A16 weight-only**（激活跑高精度，不是真 W4A4）。
3. **SM120（部分 RTX 50）** → 注意 Cutlass 上界是 `< 120`（不含），会落到 Marlin。
4. ⚠️ **我们的 clamped SwiGLU-OAI 修复（§3.3~3.5）只作用在 Cutlass 分支**。若在非 Blackwell
   卡上走了 Marlin，MoE 激活由 Marlin 路径处理，M3 的 `swigluoai_uninterleave` clamp 在该路径
   **未经我们验证**——在 A100/H100 上跑 M3 MXFP4 前，建议先小规模验证精度。
5. 想强制走 Marlin（例如调试）可用 `VLLM_DISABLED_KERNELS`/环境覆盖，但注意 MoE 的后端选择是
   `_supports_current_device()` 硬判定，Blackwell 上默认恒走 Cutlass（详见
   `new_docs/vllm_marlin_vs_cutlass_mxfp4_moe.md`）。

---

## 4. 安装

### 4.1 环境要求（关键版本，来自 `requirements/cuda.txt`）

| 组件 | 版本 | 说明 |
|---|---|---|
| GPU | NVIDIA Blackwell（SM100，如 B200） | MXFP4 cutlass / FA4 需要 |
| torch | **2.11.0**（cu13） | build-system 与 requirements 均 pin 2.11.0 |
| torchvision | 0.26.0 | |
| flashinfer-python / -cubin | **0.6.13** | MXFP8 GEMM / attention |
| nvidia-cutlass-dsl[cu13] | **4.5.2** | MXFP4 MoE / ViT FA4（**需 torch≥2.12 才能编译 CuTe FA4**，见"注意事项"） |

> ⚠️ **torch 版本注意**：本库 build 依赖 pin `torch==2.11.0`。但我们此前实测发现：
> `nvidia-cutlass-dsl 4.5.2` 的 **CuTe DSL（ViT flash-attn，SM100 上默认 fa_version=4）
> 只有在 torch≥2.12 才能成功编译**；torch 2.11 下会 `TypeError: incompatible function
> arguments`。若你在 torch 2.11 环境跑到视觉塔即时编译报错，两种解法：
> ① 给推理加 `--mm-encoder-attn-backend TORCH_SDPA` 绕开 CuTe；
> ② 或把 torch 升到 2.12.0+cu130（但需重编本库的 C 扩展，见 4.3）。

### 4.2 从源码安装（推荐 editable，因含 C++/CUDA 扩展）

```bash
cd /storage/lkk/m3/latest/vllm

# 建议用独立 venv（本库其它环境用的是 uv venv）
# 关键：pin 的 torch 已由 requirements 指定，安装时用 --no-build-isolation
# 让编译使用当前环境已装的 torch，避免 build 隔离重新拉 torch。

pip install -r requirements/build.txt      # cmake/ninja/setuptools-scm/torch 等
pip install --no-build-isolation -e .
```

若只想用**预编译**方式（不本地编译 CUDA 扩展、直接用已有 .so）：
```bash
VLLM_USE_PRECOMPILED=1 pip install --no-build-isolation -e .
```

安装后校验：
```bash
python -c "import vllm; print(vllm.__version__)"
# 期望: 0.19.1rc0...g41414c449 (或对应你的构建号)
```

### 4.3 若升级到 torch 2.12（可选，为让 ViT CuTe FA4 编译通过）

```bash
# 注意 --no-deps / 指定 cu130 index，避免连带降级；升级后需重编 vLLM C 扩展
pip install --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  torch==2.12.0 torchvision==0.27.0 triton==3.7.0
cd /storage/lkk/m3/latest/vllm
pip install --no-build-isolation -e .   # 重新编译 _vllm_fa2_C/_vllm_fa3_C/_flashmla_C/_qutlass_C 等
```

---

## 5. 运行

### 5.1 serve（`new_docs/serve_m3_mxfp4.sh`）
```bash
export CUDA_VISIBLE_DEVICES=2,3
vllm serve \
  /storage/lkk/m3/MiniMax-M3-MXFP4MoE-MXFP8-conservative \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --tool-call-parser minimax_m3 \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice \
  --port 8008
```

### 5.2 lm_eval（`new_docs/eval_vllm_gsm8k.sh` 等）
```bash
export CUDA_VISIBLE_DEVICES=0,1
lm_eval --model vllm \
  --model_args "pretrained=/storage/lkk/m3/MiniMax-M3-MXFP4MoE-MXFP8-conservative,tensor_parallel_size=2,max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,add_bos_token=True,gpu_memory_utilization=0.8,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False,reasoning_parser=minimax_m3" \
  --tasks gsm8k --batch_size 64 --output_path lm_eval_results
```
> 参考精度：方案①（MoE-MXFP4 + 共享专家 MXFP8）在 gsm8k 上 flexible≈0.952 / strict≈0.953。

---

## 6. 注意事项

1. **首次加载较慢**：MXFP8 GEMM 走 flashinfer，会触发 `[AutoTuner] Tuning mxfp8_gemm`
   （约 23 profile，数分钟）；MXFP4 MoE 走 Cutlass，不经 flashinfer autotuner。属一次性
   warmup，结果会缓存。详见 `new_docs/vllm_mxfp8_autotune_vs_mxfp4_cutlass.md`。
2. **ViT CuTe FA4 编译**：见 4.1 的 torch 版本注意；torch 2.11 上如遇编译报错，用
   `--mm-encoder-attn-backend TORCH_SDPA` 绕开。
3. **AMD-Quark 格式模型**：本库面向自量化 compressed-tensors（MXFP4/mixed）路径，**不包含**
   quark 的 torch2.12/pt2e 导入兼容 shim（那是另一条 AMD-Quark 路径的修复，见
   `new_docs/vllm_quark_torch212_pt2e_compat_fix.md`）。若要加载 AMD-Quark 模型需另行接入。

---

## 7. 关联文档（`new_docs/`）

- 权重/命名与量化 scheme：`vllm_gate_up_proj_quant_mismatch.md`、`vllm_qkv_fix_and_amd_comparison.md`、`vllm_vision_quant_bug_analysis.md`
- 稀疏注意力 block_size：`vllm_block_size_sparse_attention_fix.md`
- MXFP4 clamp / SwiGLU-OAI：`vllm_cutlass_mxfp4_clamp_fix.md`
- allreduce weight_bias：`vllm_allreduce_weight_bias_fix.md`
- MoE 后端选择：`vllm_marlin_vs_cutlass_mxfp4_moe.md`、`vllm_mxfp8_autotune_vs_mxfp4_cutlass.md`
- 量化方案：`minimax_m3_mixed_precision_quant.md`、`AutoScheme-MXFP4-MXFP8混合精度调研.md`、`quant_cmd.md`
- auto-round 路径：`auto_round_model_free_vs_rtn_analysis.md`、`auto_round_native_to_original_converter.md`
- 运行脚本：`serve_m3_mxfp4.sh`、`eval_vllm*.sh`

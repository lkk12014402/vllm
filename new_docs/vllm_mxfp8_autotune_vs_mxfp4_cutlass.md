# 方案①（MXFP4 MoE + MXFP8 共享专家）推理日志分析：为什么只 tune mxfp8_gemm

> 日志：`/storage/lkk/m3/test_11.log`（方案① 模型
> `./MiniMax-M3-MXFP4MoE-MXFP8-conservative/`，lm_eval gsm8k，vLLM
> `v0.23.1rc1.dev550+g58d6a6e60`，库路径 `/storage/lkk/xpu_vllm/vllm/vllm`）。
>
> 疑问：为什么日志里只有 `[AutoTuner]: Tuning mxfp8_gemm` 被 tune，MXFP4 没有？
> 之前没有 MXFP8 时，MXFP4 也是会被 tune 的。

---

## 0. 先说结论

**模型没问题**：方案① gsm8k **flexible=0.9522 / strict=0.9530**（≈95.2%），精度很好。

**核心结论**：决定"某个 GEMM 会不会被 tune"的，**不是量化精度（fp4 / fp8），而是它最终
选中的 kernel 是不是 FlashInfer 后端**。那条 `[AutoTuner]: Tuning mxfp8_gemm` 来自
**FlashInfer 专属的 autotuner**（`flashinfer.jit` / `flashinfer/autotuner.py`），它只 tune
FlashInfer 的 kernel。

- **MXFP8**（共享专家，是 Linear 层）→ 选中 `FlashInferCutedslMxfp8LinearKernel` →
  调 `flashinfer.mm_mxfp8` → **走 FlashInfer autotuner → 被 tune**。
- **MXFP4**（路由专家，是 MoE 层）→ 选中 `CutlassExpertsMxfp4` → 用 Cutlass 的 GEMM →
  **不经 FlashInfer autotuner → 日志里看不到它被 tune**。

**MXFP4 MoE 走 Cutlass 是一贯行为，不是这次才"不需要 tune"。**

---

## 1. 日志关键行

```
line 72: [__init__.py:758] Using FlashInferCutedslMxfp8LinearKernel for MXFP8 GEMM
line 73: [compressed_tensors_moe_w4a4_mxfp4.py:53] Using CutlassExpertsMxfp4 for MXFP4 MoE
line 203:[flashinfer.py:489] Using TRTLLM attention (use_trtllm_attention=1)
line 217:autotuner.py:651 - flashinfer.jit: [Autotuner]: Autotuning process starts ...
line 219:[AutoTuner]: Tuning mxfp8_gemm: 0/23 ... → 100%|23/23| (约 11 分钟)
line 274:gsm8k flexible-extract exact_match = 0.9522 ; strict-match = 0.9530
```

---

## 2. 两条量化路径为什么分流到不同后端

| 量化 | 作用的层 | 层类型 | 选中的 kernel | 后端 | 有 FlashInfer autotune？ |
|---|---|---|---|---|---|
| **MXFP8** | 共享专家 `shared_experts.*` | **Linear** | `FlashInferCutedslMxfp8LinearKernel` | **FlashInfer** | ✅ 是（`mm_mxfp8`） |
| **MXFP4** | 路由专家 `experts.w1/w2/w3` | **MoE** | `CutlassExpertsMxfp4` | **Cutlass** | ❌ 否 |

要点：
- 方案① 里，**共享专家（`block_sparse_moe.shared_experts`）是普通 Linear 层**，被量化为
  MXFP8 → 走 vLLM 的 MXFP8 linear kernel 选择器（`kernels/linear/__init__.py`）→ 在 SM100
  选中 `FlashInferCutedslMxfp8LinearKernel`（`mxfp8/flashinfer.py`，内部调
  `flashinfer.mm_mxfp8`）。FlashInfer 的 GEMM 在首次运行前由 `flashinfer.jit` autotuner
  对候选 tactic 做 profiling（日志里的 23 个 profile）。
- **路由专家（`block_sparse_moe.experts`）是 MoE 融合层**，量化为 MXFP4 →
  `compressed_tensors_moe_w4a4_mxfp4.py` 里
  `use_cutlass_mxfp4 = CutlassExpertsMxfp4._supports_current_device()`，在 B200/SM100 **恒 True**
  → 选 `CutlassExpertsMxfp4`（`fused_moe/experts/cutlass_moe.py::run_cutlass_moe_fp4`）。
  Cutlass 用自带的 GEMM/heuristic，**不接入 FlashInfer autotuner**，所以日志里不会出现
  "Tuning mxfp4_..."。

---

## 3. autotuner 的归属

- `[AutoTuner]: Tuning ...` 与 `autotuner.py:651 flashinfer.jit: [Autotuner]` 均来自
  **FlashInfer 包**（`/usr/local/lib/python3.12/dist-packages/flashinfer/autotuner.py`）。
- 它只对 **FlashInfer kernel** 生效（`mm_mxfp8` / `mm_fp4` / attention 的 trtllm-gen 等）。
- vLLM 侧 `enable_flashinfer_autotune=True`（见日志 `KernelConfig`），触发这些 FlashInfer
  kernel 在 warmup 阶段 autotune。
- **Cutlass / Marlin / Triton 后端有各自的选核逻辑，不走 FlashInfer autotuner**，因此不会在
  这段日志里出现。

---

## 4. "之前纯 MXFP4 也会 tune" 的真正解释

那次被 tune 的**不是 MXFP4 MoE 本身**（它一直走 Cutlass、从不进 FlashInfer autotuner），
而是**其它同样走 FlashInfer 的组件**，最可能是：

1. **FlashInfer 的 attention**（TRTLLM attention，本日志 line 203 也有
   `Using TRTLLM attention`）——它也由 FlashInfer autotuner tune；或
2. 那次 **MoE 恰好走了 FlashInfer 路径**（如 trtllm-gen MXFP4 MoE）而非 Cutlass，
   于是出现 MXFP4 相关的 FlashInfer tune。

换言之：**"tune 什么" 取决于最终选中的 kernel 是不是 FlashInfer 的，与量化位宽 fp4/fp8
无必然关系。** 本次方案① 新增的 MXFP8（共享专家）恰好走 FlashInfer linear kernel，
才第一次出现 `Tuning mxfp8_gemm`；而 MXFP4 MoE 走 Cutlass，一如既往不在此列。

---

## 5. 顺带观察（非问题）

- **首次加载 tune 慢**：`init engine ... took 734 s`，其中 `Tuning mxfp8_gemm` 约 11 分钟
  （23 个 profile，单个 profile 首个耗时 ~195 s，因含 CuTe DSL JIT 编译）。属一次性 warmup
  开销；FlashInfer autotune 结果会缓存，后续启动更快。
- 期间反复出现 `shm_broadcast.py:705 No available shared memory broadcast block found in 60s`
  —— 是 autotune/JIT 编译耗时较长导致的正常提示，非错误。
- 推理阶段有两处 Triton JIT 提示（`_compute_slot_mapping_kernel`、`_topk_index_kernel`），
  属首个 shape 的 JIT，`jit_monitor` mode=warn 只告警不报错。

---

## 6. 一句话总结

> mxfp4 不是"不需要 tune 了"，而是它走 **Cutlass** 后端，本就不归 **FlashInfer autotuner**
> 管；方案① 多出来的 **mxfp8（共享专家 Linear）** 恰好走 FlashInfer，才第一次出现
> `Tuning mxfp8_gemm`。模型精度正常（gsm8k ≈ 95.2%）。

---

## 7. 相关文件

- 日志：`/storage/lkk/m3/test_11.log`
- 混合精度方案文档：`/storage/lkk/m3/minimax_m3_mixed_precision_quant.md`
- vLLM 库：`/storage/lkk/xpu_vllm/vllm/vllm`
  - MXFP8 linear kernel 选择：`model_executor/kernels/linear/__init__.py`（:758 日志行）
  - MXFP8 FlashInfer kernel：`model_executor/kernels/linear/mxfp8/flashinfer.py`（`mm_mxfp8`）
  - MXFP4 MoE 选择：`model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_mxfp4.py`（:53 日志行）
  - Cutlass MoE：`model_executor/layers/fused_moe/experts/cutlass_moe.py`（`run_cutlass_moe_fp4`）
- FlashInfer autotuner：`/usr/local/lib/python3.12/dist-packages/flashinfer/autotuner.py`

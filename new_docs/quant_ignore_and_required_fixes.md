# MiniMax-M3 MXFP4：量化 ignore 层选择 × vLLM 已做修改 全景对照

> 本文回答一个核心问题：
> **如果我在量化命令里也 `ignore` 掉 `mlp.gate_proj / mlp.up_proj / mlp.down_proj`，
> 用现在这套 vLLM 代码，模型还能不能跑？**
>
> 结论先说：**能跑，不需要再改 vLLM 代码。**
> 下文给出完整的「报错 → 根因 → 已做修改 → 是否受 ignore-mlp 影响」对照，以及新模型评估。

---

## 0. 一句话结论

- 加 `mlp.gate_proj / mlp.up_proj / mlp.down_proj` 到 ignore，只影响 **dense 层的 MLP**
  （非 MoE 层，如 layer 0~2），让它们从「量化」变成「不量化」。
- 这**不会触发报错 3**（反而是根治报错 3 的正解），也**不影响**报错 1/2/4/5/6/7 的修复。
- 现有 vLLM 代码修改 + flashinfer 0.6.13 **全部照常生效**。
- 因此：**只加这三条 dense-mlp ignore，用现在的 vLLM 代码就能跑，零新增代码改动。**

---

## 1. 两份量化命令的 ignore/exclude 对照

| 目标 | 我们（auto-round `--ignore_layers`） | AMD（quark `exclude_layers`） |
|---|---|---|
| lm_head | `lm_head` | `*lm_head` |
| 视觉塔 | `visual` ⚠️（模型里实际叫 `vision_tower`，**匹配不到**） | `*vision_tower*` ✅ |
| 多模态投影 | ❌ 无 | `*multi_modal_projector*` |
| patch merge | ❌ 无 | `*patch_merge_mlp*` |
| MoE 路由器 | `block_sparse_moe.gate` ✅ | `*block_sparse_moe.gate` ✅ |
| 注意力 | `self_attn` | `*self_attn*` |
| embedding | `embed_tokens` | ❌（未排，但 embed 一般本就不量化） |
| **dense MLP** | ❌ **无** | `*mlp.gate_proj` `*mlp.up_proj` `*mlp.down_proj` ✅ |
| linear_attn | `linear_attn` ⚠️（模型里不存在此名） | ❌ 无 |
| shared_expert_gate | `shared_expert_gate` ⚠️（模型里不存在此名） | ❌ 无 |

**核心差异**：AMD 排除了 dense 层的 `mlp.gate_proj/up_proj/down_proj`，我们**没有**。
→ 我们的模型里这几层 dense MLP 被量化了，AMD 没量化。这正是报错 3 的根源。

> 注：共享专家 `block_sparse_moe.shared_experts.gate_proj/up_proj/down_proj` 两边都**照常量化**
> （名字里没有连续子串 `mlp.gate_proj`，glob/子串都匹配不到），行为一致，无需担心。

---

## 2. 报错 → 修改 全表（7 个报错）

| # | 报错 | 根因归类 | 已做的修改（运行时库 `/storage/lkk/xpu_vllm/vllm`） | 受 ignore-mlp 影响？ |
|---|---|---|---|---|
| 1 | vision `fc1` KeyError | **vLLM bug**：视觉塔 `mlp.fc1`→`fc1` 重命名，ignore 匹配失效 | `nvidia/model.py`：`hf_to_vllm_mapper` 去结尾点 | ❌ 无关 |
| 2 | vision `qkv_proj` KeyError | **vLLM bug**：缺 `packed_modules_mapping`，融合层 ignore 失效 | `nvidia/model.py`：补 `packed_modules_mapping` + 对齐 amd 版 | ❌ 无关 |
| 3 | `gate_up_proj` scheme 不一致 | **量化产物**：老命令 `mlp.gate` 误伤 dense `mlp.gate_proj`，一半量化一半没量化 | 无代码；靠重量化（改 ignore） | ✅ **正是这个** |
| 4 | `SWIGLUOAI requires clamp_limit` | **vLLM 缺陷**：cutlass MXFP4 MoE 未传 clamp | `config.py` + `compressed_tensors_moe_w4a4_mxfp4.py` + `cutlass_moe.py` 注入 clamp | ❌ 无关（MoE experts，非 dense） |
| 5 | `No common block size for 16` | **vLLM 缺陷**：稀疏注意力要 block=128 | `models/config.py`：`MiniMaxM3Config` 钩子钉 block_size=128 | ❌ 无关 |
| 6 | `allreduce_fusion ... weight_bias` | **flashinfer 版本旧**无此参数 | `allreduce_rms_fusion.py` 折叠偏置 + 升级 flashinfer 0.6.13 | ❌ 无关 |
| 7 | `Missing TRTLLM-GEN kernel (decode)` | **flashinfer-cubin 旧**缺 page-128 decode 内核 | 升级 flashinfer 0.6.13（含 cubin） | ❌ 无关 |

**读表要点**：报错 1/2/4/5/6/7 **全部与「是否 ignore mlp」无关**。它们的修改已完成、flashinfer 已升级，
换成 ignore-mlp 的新模型后这 6 处修改**照样必需、照样生效**。

---

## 3. 为什么 ignore-mlp 反而让报错 3 消失

报错 3 的本质是**融合分片量化方案不一致**：

```
ValueError: Found a different quantization schemes for ['gate_proj', 'up_proj']
in language_model.model.layers.0.mlp.gate_up_proj. vLLM requires all to use the same scheme.
```

vLLM 把 dense 层的 `gate_proj + up_proj` 融合成 `gate_up_proj`（`MergedColumnParallelLinear`），
**要求两个分片量化方案一致**。

| 情况 | gate_proj | up_proj | 融合检查 |
|---|---|---|---|
| 老命令（`mlp.gate` 误伤 gate_proj） | **未量化** | 已量化 | ❌ 不一致 → 报错 3 |
| 现在的模型（全不 ignore mlp） | 已量化 | 已量化 | ✅ 一致 → 过 |
| **加 ignore mlp.gate_proj/up_proj/down_proj** | **未量化** | **未量化** | ✅ 一致 → 过 |

只要 gate_proj 和 up_proj **要么都量化、要么都不量化**，融合检查就通过。
「全部 ignore」和「全部不 ignore」都满足；只有「一半一半」才炸。所以加这三条是**安全**的。

---

## 4. 新模型（ignore-mlp 版）评估

相比现在的模型，ignore-mlp 新模型有三处**从量化变成不量化**：

| 模块 | 风险 | 说明 |
|---|---|---|
| dense `mlp.gate_proj/up_proj/down_proj` | **零风险** | 语言模型普通融合层，不量化是 vLLM 安全默认，走 FP16，不需任何量化 kernel |
| `multi_modal_projector` | 低（已修机制） | vLLM 会重命名加前缀 `vision_tower.multi_modal_projector.`（`model.py:1040`），走的是报错 1/2 那套「重命名 + ignore 匹配」，你已修好 |
| `patch_merge_mlp` | 低（已修机制） | 同上，重命名成 `vision_tower.patch_merge_mlp.`（`model.py:1041`） |

**唯一要盯的一点**：启动加载阶段，若出现 `vision_tower.multi_modal_projector.*` 或
`vision_tower.patch_merge_mlp.*` 的 **KeyError**，那和报错 1/2 是**同一个病根**
（config.json 的 ignore 名与 vLLM 内部名没对上），修复模式完全一样、几分钟小改。
但因为你已修好这套机制（去结尾点 + packed_modules + 对齐 amd 版），**大概率会自动正确匹配**。

> 特别地：**如果你只加 `mlp.gate_proj/up_proj/down_proj` 这三条、其余 ignore 不动**
>（即不动 `visual` / 不加 projector/patch_merge），那就是纯粹让 dense MLP 不量化，
> 现有代码稳跑，**无任何新增风险**。

---

## 5. 实操建议

### 方案 A（最小改动，最稳）：只加 dense mlp 三条
在现有 `--ignore_layers` 后追加 `mlp.gate_proj,mlp.up_proj,mlp.down_proj`，其余不动：
```
auto-round ../MiniMax-M3/ --model_free --scheme MXFP4 \
  --ignore_layers visual,lm_head,block_sparse_moe.gate,linear_attn,shared_expert_gate,embed_tokens,self_attn,mlp.gate_proj,mlp.up_proj,mlp.down_proj \
  --format llm_compressor --output_dir "./MiniMax-M3-MXFP4"
```
→ dense MLP 不量化，报错 3 根治，其余修改全部生效，**现有 vLLM 代码直接跑**。

### 方案 B（完全对齐 AMD，最干净）：同时修正视觉命名
```
--ignore_layers vision_tower,multi_modal_projector,patch_merge_mlp,lm_head,\
block_sparse_moe.gate,self_attn,embed_tokens,mlp.gate_proj,mlp.up_proj,mlp.down_proj
```
（把 `visual`→`vision_tower`、补 `multi_modal_projector`/`patch_merge_mlp`、去掉模型里不存在的
`linear_attn`/`shared_expert_gate`。）
→ 更规范，但让 projector/patch_merge 变未量化，需按 §4 盯一下加载（有你已修的机制兜底）。

**两种方案都不需要再改 vLLM 代码。** 保留现有全部修改 + flashinfer 0.6.13，直接 `eval_vllm.sh`。

---

## 6. 相关文档索引

- `vllm_vision_quant_bug_analysis.md` —— 报错 1（vision fc1 重命名）
- `vllm_qkv_fix_and_amd_comparison.md` —— 报错 2（qkv 融合层 + packed_modules_mapping）
- `vllm_gate_up_proj_quant_mismatch.md` —— 报错 3（gate_up_proj scheme 不一致，本文根治对象）
- `vllm_cutlass_mxfp4_clamp_fix.md` —— 报错 4（cutlass MXFP4 clamp）
- `vllm_block_size_sparse_attention_fix.md` —— 报错 5（block_size 128）
- `vllm_allreduce_weight_bias_fix.md` —— 报错 6/7（weight_bias + flashinfer 升级）
- `vllm_marlin_vs_cutlass_mxfp4_moe.md` —— Marlin vs cutlass MoE 内核路径

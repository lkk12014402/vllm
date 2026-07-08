# MiniMax-M3 混合精度量化（MoE→MXFP4 / 其余→MXFP8）：ignore 验证 + 命令

> 任务来源：`task_5.md`。两个诉求：
> ① 再次确认 model-free 量化的 `--ignore_layers` 配置是否正确（对比 AMD-Quark 版）；
> ② 仿照 DeepSeek 混合精度示例（MoE 量化为 MXFP4、其余为 MXFP8），改写 MiniMax-M3 命令。
>
> 全部结论均用 auto-round 真实匹配器 + 原始 checkpoint 权重名**逐层模拟验证**。

---

## 0. 原始命令（当前 model-free MXFP4）

```bash
auto-round ../MiniMax-M3/ --model_free --scheme MXFP4 \
  --ignore_layers vision_tower,lm_head,block_sparse_moe.gate,embed_tokens,self_attn,\
patch_merge_mlp,multi_modal_projector,mlp.gate_proj,mlp.up_proj,mlp.down_proj \
  --format llm_compressor --output_dir "./MiniMax-M3-MXFP4-ignore-mlp-all"
```

---

## 1. MiniMax-M3 结构（原始 checkpoint 的 2D Linear 层）

| 模块（归一化） | 类型 | 数量 |
|---|---|---|
| `language_model.lm_head` | Linear | 1 |
| `language_model.model.embed_tokens` | Embedding(2D) | 1 |
| `...layers.N.block_sparse_moe.experts.N.w1/w2/w3` | MoE 路由专家 Linear | 7296×3 |
| `...layers.N.block_sparse_moe.gate` | 路由器 Linear | 57 |
| `...layers.N.block_sparse_moe.e_score_correction_bias` | 偏置(1D) | 57 |
| `...layers.N.block_sparse_moe.shared_experts.gate_proj/up_proj/down_proj` | 共享专家 Linear | 57×3 |
| `...layers.N.mlp.gate_proj/up_proj/down_proj` | **dense** MLP Linear（前 3 层） | 3×3 |
| `...layers.N.self_attn.{q,k,v,o}_proj` | 注意力 Linear | 60×4 |
| `...layers.N.self_attn.index_{q,k}_proj` | 稀疏注意力索引器 Linear | 57×2 |
| `...layers.N.self_attn.{q,k}_norm / index_{q,k}_norm` | RMSNorm(1D) | — |
| `...layers.N.input_layernorm / post_attention_layernorm / norm` | RMSNorm(1D) | — |
| `multi_modal_projector.linear_1/2` | Linear | 2 |
| `patch_merge_mlp.linear_1/2` | Linear | 2 |
| `vision_tower.vision_model...(self_attn / mlp.fc1/fc2 / embeddings)` | 视觉塔 Linear | 多 |

要点：
- MoE 专家命名是 **`w1/w2/w3`**（原始 repo schema），不是 `gate/up/down_proj`。
  （w1=gate、w3=up、w2=down，Mixtral 惯例。）
- dense MLP 只在前 3 层（`layers.0/1/2.mlp.*`），其余层是 MoE。
- 注意力含一套**稀疏索引器** `index_q_proj/index_k_proj`（类似 DeepSeek 的 indexer）。

### 1.1 参数量分布（总 427.0 B，实测自 safetensors header）

| 结构 | 参数量 | 占比 |
|---|---:|---:|
| **MoE 路由专家** `block_sparse_moe.experts.w1/w2/w3` | **413.1 B** | **96.74%** |
| 注意力主体 `self_attn.{q,k,v,o}_proj` | 6.42 B | 1.50% |
| MoE 共享专家 `shared_experts.gate/up/down_proj` | 3.23 B | 0.76% |
| lm_head | 1.23 B | 0.29% |
| embed_tokens | 1.23 B | 0.29% |
| dense MLP（前 3 层） | 0.68 B | 0.16% |
| vision_tower | 0.63 B | 0.15% |
| 注意力索引器 `self_attn.index_*` | 0.22 B | 0.05% |
| patch_merge_mlp / multi_modal_projector / router gate / norms | <0.3 B 合计 | <0.1% |

**路由专家一类就占 96.74%**——这是 MoE 结构决定的（256 专家 × 57 层，前向稀疏激活）。
这也解释了为何 model-free/方案① 压缩比高：只要把这 96.7% 压到 4bit，整体就逼近 4×，
其余结构无论保 bf16 还是压 MXFP8，对总体积影响都很小（详见 §4b）。

---

## 2. ignore 匹配验证：我们 vs AMD-Quark（逐层对照）

### 2.1 两份 ignore/exclude 原文

**我们（auto-round，子串/正则）**：
```
vision_tower, lm_head, block_sparse_moe.gate, embed_tokens, self_attn,
patch_merge_mlp, multi_modal_projector, mlp.gate_proj, mlp.up_proj, mlp.down_proj
```

**AMD-Quark（fnmatch glob）**：
```
*lm_head, *vision_tower*, *multi_modal_projector*, *patch_merge_mlp*,
*block_sparse_moe.gate, *self_attn*, *mlp.gate_proj, *mlp.up_proj, *mlp.down_proj
```

### 2.2 逐层结果（模拟）

| 模块 | 我们 | AMD | 一致 |
|---|---|---|---|
| `block_sparse_moe.experts.w1/w2/w3` | **量化** | **量化** | ✓ |
| `block_sparse_moe.shared_experts.gate/up/down_proj` | **量化** | **量化** | ✓ |
| `block_sparse_moe.gate`（路由器） | skip | excl | ✓ |
| dense `mlp.gate_proj/up_proj/down_proj` | ignore | excl | ✓ |
| `self_attn.{q,k,v,o}_proj` | ignore | excl | ✓ |
| `self_attn.index_{q,k}_proj`（索引器） | ignore | excl | ✓ |
| `lm_head` | ignore | excl | ✓ |
| `multi_modal_projector.*` / `patch_merge_mlp.*` | ignore | excl | ✓ |
| `vision_tower.*` | ignore | excl | ✓ |
| `embed_tokens` | skip | （类型跳过） | ✓* |
| 各 `*norm` / `input_layernorm` 等 | 类型跳过(1D) | 类型跳过(1D) | ✓* |

**结论：对所有 Linear 层，我们的 ignore 与 AMD 版完全一致 ✓**

`*` 说明：`embed_tokens`（Embedding）与各 `norm`（1D RMSNorm）两边都靠**层类型**规则
（只量化 2D Linear）自动跳过，不是名字规则差异，故无实质区别。

### 2.3 机制差异（结果相同）

- **我们**：`to_standard_regex` 子串匹配（`self_attn` → `.*self_attn.*`）。`gate`/`embed`
  由 auto-round 预定义 `_BLOCK_NAME_TO_IGNORE=["shared_expert_gate.", ".gate.", "embed", "conv"]`
  **自动跳过**。
- **AMD**：`fnmatch` glob（`*self_attn*`、`*mlp.gate_proj`）。

**冗余项提醒**：你命令里的 `block_sparse_moe.gate`、`embed_tokens` 是冗余（已被自动跳过），
留着无害。

**关键正确性**：`mlp.gate_proj` 只命中 **dense** MLP，**不误伤**
`block_sparse_moe.shared_experts.gate_proj`（子串 `mlp.gate_proj` 不连续出现），
两边一致 —— 所以共享专家在两版里都被量化。

> 综上，第 ① 问答复：**当前 ignore 配置正确，且与 AMD-Quark 版等价。**

---

## 3. 混合精度：DeepSeek 范式 → MiniMax-M3

### 3.1 参考示例（DeepSeek）

```bash
auto-round deepseek-ai/DeepSeek-V4-Pro --model_free \
  --scheme MXFP8 \
  --ignore_layers compressor,indexer.weights_proj \
  --layer_config "{ffn.experts:{bits:4,data_type:mx_fp}}" \
  --format llm_compressor --output_dir "./DeepSeek-V4-Pro-MXFP4-Mixed"
```

范式：**全局 MXFP8 + `--layer_config` 把专家覆盖为 MXFP4（bits:4, data_type:mx_fp）+ 极少 ignore**。

### 3.2 语法验证（已确认）

- `--layer_config` 在 **model-free 路径生效**（`_PatternMatcher.resolve_scheme` 会读它）。
- DeepSeek 那种无引号嵌套 JSON 能被 `parse_layer_config_arg` 正确解析：
  `{block_sparse_moe.experts:{bits:4,data_type:mx_fp}}` → `{'block_sparse_moe.experts': {'bits':4,'data_type':'mx_fp'}}`
- scheme 定义：`MXFP4 = bits4/mx_fp/group_size32`；`MXFP8 = bits8/mx_fp/group_size32`。
- layer_config key `block_sparse_moe.experts` **精确匹配路由专家、不误伤 shared_experts**
  （`block_sparse_moe.experts` 不是 `block_sparse_moe.shared_experts` 的连续子串）。
- auto-round model-free **原生支持混合导出**：顶层 `format="mixed-precision"`，
  生成多个 config_groups，每组带独立 format（`mxfp4-pack-quantized` / `mxfp8-quantized`）。

---

## 4. 两套命令（均已逐层模拟验证）

### 方案①（保守 · 推荐）— 沿用已验证 ignore，只把共享专家提到 MXFP8

```bash
auto-round ../MiniMax-M3/ --model_free \
  --scheme MXFP8 \
  --ignore_layers vision_tower,lm_head,block_sparse_moe.gate,embed_tokens,self_attn,\
patch_merge_mlp,multi_modal_projector,mlp.gate_proj,mlp.up_proj,mlp.down_proj \
  --layer_config "{block_sparse_moe.experts:{bits:4,data_type:mx_fp}}" \
  --format llm_compressor \
  --output_dir "./MiniMax-M3-MXFP4MoE-MXFP8-conservative"
```

模拟结果：

| 层 | 精度 |
|---|---|
| 路由专家 `block_sparse_moe.experts.w1/w2/w3` | **MXFP4** |
| 共享专家 `shared_experts.gate/up/down_proj` | **MXFP8** |
| dense `mlp.*` / `self_attn.*` / vision / projector / lm_head | **bf16**（不变） |
| router gate / embed | skip |

特点：直接针对精度诉求（共享专家 MXFP4→MXFP8 更少损失），attention/dense 保持 bf16
（比 MXFP8 更高精度），沿用已知可跑的 ignore，**vLLM 风险最低**。

### 方案②（激进 · 忠实照搬 DeepSeek）— attention+dense 也压到 MXFP8

```bash
auto-round ../MiniMax-M3/ --model_free \
  --scheme MXFP8 \
  --ignore_layers vision_tower,lm_head,multi_modal_projector,patch_merge_mlp \
  --layer_config "{block_sparse_moe.experts:{bits:4,data_type:mx_fp}}" \
  --format llm_compressor \
  --output_dir "./MiniMax-M3-MXFP4MoE-MXFP8-aggressive"
```

模拟结果：

| 层 | 精度 |
|---|---|
| 路由专家 | **MXFP4** |
| 共享专家 / dense `mlp.*` / `self_attn.{q,k,v,o}_proj` / `self_attn.index_*` | **MXFP8** |
| `self_attn.*_norm`（1D） / vision / projector / lm_head | bf16 |
| router gate / embed | skip |

特点：体积最小；但 **attention 走 MXFP8 + mixed-precision 格式在 vLLM minimax_m3 路径未验证**。

### 方案③（方案①的进阶）— 在方案①基础上把主注意力 q/k/v/o 也压到 MXFP8，索引器 index_* 仍保 bf16

```bash
auto-round ../MiniMax-M3/ --model_free \
  --scheme MXFP8 \
  --ignore_layers vision_tower,lm_head,block_sparse_moe.gate,embed_tokens,\
patch_merge_mlp,multi_modal_projector,mlp.gate_proj,mlp.up_proj,mlp.down_proj,\
self_attn.index_q_proj,self_attn.index_k_proj,self_attn.index_q_norm,self_attn.index_k_norm,\
self_attn.q_norm,self_attn.k_norm \
  --layer_config "{block_sparse_moe.experts:{bits:4,data_type:mx_fp}}" \
  --format llm_compressor \
  --output_dir "./MiniMax-M3-MXFP4MoE-MXFP8-attn"
```

与方案①的唯一区别：**把笼统的 `self_attn` 一项，拆成精确 ignore
`self_attn.index_{q,k}_proj / index_{q,k}_norm / {q,k}_norm`** —— 这样主注意力
`q/k/v/o_proj` 落回全局 MXFP8，而稀疏索引器 `index_*` 与所有注意力 norm 仍保 bf16。

模拟结果（已验证精确区分）：

| 层 | 精度 |
|---|---|
| 路由专家 `block_sparse_moe.experts.w1/w2/w3` | **MXFP4** |
| 共享专家 `shared_experts.*` | **MXFP8** |
| 主注意力 `self_attn.{q,k,v,o}_proj` | **MXFP8** |
| 稀疏索引器 `self_attn.index_{q,k}_proj` | **bf16**（ignore） |
| 注意力 norm `self_attn.{q,k}_norm / index_*_norm` | bf16 |
| dense `mlp.*` / vision / projector / lm_head | bf16 |
| router gate / embed | skip |

> 匹配正确性：`self_attn.q_proj`（量化）与 `self_attn.index_q_proj`（ignore）虽有子串关系，
> 但 ignore 判定**优先于** scheme 匹配，且 index_q_proj 被显式 ignore 命中，故二者精确区分、
> 不会相互误伤。索引器是稀疏注意力的敏感组件，保 bf16 更稳。

---

## 4b. 各方案压缩比对照

| 方案 | 路由专家 | 共享专家 | 主注意力 q/k/v/o | 索引器 index_* | dense/vision/其余 | 体积 | 压缩比 | 平均位宽 |
|---|---|---|---|---|---|---|---|---|
| 原始 bf16 | 16 | 16 | 16 | 16 | 16 | 854 GB | 1.00× | 16.0 bit |
| 纯 model-free MXFP4 | MXFP4 | MXFP4 | bf16 | bf16 | bf16 | 243 GB | 3.52× | 4.54 bit |
| **方案①（推荐）** | MXFP4 | MXFP8 | bf16 | bf16 | bf16 | 244 GB | 3.50× | 4.57 bit |
| **方案③** | MXFP4 | MXFP8 | **MXFP8** | bf16 | bf16 | 238 GB | 3.59× | 4.46 bit |
| 方案②（激进） | MXFP4 | MXFP8 | MXFP8 | MXFP8 | dense+attn MXFP8 | 237 GB | 3.60× | 4.44 bit |

> MXFP4/MXFP8 均含 E8M0 per-block scale 开销：group_size=32 → +0.25 bit/参数（4→4.25、8→8.25）。

**关键结论：整体压缩比几乎完全由占 96.74% 的路由专家决定。**
- 把这 96.74% 从 16bit→MXFP4，整体就逼近 4×（含 scale 开销实测 3.5×）。
- 其余 3.3% 的层（attention/shared/dense/vision）无论 bf16 还是 MXFP8，对整体体积影响都极小：
  方案①→方案③（attention 也压 MXFP8）仅 244→238 GB（约 -2.5%）；方案①→方案②仅差 7 GB。
- 因此**方案①/③ 的取舍逻辑**：用几乎可忽略的体积代价（把 3.3% 敏感层留在高精度），换取精度稳健。
  想再抠一点体积就用方案③（多量化 1.5% 的主注意力），但收益有限、风险略增。

---

## 5. 重要提醒

1. **混合会产出 `format="mixed-precision"`**（两个 config_groups：mxfp4 + mxfp8）。
   auto-round 导出没问题；但 **vLLM 的 minimax_m3 + compressed-tensors 能否加载 mixed-precision
   需实测** —— 之前调试全是单一 MXFP4，方案②的 attention-MXFP8 更未验证。

2. **MXFP8 维度约束**：`group_size=32`，被量化层的 `in_features` 需能被 32 整除
   （dense/attention 一般满足；导出报维度错即是此因）。

3. **推荐路径**：先跑**方案①**（精度改进明确、风险低）→ vLLM 跑通验证 mixed-precision
   加载链路 → 如需进一步压缩，优先**方案③**（在方案①上多量化主注意力 q/k/v/o 到 MXFP8，
   索引器保 bf16，体积 244→238 GB），再考虑更激进的方案②。

---

## 6. 附录：核对命令

```bash
# 结构：列出原始 checkpoint 的所有权重名模式
python3 -c "import json,glob,re;wm=json.load(open(glob.glob('/storage/lkk/MiniMax-M3/*.index.json')[0]))['weight_map'];\
import collections;c=collections.Counter(re.sub(r'\.\d+\.','.N.',k) for k in wm);\
[print(v,p) for p,v in sorted(c.items())]"

# layer_config 解析自检
python3 -c "import sys;sys.path.insert(0,'/storage/lkk/m3/auto-round');\
from auto_round.utils.common import parse_layer_config_arg;\
print(parse_layer_config_arg('{block_sparse_moe.experts:{bits:4,data_type:mx_fp}}'))"
```

## 7. 相关文件

- 本文档：`/storage/lkk/m3/minimax_m3_mixed_precision_quant.md`
- 任务原文：`/storage/lkk/m3/task_5.md`
- 配套分析：`auto_round_model_free_vs_rtn_analysis.md`、`auto_round_native_to_original_converter.md`
- 参照产物：`/storage/lkk/m3/MiniMax-M3-MXFP4-ignore-mlp-all`（model-free）、`/storage/lkk/MiniMax-M3-amd`（AMD-Quark）
- 原始模型：`/storage/lkk/MiniMax-M3`

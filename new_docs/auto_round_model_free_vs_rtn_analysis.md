# auto-round 两条量化路径对比分析：model-free vs 非 model-free(RTN)

> 结论先行：在 **transformers 5.12.0** 环境下，auto-round 的 **model-free** 路径产出的
> MXFP4 模型可以被 vLLM（`models/minimax_m3`）正确加载并推理；而 **非 model-free（RTN，
> `--iters 0`）** 路径产出的模型，因为经过 transformers **原生 `minimax_m3_vl`** 实现的
> `from_pretrained → save_pretrained` 往返，被改写成了一套**与 vLLM 加载器不兼容的
> canonical schema**（命名、config 字段、激活函数、层融合全变），导致：
> 加载报错（`layer_types` 等字段），手动删字段后能加载但**精度为 0**。
>
> **建议：MiniMax-M3 → vLLM 的 MXFP4 量化，请使用 model-free 路径。**

---

## 1. 背景：两条量化命令

来自 `/storage/lkk/m3/quant.md`，两条命令**只差两个开关**，其余（scheme、ignore_layers、
format）完全相同：

```bash
# (A) model-free —— vLLM 能跑
auto-round ../MiniMax-M3/ --model_free --scheme MXFP4 \
  --ignore_layers vision_tower,lm_head,block_sparse_moe.gate,embed_tokens,self_attn,\
patch_merge_mlp,multi_modal_projector,mlp.gate_proj,mlp.up_proj,mlp.down_proj \
  --format llm_compressor --output_dir "./MiniMax-M3-MXFP4-ignore-mlp-all"

# (B) 非 model-free / RTN —— vLLM 报错 / 精度 0
auto-round ../MiniMax-M3/ --scheme MXFP4 --iters 0 \
  --ignore_layers vision_tower,lm_head,block_sparse_moe.gate,embed_tokens,self_attn,\
patch_merge_mlp,multi_modal_projector,mlp.gate_proj,mlp.up_proj,mlp.down_proj \
  --format llm_compressor --output_dir "./MiniMax-M3-MXFP4-ignore-mlp-all-rtn"
```

产出模型：
- (A) `/storage/lkk/m3/MiniMax-M3-MXFP4-ignore-mlp-all/`
- (B) `/storage/lkk/m3/MiniMax-M3-MXFP4-ignore-mlp-all-rtn/MiniMax-M3-mxfp-w4g32/`

两者都用 `--format llm_compressor`（导出为 compressed-tensors `mxfp4-pack-quantized`）。

---

## 2. 根本原因：transformers 5.12.0 内置了原生 `minimax_m3_vl`

关键事实：

```text
transformers.__version__ = 5.12.0
transformers/models 下存在原生实现: minimax_m3_vl, minimax_m2, minimax
```

原始 MiniMax-M3 仓库是 **remote-code 模型**：

```json
// /storage/lkk/MiniMax-M3/config.json
"architectures": ["MiniMaxM3SparseForConditionalGeneration"],
"auto_map": { "AutoConfig": "configuration_minimax_m3_vl.MiniMaxM3VLConfig" },
"text_config": { "architectures": ["MiniMaxM3SparseForCausalLM"],
                 "hidden_act": "swigluoai",
                 "sparse_attention_config": {...},  // 原始字段
                 "rope_theta": ... }
```

两条路径对模型的**处理方式不同**，这是一切差异的源头：

### (A) model-free —— 不建模型，直接改权重张量
`auto_round/compressors/model_free.py` 的 `ModelFreeCompressor` **不调用 `from_pretrained`**、
不实例化 `nn.Module`。它直接对 safetensors 里的**权重张量按名字**做 MXFP4 量化/打包，
然后把原始 `config.json` 基本**原样透传**（仅注入 `quantization_config`）。

→ 输出保留原始 repo 的**命名与 config schema**（`block_sparse_moe.experts.w1/w2/w3`、
`swigluoai`、`sparse_attention_config`、`language_model.model.layers.*`）。
**这正是 vLLM `models/minimax_m3` 加载器所期望的 schema。**

### (B) 非 model-free / RTN —— 真正建模型再保存
非 model-free 路径（`auto_round/compressors/entry.py`）会 `from_pretrained` 把模型
**真正实例化**出来做 RTN（`--iters 0`，round-to-nearest，无优化迭代）量化，然后
`save_pretrained` 写回。

在 transformers 5.12.0 下，`from_pretrained` 会把它解析成**原生 `minimax_m3_vl`
实现**（不再走 repo 里的 remote-code modeling 文件）。原生实现使用 HuggingFace 的
**canonical schema**，`save_pretrained` 就把这套「重命名后的」结构和 config 写了出去。

---

## 3. 两个模型的具体差异（实测）

### 3.1 config.json 顶层键

| | model-free | rtn |
|---|---|---|
| 仅一方有 | `torch_dtype` | `dtype`, `merged_hidden_size`, `tie_word_embeddings` |

`dtype`（新版 transformers 命名，取代 `torch_dtype`）与 `merged_hidden_size` 是
**经过 transformers 模型往返**的直接证据；`merged_hidden_size` 更直接暗示了 dense MLP 的
gate/up 被**融合**。

### 3.2 `text_config` —— 差异巨大（语义级）

| 字段 | model-free（原始） | rtn（原生 transformers） |
|---|---|---|
| `hidden_act` | **`swigluoai`** | **`silu`** |
| 稀疏注意力 | `sparse_attention_config` | `layer_types` + `mlp_layer_types` + `index_block_size/index_head_dim/index_local_blocks/index_n_heads/index_topk_blocks` |
| RoPE | `rope_theta` | `rope_parameters` |
| 其他 rtn 新增 | — | `attention_dropout, bos/eos/pad_token_id, initializer_range, model_type, output_router_logits, router_aux_loss_coef, router_jitter_noise, use_cache` |
| model-free 独有 | `moe_layer_freq`, `sparse_attention_config` | — |

`layer_types`（长度 60，样例 `['full_attention','full_attention','full_attention',
'minimax_m3_sparse','minimax_m3_sparse','minimax_m3_sparse', ...]`）就是你在 rtn 的
config.json 里看到的「莫名其妙字段」——它来自原生 transformers 的 config 约定，
**vLLM 的 minimax_m3 config 解析不认这套**，因此加载报错。

### 3.3 `vision_config`

| 字段 | model-free | rtn |
|---|---|---|
| `model_type` | `clip_vision_model` | `minimax_m3_vl_vision` |
| rtn 新增 | — | `spatial_merge_size, temporal_patch_size, rope_parameters, dtype` |

### 3.4 权重张量命名 —— 完全不同的 schema

| 部件 | model-free（vLLM 期望） | rtn |
|---|---|---|
| MoE 专家 | `...block_sparse_moe.experts.*.w1/w2/w3` | `...mlp.experts.*.gate_proj/up_proj/down_proj` |
| dense MLP | `...mlp.gate_proj.weight` + `...mlp.up_proj.weight`（**分开**） | `...mlp.gate_up_proj.weight_packed`（**合并**） |
| 层前缀 | `language_model.model.layers.*` | `model.language_model.layers.*` |
| 张量总数 | 45475 | 45361 |

量化后缀统计：
- model-free：`weight_packed`×22059, `weight_scale`×22059, 明文 `.weight`×1039
- rtn：`weight_packed`×22005, `weight_scale`×22005, 明文 `.weight`×1033

### 3.5 dense MLP 是否被量化（关键）

实测：
- **rtn**：`language_model...layers.0.mlp.gate_up_proj.weight_packed` **存在** → dense gate_up **被 MXFP4 量化了**。
- **model-free**：`...layers.0.mlp.gate_proj.weight` 存在且**无** `weight_packed` → dense **保持 fp、未量化**（符合 ignore 意图）。

---

## 4. `--ignore_layers` 为何在 RTN 路径失效

两个模型的 `quantization_config`（都是 compressed-tensors `mxfp4-pack-quantized`）里，
ignore 列表的**形态与命中情况**不同：

**model-free** —— 逐层显式名，原始 schema，三个 dense 投影都单独列出，**全部命中**：
```
language_model.model.layers.0.mlp.gate_proj
language_model.model.layers.0.mlp.up_proj
language_model.model.layers.0.mlp.down_proj
...（ignore 总数 1039，完全展开）
```

**rtn** —— 用正则 + 部分显式名（前缀还变成了 `model.language_model.layers`）：
```
re:.*mlp\.gate_proj.*
re:.*mlp\.up_proj.*
re:.*mlp\.down_proj.*
model.language_model.layers.0.mlp.down_proj
...（ignore 总数 620）
```

**失效机制**：transformers 原生实现把 dense 的 `gate_proj` 和 `up_proj` **融合成了
`gate_up_proj`**。正则 `re:.*mlp\.gate_proj.*` 只能匹配字面 `mlp.gate_proj`，
而融合后的名字是 `mlp.gate_up_proj`——**既不含 `mlp.gate_proj` 也不含 `mlp.up_proj`
子串**，于是 gate_up_proj **漏网、被量化**。只有未融合的 `down_proj` 被显式忽略、保留 fp。

> 一句话：ignore 规则是针对「未融合」的原始层名写的，遇到原生 transformers 的
> 「融合层」名字就匹配不上了。

---

## 5. 为什么 rtn 模型「删掉字段能加载、但精度为 0」

删掉 `layer_types` 等字段只是让 vLLM **config 解析**过关，但底层的语义/权重错位仍在，
多重问题叠加，必然输出垃圾：

1. **激活函数错**：config 写 `silu`，而真实模型是 `swigluoai`（门控 SwiGLU 的 OAI 变体，
   含 clamp/alpha 等，与纯 `silu`-GLU 数值不同）→ 所有 MLP 前向计算错。
2. **dense gate_up 被误量化**（§4）：本该保 fp 的 dense MLP 被 MXFP4 量化，进一步引入误差。
3. **命名 schema 不匹配**（§3.4）：`mlp.experts` vs vLLM 期望的 `block_sparse_moe.experts`、
   `model.language_model` vs `language_model.model` → vLLM 加载器映射不上，MoE 专家权重
   要么未加载、要么错位 → 输出与随机无异。
4. **config 字段不被识别**：`layer_types`/`index_*`/`rope_parameters` 等 vLLM minimax_m3
   不认 → 未删除时直接加载报错（也就是你观察到的现象）。

因此：**加载成功 ≠ 语义正确**。前 3 条决定了即使加载通过，精度也是 0。

---

## 6. 结论与建议

### 6.1 直接结论
- vLLM 当前的 `models/minimax_m3` 实现是**按原始 MiniMax-M3(remote-code) 的命名与
  config schema 写的**。
- **model-free** 路径原样保留该 schema → 兼容，能加载能推理。
- **非 model-free(RTN)** 路径经 transformers 5.12 **原生 `minimax_m3_vl`** 往返 →
  schema 被整体改写 → 与 vLLM 加载器不兼容，且激活/ignore/融合三重错位 → 精度 0。
- 问题**不在权重数值精度**，而在**模型结构表示（schema）与 config 语义**。

### 6.2 推荐做法
**MiniMax-M3 → vLLM 的 MXFP4 量化，使用 model-free 路径。** 它是目前唯一与 vLLM
兼容的产物。

### 6.3 如果一定要用非 model-free（例如需要真实校准/`--iters>0` 的 tuning）
必须让 auto-round 走**原始 remote-code 建模**、而非 transformers 原生实现，使输出 schema
与原始一致。可尝试的方向（按推荐度）：

1. **Pin 一个没有原生 `minimax_m3_vl` 的旧版 transformers**，强制 `from_pretrained` 走
   repo 的 `configuration_minimax_m3_vl.py` / modeling remote code。这是最省事、最能保证
   schema 一致的办法。
2. 或者：在 vLLM 侧为「原生 transformers schema」增加权重/命名映射与 config 兼容
   （`mlp.experts↔block_sparse_moe.experts`、`gate_up_proj` 拆分、`silu→swigluoai`、
   `layer_types/index_*↔sparse_attention_config`）——工作量大、且要同时对齐激活语义。
3. 无论哪条，都要**修正 ignore 规则**以覆盖融合层名（增加 `re:.*mlp\.gate_up_proj.*`），
   避免 dense gate_up 被误量化。

### 6.4 附注（清理）
`-rtn` 目录里有 `.config.json.swp`（vim 残留）与手改 config 的痕迹，建议清理，避免干扰
后续加载与对比。

---

## 附录 A：复现/核对用命令

```bash
MF=/storage/lkk/m3/MiniMax-M3-MXFP4-ignore-mlp-all
RTN=/storage/lkk/m3/MiniMax-M3-MXFP4-ignore-mlp-all-rtn/MiniMax-M3-mxfp-w4g32

# 顶层键 / text_config 差异
python3 -c "import json;a=json.load(open('$MF/config.json'));b=json.load(open('$RTN/config.json'));\
print('top only rtn:',sorted(set(b)-set(a)));\
print('act mf/rtn:',a['text_config'].get('hidden_act'),b['text_config'].get('hidden_act'))"

# dense gate_up 是否被量化
python3 -c "import json;wm=json.load(open('$RTN/model.safetensors.index.json'))['weight_map'];\
print('rtn gate_up_proj.weight_packed:', 'language_model.model.layers.0.mlp.gate_up_proj.weight_packed' in wm)"
python3 -c "import json;wm=json.load(open('$MF/model.safetensors.index.json'))['weight_map'];\
print('mf gate_proj.weight(未量化):', 'language_model.model.layers.0.mlp.gate_proj.weight' in wm)"

# 原始模型 schema
python3 -c "import json;t=json.load(open('/storage/lkk/MiniMax-M3/config.json'))['text_config'];\
print('hidden_act=',t.get('hidden_act'),'| sparse_attention_config=', 'sparse_attention_config' in t)"
```

## 附录 B：关键差异速查表

| 观察点 | model-free（✅ vLLM） | rtn（❌ 精度0） | 影响 |
|---|---|---|---|
| 处理方式 | 直接量化权重张量 | from_pretrained→save_pretrained | 源头 |
| schema | 原始 remote-code | transformers 原生 canonical | 源头 |
| hidden_act | swigluoai | silu | MLP 计算错 |
| dense MLP | 分开 + fp | 融合 gate_up + 量化 | ignore 失效 + 误量化 |
| MoE 专家名 | block_sparse_moe.experts.w1/w2/w3 | mlp.experts.gate/up/down_proj | vLLM 映射不上 |
| 层前缀 | language_model.model.layers | model.language_model.layers | vLLM 映射不上 |
| 稀疏注意力 | sparse_attention_config | layer_types/index_* | 加载报错 |
| rope | rope_theta | rope_parameters | 加载/语义 |
| vision model_type | clip_vision_model | minimax_m3_vl_vision | 语义 |

# auto-round 非 model-free → vLLM 兼容:native schema 转换器（实现 + 运行命令）

> 配套分析文档：`auto_round_model_free_vs_rtn_analysis.md`（讲清了为什么非 model-free
> 产物 vLLM 加载报错 / 精度为 0）。本文档记录**修复方案 A（独立后处理转换器）** 的
> 实现、验证结果与完整运行命令。
>
> 交付物：`/storage/lkk/m3/convert_native_to_original_schema.py`

---

## 1. 问题回顾（一句话）

auto-round 非 model-free（`from_pretrained` 建原生 `minimax_m3_vl` 模型再 `save_pretrained`）
产出的是 **transformers 原生 schema**（命名/config 全变），vLLM 的 `models/minimax_m3`
加载器只认**原始 checkpoint schema**（= model-free / AMD-Quark 的产物）。

修复目标：把非 model-free 的导出目录**转回原始 schema**，同时保留其 `--iters>0` 的
真实 tuning 能力。

---

## 2. 为什么不能直接复用现成 mapping（实测结论）

transformers 5.12 有权威的 `transformers.conversion_mapping.get_checkpoint_conversion_mapping`，
auto-round 也封装了 `revert_checkpoint_conversion_mapping`。但实测证明**不足以**修复：

1. **早退出 bug**：`revert_checkpoint_conversion_mapping` 命中一条规则就 `return`。名字先被
   前缀规则 `model.language_model.→language_model.model.` 命中就停，MoE 规则
   （`mlp.experts`→`block_sparse_moe.experts`）再也匹配不到。
2. **拆分/融合命名对不上**：mapping 的融合逆规则针对 **3D fused** `mlp.experts.gate_up_proj`↔`w1/w3`；
   但 auto-round 已把 experts 拆成 per-expert `experts.{i}.gate_proj/up_proj/down_proj`，
   缺少 `gate_proj→w1, up_proj→w3, down_proj→w2` 这一级规则。
3. **dense 需要拆张量**：`mlp.gate_up_proj`→`gate_proj`+`up_proj` 是一个张量拆两个，纯 rename
   做不到，且要求 dense 未量化（fp）。

> 结论：必须写 **MiniMax-M3 专用转换器**。

---

## 3. 转换规则（native → original）

| native（auto-round 非 model-free） | original（vLLM 兼容） | 处理 |
|---|---|---|
| `...mlp.experts.{i}.gate_proj.*` | `...block_sparse_moe.experts.{i}.w1.*` | rename |
| `...mlp.experts.{i}.up_proj.*` | `...block_sparse_moe.experts.{i}.w3.*` | rename |
| `...mlp.experts.{i}.down_proj.*` | `...block_sparse_moe.experts.{i}.w2.*` | rename |
| `...mlp.gate.weight` | `...block_sparse_moe.gate.weight` | rename |
| `...mlp.gate.e_score_correction_bias` | `...block_sparse_moe.e_score_correction_bias` | rename |
| `...mlp.shared_experts.down_proj.*` | `...block_sparse_moe.shared_experts.down_proj.*` | rename |
| `...mlp.shared_experts.gate_up_proj.*` | `...block_sparse_moe.shared_experts.gate_proj.*` + `up_proj.*` | **split dim0** |
| `...mlp.gate_up_proj.*`（dense） | `...mlp.gate_proj.*` + `...mlp.up_proj.*` | **split dim0** |
| `...mlp.down_proj.*`（dense） | 不变 | — |
| `...self_attn.indexer.q_proj/k_proj/q_norm/k_norm.*` | `...self_attn.index_q_proj/index_k_proj/index_q_norm/index_k_norm.*` | rename |
| 其它（embed/norm/layernorm/self_attn.{q,k,v,o}/vision_tower/projector…） | 不变 | — |

**split dim0 的正确性**：MXFP4 pack-quantized 沿 **输入维(dim1)** 打包、分组 scale 也在
dim1；**输出维(dim0 = 2*inter)** 不动。因此 `gate_up_proj` 沿 dim0 切半即可干净拆成
gate（前半）/ up（后半），对 `weight` / `weight_packed` / `weight_scale` 一致适用。已用
shape 断言 + 数值断言验证。

**config.json**：整体替换为**原始 MiniMax-M3 的 config.json**（恢复 `hidden_act=swigluoai`、
`sparse_attention_config`、`rope_theta`、vision `clip_vision_model` 等），仅注入
`quantization_config`。

**quantization_config.ignore 重生成**：不做脆弱的正则逐条转换，而是**从转换后的权重直接推导**——
凡是有 `.weight`/`.bias` 但无 `.weight_packed` 等量化后缀的模块 = 未量化 = 进 ignore。
这与 model-free 的全展开 ignore 风格一致，且如实反映权重的量化状态。

---

## 4. 关键前置：修 ignore（否则 dense 被误量化）

实测 auto-round 的 ignore 匹配（`to_standard_regex` + `re.search`）在 **native 模型**下：

| ignore 项 | 生成的 regex | 命中 |
|---|---|---|
| `mlp.gate_proj` | `.*mlp\.gate_proj.*` | **0 个（死规则！）** |
| `mlp.up_proj` | `.*mlp\.up_proj.*` | **0 个（死规则！）** |
| `mlp.down_proj` | `.*mlp\.down_proj.*` | 仅 dense `mlp.down_proj` ✓ |
| `mlp.gate_up_proj` | `.*mlp\.gate_up_proj.*` | 仅 dense `mlp.gate_up_proj` ✓（不误伤 shared_experts/experts） |

原因：native 模型里 dense 的 gate/up 融合成 `mlp.gate_up_proj`，`mlp.gate_proj` 这种裸名
匹配不到任何模块。所以你原命令里的 `mlp.gate_proj,mlp.up_proj` 是**无效规则**，dense
gate_up 实际被量化了 → 这是精度为 0 的原因之一。

**修复**：`--ignore_layers` 里把 `mlp.gate_proj,mlp.up_proj` 换成 `mlp.gate_up_proj`
（保留 `mlp.down_proj`）。`mlp.gate_up_proj` 精确命中 dense 融合层，且**不误伤**
`shared_experts.gate_up_proj`（要量化）与 `experts.*`（要量化）。

---

## 5. 验证结果（全部通过）

### 5.1 dry-run 对照 model-free（真实 rtn 产物，只读 safetensors header）
```
[dry-run] src tensors=45361 renamed=44232 split=120 -> out tensors=45481
[dry-run] converted keys=45481 expected keys=45475
[dry-run] shape mismatches on common keys: 0
[dry-run] regenerated ignore=1033 expected-model ignore=1039
[dry-run] ignore only in converted: 0
[dry-run] ignore only in expected : 6   (仅 dense 层 0/1/2 的 gate/up)
```
- **shape 零错**；命名与 ignore 与 model-free **零差异**。
- 唯一残留 = dense 层 0/1/2 的 gate/up：转换后是量化的（12 个 packed+scale），model-free
  是 fp（6 个 weight）。这纯粹因为**当前 rtn 产物 dense 被误量化**（ignore 死规则），
  **不是转换器问题**。修 ignore 重量化后即消失。

### 5.2 端到端冒烟测试（小张量，验证 split/save/load/config）
```
[OK] dense gate_up split: gate=前半, up=后半 正确
[OK] experts gate->w1, up->w3, down->w2 正确
[OK] shared_experts / router(gate + e_score_correction_bias) / indexer 正确
[OK] config.json = 原始 config + quantization_config
[OK] aux 文件已拷贝
全部冒烟测试通过 ✓
```

> 说明：未做 vLLM 端到端精度验证（量化需 GPU + 数百 GB + 数小时）。命名/shape/config/ignore
> 已全面验证；修 ignore 重量化后，转换输出将与 model-free 逐一致。

---

## 6. 完整运行流程

### 第 1 步：重量化（修 ignore，让 dense 保 fp）
把死规则 `mlp.gate_proj,mlp.up_proj` 换成 `mlp.gate_up_proj`：
```bash
auto-round ../MiniMax-M3/ --scheme MXFP4 --iters 0 \
  --ignore_layers vision_tower,lm_head,block_sparse_moe.gate,embed_tokens,self_attn,\
patch_merge_mlp,multi_modal_projector,mlp.gate_up_proj,mlp.down_proj \
  --format llm_compressor --output_dir "./M3-rtn-native"
```
> 需要真实 tuning 时把 `--iters 0` 改成 `--iters 200` 等（这正是非 model-free 的价值）。

### 第 2 步：预检（推荐，只读 header，不搬数据）
```bash
python3 /storage/lkk/m3/convert_native_to_original_schema.py \
  --src ./M3-rtn-native/<自动生成的子目录> \
  --orig-config /storage/lkk/MiniMax-M3 \
  --expected /storage/lkk/m3/MiniMax-M3-MXFP4-ignore-mlp-all \
  --dry-run
```
期望：`shape mismatches: 0` 且 `ignore only in converted: 0`、`only in expected: 0`
（修 ignore 后 dense 也 fp，差异应归零）。

### 第 3 步：正式转换
```bash
python3 /storage/lkk/m3/convert_native_to_original_schema.py \
  --src ./M3-rtn-native/<子目录> \
  --orig-config /storage/lkk/MiniMax-M3 \
  --dst ./M3-rtn-vllm
```
产出 `./M3-rtn-vllm`（original schema，含转换后的 safetensors + index.json + 原始
config.json + 注入的 quantization_config + tokenizer/processor 等 aux 文件）。

### 第 4 步：推理评测
把 `eval_vllm.sh` 的模型路径指向 `./M3-rtn-vllm`。

---

## 7. 转换器 CLI 参数

```
--src           native-schema 导出目录（auto-round 非 model-free 输出的模型目录）
--dst           输出目录（original schema）；除 --dry-run 外必填
--orig-config   原始 MiniMax-M3 模型目录（config.json + aux 文件来源）
--dry-run       只校验命名/shape，不搬权重
--expected      dry-run 时对照的已知正确 original-schema 模型目录（如 model-free 产物）
--max-shard-gb  输出分片大小上限，默认 5.0
```

---

## 8. 实现要点（供后续集成 auto-round 参考）

- 转换逻辑集中在 `convert_name(name)`：返回 `("rename", [n])` 或 `("split", [n1,n2])`。
- 张量拆分 `_split_dim0`：沿 dim0 切半（gate=前半、up=后半）。
- 权重写出走**流式**：按源 shard 逐个 `load_file → 转换 → 累积 → flush`，控制内存与分片大小，
  最后统一重命名为 `model-XXXXX-of-YYYYY.safetensors` 并写 `model.safetensors.index.json`。
- `regenerate_ignore(all_names)`：从转换后权重推导 ignore（未量化模块），鲁棒且对齐 model-free。
- 源模型目录通过 `--orig-config` 显式给出（对应 auto-round 内部的
  `model.config._name_or_path` / `_resolve_model_source_dir`）。

### 第二层（可选）：集成进 auto-round
把上述转换 + config 拷贝 + ignore 修正，接入
`auto_round/export/export_to_llmcompressor/export.py :: save_quantized_as_llmcompressor`
的导出末尾（注意需同时覆盖 `save_model` 常规路径与 `ShardWriter` immediate-saving 路径），
即可让非 model-free 直接产出 vLLM 兼容模型，无需后处理。

---

## 9. 相关文件

- 转换器：`/storage/lkk/m3/convert_native_to_original_schema.py`
- 分析文档：`/storage/lkk/m3/auto_round_model_free_vs_rtn_analysis.md`
- 参照产物（原始 schema，能跑）：`/storage/lkk/m3/MiniMax-M3-MXFP4-ignore-mlp-all`（model-free）、
  `/storage/lkk/MiniMax-M3-amd`（AMD-Quark）
- 待转换产物（native schema）：`/storage/lkk/m3/MiniMax-M3-MXFP4-ignore-mlp-all-rtn/MiniMax-M3-mxfp-w4g32`
- 原始模型：`/storage/lkk/MiniMax-M3`

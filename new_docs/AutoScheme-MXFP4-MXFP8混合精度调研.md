# AutoRound 自动混合精度（MXFP4 + MXFP8）调研

> 目标：在 MiniMax-M3 的 MXFP4 量化中提升精度。结论是 auto-round **原生支持**自动 tune
> MXFP8/MXFP4 的混合精度分配，会自动决定「哪些层用 MXFP8、哪些层用 MXFP4」。
> 本文记录功能位置、自动 tune 原理、使用方法与注意事项。

---

## 1. 结论速览

- 功能名：**AutoScheme**（auto-round v0.8+ 引入），代码在 `auto_round/auto_scheme/`。
- 输入一个**目标平均 bits**（`avg_bits` / `--target_bits`）和候选集 `options=("MXFP4","MXFP8")`，
  它会用 **delta-loss 敏感度打分 + 背包式预算求解**，自动把 MXFP8 分配给最敏感的层、
  其余层用 MXFP4，使整体平均 bits 命中目标且总精度损失最小。
- **可以真实导出**：`llm_compressor` 格式支持混合导出，写成 `"format": "mixed-precision"`，
  按 `(bits, data_type)` 分组（见 `export_to_llmcompressor/export_to_fp.py` 的
  `_build_mixed_fp_quantization_config`）。文档里"混合数据类型无法导出"的说明已过时。
- **关键约束**：AutoScheme 的打分必须要校准数据，因此**不能配合 `--model_free`**
  （`compressors/entry.py:296` 强制走 `CalibratedRTNCompressor`）。用 `--iters 0`（RTN，快）
  或 `--iters 200`（调优感知，慢但更准）。

---

## 2. 自动 tune 原理：如何判断哪些层用 MXFP8、哪些用 MXFP4

默认方法名为 `default` / `DeltaLoss`，实现于 `auto_round/auto_scheme/delta_loss.py`，
注册见函数 `gen_layer_config`（`@register_scheme_methods(("default","DeltaLoss"))`）。

### 2.1 核心思想：一阶泰勒近似的"量化敏感度"

对每个候选方案（MXFP4、MXFP8）分别跑一遍：把每一层用 RTN（`iters=0`）做
量化→反量化（QDQ），得到量化误差 `ΔW = W - QDQ(W)`；然后用一小批校准数据做
**前向 + 反向**，拿到损失对量化权重的梯度 `g = ∂L/∂QDQ(W)`。该层在该方案下的敏感度分数：

```
weight_score = Σ | g ⊙ (W - QDQ(W)) |          # 一阶估计：量化该层权重带来的 loss 增量
act_score    = Σ | g_x ⊙ (x - QDQ(x)) | / N     # 激活侧同理（仅当 act_bits ≤ 8 时，按样本平均）
mix_score    = weight_score + act_score          # 该层在该方案下的总敏感度分数
```

对应代码：`AutoSchemeWrapperLinear.post_init_qdqw` 里的 `save_grad`（权重侧）和
`_qdq_act` 里的 `save_grad`（激活侧），最终写入每层的 `m.mix_score`
（`get_score_for_scheme` 收集为 `scores_dict[name] = [layer_bits, mix_score]`）。

**直觉**：`mix_score` 越大，说明"用这个（更激进的）方案量化这一层，对最终 loss 伤害越大"，
即这一层越敏感。MXFP4 的误差 `ΔW` 比 MXFP8 大，因此同一层在 MXFP4 下的 `mix_score`
通常明显高于 MXFP8——分数差就代表"把这层从 MXFP8 降到 MXFP4 会额外损失多少精度"。

### 2.2 从分数到分配：带预算约束的最小损失求解（背包/DP）

打完分后，每一层都有一组候选：`(方案, 比特开销 bits_cost, 损失 loss_cost)`。
- MXFP8：`bits_cost` 高、`loss_cost` 低；
- MXFP4：`bits_cost` 低、`loss_cost` 高。

目标是：**为每层各选一个方案，使 Σ loss_cost 最小，且 Σ bits_cost ≤ 预算**，
其中预算 `target_params_cnt = 总参数量 × avg_bits`。

这是一个有界背包问题，用动态规划求解：`choose_bits_per_layer_with_path(total_scores, P)`
（`delta_loss.py:1072`）。它以"累计比特数"为状态、"累计损失"为代价，逐层扩展，
配合 **Pareto 剪枝**（相同/更高比特但损失更大的状态被丢弃）和 **beam 宽度限制**
（`max_states`，防止层数多时状态爆内存），最后取"总损失最小的可行解"。

**最终效果**：敏感层（MXFP4 下 delta-loss 高）被分配 MXFP8，不敏感层被分配 MXFP4，
在满足平均 bits 预算的前提下整体精度损失最小。

### 2.3 特殊层处理

- **fixed 层**：用户在 `layer_config` 里显式指定的层不参与打分，直接固定并从预算里扣除。
- **Embedding**：小样本下打分不可靠，用启发式按 `avg_bits` 与是否 tie_word_embeddings
  选一个方案（`_select_embedding_scheme_index`），不进 DP。
- **lm_head**：单独的 head trick（`_apply_head_trick`）。
- **shared_layers**：把需共享精度的层（如融合的 QKV / MoE 专家）合并成一个 DP 单元，
  强制同方案，避免融合 kernel 因精度不一致报错（`parse_shared_layers` + 合并 score）。
- **MoE 模型**：默认自动增大校准样本数（`nsamples=64`）以获得更稳的 recipe。

---

## 3. 命令行用法（在原命令基础上改）

> 注意：去掉 `--model_free`，改用 `--iters 0`；`ignore_layers` 与原来保持一致。

```bash
auto-round ../MiniMax-M3/ --iters 0 \
  --avg_bits 4.5 --options "MXFP4,MXFP8" --ignore_scale_zp_bits \
  --ignore_layers vision_tower,lm_head,block_sparse_moe.gate,embed_tokens,self_attn,patch_merge_mlp,multi_modal_projector,mlp.gate_proj,mlp.up_proj,mlp.down_proj \
  --format llm_compressor --output_dir "./MiniMax-M3-MXFP4-MXFP8-auto"
```

- `--avg_bits`（别名 `--target_bits`）：越接近 4 → 更多 MXFP4（更省、精度略低）；
  越接近 8 → 更多 MXFP8（更准、更大）。先试 `4.5`，不够精度再上 `5.0`。
- `--options "MXFP4,MXFP8"`：候选集。
- `--ignore_scale_zp_bits`：算平均 bits 时忽略 scale/zp 开销，让 avg_bits 语义接近纯元素位宽。
- `--iters 0` 快（秒~分钟级 RTN 打分）；`--iters 200` 慢但更准。

## 4. 关键参数（`AutoScheme`）

| 参数 | 说明 |
|------|------|
| `avg_bits: float` | 目标平均 bits，仅统计待量化层。必须落在候选集的最小/最大平均 bits 之间。 |
| `options` | 候选方案：字符串 `"MXFP4,MXFP8"` / 列表 / `QuantizationScheme`。 |
| `ignore_scale_zp_bits: bool` | 算平均 bits 时是否忽略 scale/zp 开销（推荐 True）。 |
| `shared_layers` | 需共享同一方案的层分组（支持正则与全名，按 block 匹配）。 |
| `nsamples / seqlen / batch_size / dataset` | 打分用校准配置；MoE 默认 nsamples=64。 |
| `device_map` | AutoScheme 显存占用更高，可单独指定设备映射。 |
| `low_gpu_mem_usage=True` | 省显存但更慢（默认开启，大模型建议保留）。 |

## 4.1 MiniMax-M3（MoE）的 `shared_layers` 配置

> ⚠️ **关键：量化时是 native schema，不是 vLLM/原始 checkpoint schema。**
> AutoScheme 必须走非 model_free 路径，auto-round 通过 transformers 的 native 实现
> (`model_type=minimax_m3_vl`, transformers≥5) 加载模型，模块命名与 vLLM 需要的原始
> checkpoint 不同，需经 `convert_native_to_original_schema.py` 转换后才能被 vLLM 加载。
> **shared_layers / ignore_layers 必须写 native 名，否则匹配不到、静默失效。**

### 两套 schema 对照

| 含义 | 量化时（native，auto-round 内部/导出） | 转换后（original，vLLM 加载） |
|------|-----------------------------------------|-------------------------------|
| MoE 块 | `...layers.N.mlp` | `...layers.N.block_sparse_moe` |
| 专家 gate | `mlp.experts.M.gate_proj` | `block_sparse_moe.experts.M.w1` |
| 专家 up | `mlp.experts.M.up_proj` | `block_sparse_moe.experts.M.w3` |
| 专家 down | `mlp.experts.M.down_proj` | `block_sparse_moe.experts.M.w2` |
| 共享专家 | `mlp.shared_experts.gate_up_proj`(融合) + `down_proj` | `...shared_experts.gate_proj`+`up_proj`+`down_proj` |
| 路由器 | `mlp.gate`（自定义模块，非 Linear，**不会被量化**） | `block_sparse_moe.gate` |

注：native 里 128 专家本是单个融合 3D Parameter，auto-round 的
`prepare_model_for_moe_quantization` 会把它**拆成 per-expert 的 nn.Linear**，
并把 `gate_up` 拆成 `gate_proj`+`up_proj`，所以量化粒度是 per-expert 的
`mlp.experts.M.gate_proj/up_proj/down_proj`。

### 为什么 MoE 必须配 `shared_layers`
部署时 vLLM/SGLang 的 `FusedMoE` 把**同一层 128 专家**堆成融合 GEMM：
`gate_proj`+`up_proj` → gate_up GEMM、`down_proj` → down GEMM，各自必须整层统一方案。
若不加 shared_layers，AutoScheme 会对每个专家独立选方案 → 融合权重方案不一 → **vLLM 无法加载**。

`parse_shared_layers` 按 block 逐层匹配：一条正则在**每层内部**把 128 专家聚成一组，
**层间**仍可各自 MXFP4/MXFP8；同时把 DP 单元从 `~57×128×3≈2 万` 降到 `~百级`。

### 推荐配置（API，native 正则；脚本已内置 `SHARED_LAYERS_MOE`）
```python
shared_layers = [
    # 每层内 128 专家的 gate(gate_proj)+up(up_proj) 同精度（对应 vLLM gate_up 融合）
    [r".*\.mlp\.experts\.\d+\.gate_proj$", r".*\.mlp\.experts\.\d+\.up_proj$"],
    # 每层内 128 专家的 down(down_proj) 同精度
    [r".*\.mlp\.experts\.\d+\.down_proj$"],
]
```
共享专家 `mlp.shared_experts.gate_up_proj` 是单个融合 Linear，天然单一方案，无需分组。

**更保守替代**（整层专家统一）：
```python
shared_layers = [[r".*\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"]]
```

### 与转换脚本 / vLLM 的兼容性检查
1. **专家层内一致性**：脚本内置 `verify_expert_consistency()` 会在量化后检查每个 MoE 层的
   `gate_up` / `down` 组是否方案一致，不一致会明确报错（即 shared_layers 没生效）。
2. **导出格式**：确认 `config.json` 出现 `"format": "mixed-precision"`，且同层专家 target 落在同一 group。
3. **转换脚本**：`convert_native_to_original_schema.py` 仅做 1:1 改名 + gate_up 按 dim0 拆分，
   不改量化方案；只要量化时专家层内一致，转换后即满足 vLLM FusedMoE 要求。
4. **框架支持**：确认 vLLM(`/storage/lkk/vllm_xpu/vllm`) 版本支持 compressed-tensors 的
   **逐层不同** MoE 方案（部分层 MXFP4、部分层 MXFP8）。若不支持，把整个 MoE 统一到单一方案。

### 关于 ignore_layers 的提醒（原命令）
原命令的 `ignore_layers` 用的是 **original schema** 名，在 native 下部分不匹配：
- `block_sparse_moe.gate`（native 是 `mlp.gate`）→ 不匹配，但路由器非 Linear 本就不量化，无影响。
- `mlp.gate_proj` / `mlp.up_proj`（native 的 dense 层是融合的 `mlp.gate_up_proj`）→ 不匹配，
  意味着前 3 个 dense 层的 gate_up 仍会被量化。若确需跳过 dense 层，改用 native 名
  `mlp.gate_up_proj`（注意别误伤 `mlp.experts.*`，可用更精确的正则/层号）。
- `self_attn`、`vision_tower`、`embed_tokens`、`multi_modal_projector` 在 native 下仍能匹配。

## 5. 成本与限制

- **显存/耗时**（A100，官方文档）：MXFP4/MXFP8 每个 option 约 `60s × option 数 × 模型规模`，
  显存约为 BF16 加载的 1.1~1.5×。M3 很大，建议保留 `low_gpu_mem_usage=True`。
- **限制**：Embedding 层不参与自动量化，直接用候选中最高精度方案；
  融合层需用 `shared_layers` 保证同精度。

## 6. MiniMax-M3 的 MoE 结构与三套层命名（重要背景）

### 6.1 M3 的 MoE 结构（实测数字）

| 项目 | 数值 |
|------|------|
| Transformer 总层数 | 60（**前 3 层 dense MLP + 后 57 层 MoE**） |
| 每个 MoE 层专家数 | 128（top-4 激活） |
| 共享专家 | 1 个（常开，DeepSeek 风格） |
| 每个专家结构 | SwiGLU MLP = 3 个 Linear（w1=gate, w3=up, w2=down） |
| 每个 MoE 层专家 Linear 数 | 128 × 3 = **384** |
| 全模型专家 Linear 总数 | 57 × 384 = **21,888** |
| hidden / 专家 inter / dense inter | 6144 / 3072 / 12288 |

> `block_sparse_moe.experts` 只出现在 **57 个 MoE 层**里（layers 3~59）；layers 0~2 是 dense。

### 6.2 与"普通 MoE"（Mixtral）的差异

| 特性 | Mixtral | MiniMax-M3 |
|------|---------|------------|
| 专家数 / 激活 | 8 top-2 | **128 top-4**（细粒度） |
| 共享专家 | 无 | **1 个常开** |
| gate/up | w1、w3 分开 | **gate_up 融合**（2×inter，GPT-OSS 风格） |
| 专家存储(native) | per-expert 模块 | **3D 堆叠大张量**（吞吐更高） |
| 路由器打分 | softmax | **sigmoid + e_score_correction_bias**（无辅助损失均衡） |
| 层布局 | 全 MoE | **前 3 dense + 后 57 MoE** |

对量化最关键：**128 专家在推理时被 vLLM 融合成一个 GEMM，同层专家必须同一方案**（→ 需 `shared_layers`）；
且 **gate_up 融合 → gate(w1) 与 up(w3) 必须同精度**。

### 6.3 三套层命名（同一批权重的不同布局）

同一层 128 专家的权重存在三种命名/布局，理解它们是配置 `shared_layers`/`ignore_layers` 和转换脚本的关键：

| 状态 | 何时出现 | 专家命名 | 布局 |
|------|----------|----------|------|
| **① original** | MiniMax 发布的 checkpoint / **vLLM 需要的** | `...block_sparse_moe.experts.M.w1/w2/w3.weight` | **per-expert**，每层 384 张量 |
| **② native（3D 融合）** | transformers 5 内置实现 `from_pretrained` 后的**内存布局**，`save_pretrained` 也写这个 | `...mlp.experts.gate_up_proj` `[128,2·3072,6144]` + `...mlp.experts.down_proj` `[128,6144,3072]` | **128 专家堆成 3D**，gate/up 融合，每层 2 张量 |
| **③ native（unfused）** | auto-round **量化时/导出的**布局（`prepare_model_for_moe_quantization` 把 3D 拆回 per-expert Linear，并把 gate_up 拆成 gate/up） | `...mlp.experts.M.gate_proj/up_proj/down_proj` | **per-expert**，前缀是 native 的 `mlp` |

> `shared_layers` / `ignore_layers` 必须匹配 **③（native unfused）**，即 `mlp.experts.M.gate_proj/...`。
> 量化导出的模型也是 ③ 的命名，需转换脚本映射回 **①** 才能给 vLLM。

**其它对应的改名**（native ③ → original ①，见转换脚本）：

| 含义 | native（③，量化/导出） | original（①，vLLM） |
|------|------------------------|----------------------|
| MoE 块前缀 | `mlp` | `block_sparse_moe` |
| 专家 gate/up/down | `experts.M.gate_proj/up_proj/down_proj` | `experts.M.w1/w3/w2` |
| 路由器 | `mlp.gate`（非 Linear，不量化） | `block_sparse_moe.gate` |
| 共享专家 | `mlp.shared_experts.gate_up_proj`(融合)+`down_proj` | `shared_experts.gate_proj`+`up_proj`+`down_proj` |
| dense MLP(0~2层) | `mlp.gate_up_proj`(融合)+`down_proj` | `mlp.gate_proj`+`up_proj`+`down_proj` |
| 注意力 indexer | `self_attn.indexer.q_proj/k_proj/q_norm/k_norm` | `self_attn.index_q_proj/index_k_proj/...` |
| 顶层前缀 | `model.language_model.` | `language_model.model.` |
| 视觉塔 | `model.vision_tower.` | `vision_tower.vision_model.` |

## 7. 为什么 auto-round 导出的模型必须经过转换脚本

### 7.1 根因：非 model_free 路径写出的是 native schema

- AutoScheme（含本方案的混合精度）**必须走非 model_free 路径**（需要校准数据打分）。
- 该路径用 `transformers.from_pretrained` 加载模型；在 transformers≥5 中 M3 有 native 实现
  （`model_type=minimax_m3_vl`），加载后是 **native 3D 融合布局（②）**。
- auto-round 量化时把专家 unfuse 成 per-expert（③），量化完 `save_pretrained` 导出的
  模型仍是 **native 命名（③ 的 `mlp.experts...`、`self_attn.indexer...` 等）**。
- 而 vLLM 的 `models/minimax_m3` loader 期望的是**原始 checkpoint 命名（①）**，
  直接加载 native 命名会失败（即使手动改 config 也会精度为 0）。

> 对比：**model_free 路径**是直接对 checkpoint 文件做 file-to-file 量化，命名保持 ①，
> 因此**不需要**转换脚本；但 model_free 不能配 AutoScheme，且你反馈精度较差。

### 7.2 转换脚本 `convert_native_to_original_schema.py` 做了什么

把 native schema 的导出**改写回 original schema**，三件事：

1. **张量改名（1:1）**：按 6.3 的映射表逐个重命名，例如
   `...mlp.experts.5.gate_proj.weight_packed` → `...block_sparse_moe.experts.5.w1.weight_packed`
   （量化产物 `.weight_packed/.weight_scale/.weight_shape/...` 一并搬迁）。
2. **融合张量按 dim0 拆分**：native 里 dense 与 shared 的 `gate_up_proj` 是融合的
   （输出维 = 2×inter），沿 **dim0 前一半 = gate、后一半 = up** 切开成两个张量。
   （MXFP4 是沿输入维 dim1 打包、scale 也在 dim1，输出维 dim0 不受影响，故可干净切分，脚本有 shape 断言校验。）
3. **config.json 重建**：用**原始模型的 config.json**（恢复 hidden_act、sparse_attention_config、
   rope_theta、vision model_type 等），只注入转换后的 `quantization_config`，并把其中
   `ignore`/`targets` 的层名也一并从 native 改成 original。

### 7.3 转换脚本会不会破坏混合精度？—— 不会，但有前提

- 脚本只做**改名 + 拆分 + 搬运量化产物**，**完全不改量化方案**（bits/data_type/scale 都原样搬）。
- 因此混合精度（哪些层 MXFP4、哪些 MXFP8）在转换前后**完全保留**。
- **前提**：量化时同一 MoE 层的 128 专家必须**方案一致**（靠 `shared_layers` 保证）。
  若不一致，vLLM 把它们堆成融合 GEMM 时会因 packed dtype/shape 不同而**无法加载**——
  这正是脚本 `auto_scheme_mxfp_mix.py` 里 `verify_expert_consistency()` 兜底检查的点。

### 7.4 完整流程

```
原始 checkpoint (schema ①)
        │  auto-round + AutoScheme (非 model_free, iters=0)
        │  ├─ transformers 加载 → native 3D 融合 (②)
        │  ├─ prepare_model_for_moe_quantization → unfuse per-expert (③)
        │  ├─ AutoScheme delta-loss 打分 + DP 分配 (shared_layers 保证专家层内一致)
        │  └─ 导出 llm_compressor "mixed-precision" (命名仍是 native ③)
        ▼
native-schema 量化模型 (③)
        │  convert_native_to_original_schema.py
        │  ├─ 张量改名 ③ → ①
        │  ├─ gate_up 按 dim0 拆分
        │  └─ 用原始 config.json + 转换后的 quantization_config
        ▼
original-schema 量化模型 (①)  →  /storage/lkk/vllm_xpu/vllm 推理
```

> 用法：`python convert_native_to_original_schema.py --src <auto-round输出> --dst <转换输出> --orig-config ../MiniMax-M3/`
> 建议先 `--dry-run [--expected <已知可用模型>]` 校验层名/shape 再正式转换。

## 7.5 实战排查：vLLM 推理精度为 0 的两个根因（已修复）

一次实跑（`avg_bits=4.5`）中，量化本身成功（AutoScheme 选了 13 个 MoE 层为 MXFP8，
`shared_layers` 生效、每层 128 专家一致），但 vLLM 上 gsm8k 精度为 **0**。定位到两个 bug：

### Bug ① 转换脚本未翻译 `config_groups[*].targets`（精度 0 的直接原因）
- mixed-precision 导出里，MXFP8 组的 `targets` 是 **native 名**
  `model.language_model.layers.16.mlp.experts.0.gate_proj`。
- 旧版 `convert_native_to_original_schema.py` 只改张量名和 `ignore`，**没动 config_groups.targets**。
- vLLM 用 **original 名** `language_model.model.layers.16.block_sparse_moe.experts.0.gate_proj`
  匹配（见 `compressed_tensors_moe.get_moe_method`：用 `{experts}.0.gate_proj/up_proj/down_proj`）
  → 全部匹配不到 → 所有 MoE 层落到默认组 MXFP4 → 但 MXFP8 层张量是 `weight`(fp8) 而非
  `weight_packed`(fp4) → 被当 MXFP4 误读 → 权重全乱 → acc 0。
- **修复**：`convert_quantization_config` 增加对 `config_groups[*].targets` 的 native→original
  转换；并在名字映射里补上 `mlp.experts.`→`block_sparse_moe.experts.`、`mlp.shared_experts.`→
  `block_sparse_moe.shared_experts.`（**保留 gate_proj/up_proj/down_proj 投影名**，因为
  vLLM 用投影名匹配 scheme，而张量名才用 w1/w2/w3，二者由 vLLM 的 ckpt 映射对应）。

### Bug ② vLLM 要求同层 gate/up/down 三投影方案一致
- `get_moe_method` 会检查每个 MoE 层 `{experts.0.gate_proj, up_proj, down_proj}` 的 scheme，
  不一致直接 `raise "All MoE projections need to have same quantization scheme"`。
- 早期 `shared_layers` 把 **gate_up 与 down 分成两组**，导致 AutoScheme 产生
  "gate_up=MXFP8 但 down=MXFP4" 的层（实测 7 层：17,23,24,25,40,43,44）→ vLLM 无法加载。
  （本次被 Bug① 掩盖成 acc 0；修好 ① 后 ② 会变成硬报错。）
- **修复**：`SHARED_LAYERS_MOE` 改为把每层 `gate_proj+up_proj+down_proj`（外加 shared_experts）
  **全部绑成一个方案** → 整层 MXFP8 或整层 MXFP4，层间仍可混合。

### 结论与正确做法
- 两个 bug 都已在 `auto_scheme_mxfp_mix.py` 和 `convert_native_to_original_schema.py` 中修复。
- Bug ② 是物理性的（磁盘上 down 已是 MXFP4 打包），**必须用修正后的 shared_layers 重新量化**，
  无法仅靠改 config 挽救。
- vLLM 的 per-layer MoE 方案是逐层选择的（每个 `RoutedExperts` 单独调 `get_moe_method`），
  所以"部分层 MXFP8、部分层 MXFP4"是**支持**的；`compressed_tensors_moe_w8a8_mxfp8.py` 提供 MXFP8 MoE 核。
- 脚本内置的 `verify_expert_consistency()` 已升级为**按层校验 gate/up/down 一致**，重量化后应显示 ✅。

### 正确流程（重跑）
```
1) python auto_scheme_mxfp_mix.py --model ../MiniMax-M3/ --avg_bits 4.5   # 修正 shared_layers 后重量化
   → 确认输出 [校验] ✅（整层 MXFP8=N 层，整层 MXFP4=M 层）
2) python convert_native_to_original_schema.py --src <上一步输出> --dst <转换输出> --orig-config /storage/lkk/MiniMax-M3
   → 确认转换后 config.json 的 config_groups.group_0.targets 已是
     language_model.model.layers.*.block_sparse_moe.experts.*.gate_proj（original 名）
3) vLLM 加载 <转换输出> 评测
```

## 7.6 实战排查（续）：Bug①②修复后仍 acc=0 —— 第③个根因：dense 首层被误量化

修好 Bug①②、config/targets 正确、逐层一致后，vLLM 仍 gsm8k=0（strict 与 flexible 均 0）。
逐项排查确认：**磁盘权重数值正常**（MXFP8 MoE 反量化 absmean≈0.027、dense MXFP8≈0.015，均健康），
config、vLLM 解析均正确。关键信号是 **flexible-extract 也为 0（1319 题无一提取到答案）→
模型从首层就输出乱码 → 前几层被污染**。

### Bug ③ dense MLP 首层被误量化成 MXFP8（ignore 未命中 native 融合名）
- auto-round 的 ignore 匹配（`get_fp_layer_names`, `compressors/utils.py:1010`）在**量化时**对
  **native 模块名**做子串匹配。
- native 的 dense MLP 是**融合的单个 Linear** `mlp.gate_up_proj`（`MiniMaxM3VLDenseMLP.gate_up_proj`），
  而原命令 ignore 用的是 `mlp.gate_proj` / `mlp.up_proj`，**子串匹配不到 `mlp.gate_up_proj`**。
- 结果：dense 的 `gate_up` 被量化成 MXFP8，`down_proj`（未融合，匹配上）保持 BF16 →
  **前 3 层 dense 半 MXFP8 半 BF16，首层污染 → 整个前向崩溃 → 精度 0**。
- 对照：工作正常的纯 MXFP4 模型里 dense 层全是 BF16（0.937）。
- **修复**：`IGNORE_LAYERS` 把 dense 改用 native 融合名 `mlp.gate_up_proj`（保留 `mlp.down_proj`）。
  已验证 `mlp.gate_up_proj` 子串只匹配 dense，不误伤 `experts.*.gate_proj` 与
  `shared_experts.gate_up_proj`。脚本新增 `verify_dense_ignored()` 兜底检查 dense 是否被误量化。

### schema 陷阱总结（三处名字都要用 native 名）
| 配置项 | 作用时机 | 必须用的名字 | 原命令的错 |
|--------|----------|--------------|------------|
| `shared_layers` | 量化时按 native 模块名分组 | `mlp.experts.*.gate_proj/...` | 曾用 original `block_sparse_moe...w1` |
| `ignore_layers` | 量化时按 native 模块名子串匹配 | dense 用 `mlp.gate_up_proj` | 用 `mlp.gate_proj/up_proj`（融合名不匹配） |
| `config_groups.targets` | vLLM 加载时按 original 名匹配 | 转换脚本翻译成 `block_sparse_moe...gate_proj` | 转换脚本漏翻译 |

### 若 dense 修复后仍失败：隔离 MoE MXFP8
dense 修复后重量化重测。若仍 0，则问题在 **MoE MXFP8 的 vLLM 运行时路径**（而非配置/权重）：
- 先跑一版 `--avg_bits 4`（几乎无 MXFP8）确认整条链路正常；
- 再逐步提高 avg_bits 引入少量 MXFP8 MoE 层，二分定位；
- 若确系 vLLM MXFP8 MoE kernel 问题，退回全 MXFP4，或改用 W8A16 等 vLLM 更成熟的混合方案。

## 8. 参考代码位置

- 入口与参数：`auto_round/auto_scheme/gen_auto_scheme.py`（`AutoScheme`、`GenScheme`）
- 打分与分配：`auto_round/auto_scheme/delta_loss.py`
  （`get_score_for_scheme`、`AutoSchemeWrapperLinear`、`choose_bits_per_layer_with_path`、`gen_layer_config`）
- CLI 集成：`auto_round/cli/parser.py`、`auto_round/cli/main.py`
- 混合导出：`auto_round/export/export_to_llmcompressor/export_to_fp.py`
  （`_build_mixed_fp_quantization_config` → `"format": "mixed-precision"`）
- MoE unfuse：`auto_round/modeling/fused_moe/moe_experts_interface.py`
  （`prepare_model_for_moe_quantization`，把 3D 专家拆成 per-expert Linear 并拆 gate_up）
- native 实现：`transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py`
  （`MiniMaxM3VLExperts` 用 3D `nn.Parameter`；`MiniMaxM3VLSparseMoeBlock` 属性名 `mlp`）
- 转换脚本：`convert_native_to_original_schema.py`（native → original，供 vLLM）
- 官方文档：`docs/step_by_step_CN.md` 的「AutoScheme」章节

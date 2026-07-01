# AutoRound MoE 模块替换 与 vLLM 推理兼容性分析

> 分析对象  
> - 量化脚本：`/storage/lkk/m3/t_mxfp4.py`  
> - 运行日志：`/storage/lkk/m3/test.log`  
> - 本人产物：`/storage/lkk/m3/MiniMax-M3_autoround_rtn_mxfp4_llmc/MiniMax-M3-mxfp-w4g32`（MiniMax‑M3，MXFP4，llm_compressor 格式）  
> - 同事产物：`/storage/xinhe/zai-org/GLM-5.2-MXFP4`（GLM‑5.2，MXFP4，同样 auto-round 导出）  
> - 参考源码：`/storage/lkk/m3/auto-round/auto_round/modeling/fused_moe/*` 与 `vllm 0.22.1`

---

## 一句话结论

1. **日志里看到的 “MoE module 被替换” 是 auto-round 量化阶段在内存里做的临时改造**（把融合的 3D 专家权重拆成「每个 expert 一组 `nn.Linear`」），目的是让每个专家的权重能被独立量化。**它不会改变模型架构，也不会破坏权重的语义。**
2. **保存到磁盘的格式（compressed-tensors / `mxfp4-pack-quantized`）是标准的「每专家」布局**，和同事的 GLM‑5.2 产物**结构完全一致**，vLLM 的 compressed-tensors MoE 加载器本来就吃这种布局。所以 “MoE 被替换” **本身不会导致无法加载/推理**。
3. **你担心的 “MiniMax-M3 不能用 vLLM 推理” 的真正原因不在量化、也不在 MoE 替换，而在于：当前环境的 vLLM(0.22.1) 根本没有注册 `MiniMaxM3SparseForConditionalGeneration` 这个架构。** 而 GLM‑5.2 的架构 `GlmMoeDsaForCausalLM` 是被 vLLM 支持的，所以同事的模型能跑。

---

## 一、日志里发生了什么：为什么 MoE 模块被替换

关键日志片段：

```
replace_modules.py L120: Experts (before replacement) [model.language_model.layers.3.mlp.experts] (MiniMaxM3VLExperts):
MiniMaxM3VLExperts()
...
moe_experts_interface.py L655: [MoE Prep] Unfused 57 MOE experts modules
replace_modules.py L93: Prepared 57 MOE modules for quantization
replace_modules.py L120: Experts (after replacement) [model.language_model.layers.3.mlp.experts] (MiniMaxM3VLExperts):
MiniMaxM3VLExperts(
  (0-127): 128 x _ExpertContainer(
    (down_proj): Linear(in_features=3072, out_features=6144, bias=False)
    (gate_proj): Linear(in_features=6144, out_features=3072, bias=False)
    (up_proj):   Linear(in_features=6144, out_features=3072, bias=False)
  )
)
```

### 1.1 原因：融合(fused)的专家权重无法被逐专家量化

像 MiniMax‑M3 / Qwen3‑MoE / GLM 这类新版 transformers MoE，专家权重在内存里是**融合成一个 3D 大张量**存放的，例如：

- `experts.gate_up_proj` 形状 `(num_experts, hidden, 2*intermediate)`
- `experts.down_proj` 形状 `(num_experts, intermediate, hidden)`

forward 时用一个 “grouped/批量 matmul” 一次算完所有专家。这种 3D 融合参数**不是 `nn.Linear`，量化器无法对每个专家分别统计 scale / 打包权重**。

因此 auto-round 在量化前调用 `prepare_model_for_moe_quantization()`（`moe_experts_interface.py`）做 **un-fuse（解融合）**：

- `_detect_expert_projections()`：识别 3D 专家参数（`gate_up_proj/gate_proj/up_proj/down_proj` 等）。
- Phase 1：把融合的 `gate_up_proj` 沿输出维 `chunk` 拆成 `gate_proj` + `up_proj`。
- Phase 2：`_unfuse_single_projection()` 把每个 3D 参数沿 expert 维 `unbind`，切成 `num_experts` 个 2D 切片，分别塞进 `nn.Linear`。
- Phase 3：把每个专家的 `gate_proj/up_proj/down_proj` 组装成一个编号子模块 `_ExpertContainer`，挂成 `experts.0 / experts.1 / ...`。
- 同时把 forward 切换成 `linear_loop_experts_forward`（逐专家循环跑 `nn.Linear`），并设 `config._experts_implementation = "linear_loop"`，让量化期间逻辑等价、数值不变。

> 这就是日志里 `MiniMaxM3VLExperts()` → `128 x _ExpertContainer(...)` 的来历。它**只是把“一个大融合张量”重排成“128 个标准 Linear”**，权重数值一一对应，没有任何丢失或改写。

### 1.2 这是 “借用” llm-compressor 的同款思路

`replace_modules.py` 顶部注释明确写了它改编自 llm-compressor 的 `moe_context.py`。也就是说 **auto-round 和 llm-compressor 对 MoE 的处理思路是一致的**——临时拆专家、量化、再以「每专家」格式落盘。这点很重要：它保证了产物落在 compressed-tensors 生态的标准格式上。

### 1.3 日志里其它几条信息

- `Using predefined ignore_layers: ...layers.[3-59].mlp.gate`：把 MoE 路由器 `mlp.gate` 排除在量化外（路由器很小且精度敏感），符合预期。
- `MoE layer detected: optimized RTN is disabled ...`：MoE 层走的是普通 RTN（`iters=0` 是 RTN 模式），只是关掉了 opt-RTN 加速，不影响正确性。
- 几条 `'visual' / 'linear_attn' / 'shared_expert_gate' ... does not match any supported layers` 的 WARNING：你 `fp_layers` 里写的这些名字在该模型里没有完全对应的层名，auto-round 忽略它们即可，不是错误。

---

## 二、落盘后的格式：和 GLM‑5.2 完全同构

虽然内存里专家被 un-fuse 成 128 个 Linear，但**真正写到 safetensors 里的是标准 compressed-tensors「每专家」布局**。两个模型实际的 index key 对比：

**MiniMax‑M3（本人产物）`model.safetensors.index.json`：**
```
language_model.model.layers.3.mlp.experts.0.down_proj.weight_packed
language_model.model.layers.3.mlp.experts.0.down_proj.weight_scale
language_model.model.layers.3.mlp.experts.0.gate_proj.weight_packed
language_model.model.layers.3.mlp.experts.0.gate_proj.weight_scale
language_model.model.layers.3.mlp.experts.0.up_proj.weight_packed
language_model.model.layers.3.mlp.experts.0.up_proj.weight_scale
... (每个 expert 一组)
```

**GLM‑5.2（同事产物）`model.safetensors.index.json`：**
```
model.layers.10.mlp.experts.0.down_proj.weight_packed
model.layers.10.mlp.experts.0.down_proj.weight_scale
model.layers.10.mlp.experts.0.gate_proj.weight_packed
model.layers.10.mlp.experts.0.gate_proj.weight_scale
model.layers.10.mlp.experts.0.up_proj.weight_packed
model.layers.10.mlp.experts.0.up_proj.weight_scale
model.layers.10.mlp.shared_experts.*.weight_packed / weight_scale
```

**两者的专家权重布局逐字段一致**：`experts.{i}.{gate,up,down}_proj.{weight_packed, weight_scale}`。

量化元数据也同构（`quantization_config.json` / `config.json`）：

| 字段 | MiniMax‑M3 | GLM‑5.2 |
|---|---|---|
| `quant_method` | `compressed-tensors` | `compressed-tensors` |
| `format` | `mxfp4-pack-quantized` | `mxfp4-pack-quantized` |
| weights | num_bits=4, type=float, group_size=32, symmetric | 同 |
| input_activations | 4-bit float, group_size=32, dynamic | 同 |

> 也就是说：**从“量化格式”和“MoE 专家存储格式”这两个维度看，你的 MiniMax‑M3 产物和能正常推理的 GLM‑5.2 产物是同一类东西**。auto-round 的 MoE 替换没有把你的模型搞成奇怪的非标准结构。

### 2.1 vLLM 端如何吃这种格式（确认兼容）

vLLM 0.22.1 内已具备对应的加载器：

- `compressed_tensors/compressed_tensors.py` 识别 `_is_mxfp4(...)` 并走 `CompressedTensorsW4A4Mxfp4`。
- `compressed_tensors_moe/compressed_tensors_moe_w4a4_mxfp4.py` 提供 `CompressedTensorsW4A4Mxfp4MoEMethod`，专门处理 `mxfp4-pack-quantized` 的 MoE。
- vLLM 的 `FusedMoE` 层有自己的 `weight_loader`，会把 checkpoint 里**每专家**的 `experts.{i}.gate_proj/up_proj/down_proj` 再**重新融合**进它内部的 fused 参数。

所以 “auto-round 把专家拆开” 和 “vLLM 把专家融合” 是**互逆且匹配**的：磁盘上是「每专家」标准格式，vLLM 加载时自己再 fuse。**这条链路对 GLM‑5.2 成立，对 MiniMax‑M3 在格式层面同样成立。**

---

## 三、那为什么 GLM‑5.2 能跑、而 MiniMax‑M3 可能跑不了？

差别**不在量化、不在 MoE 替换**，而在 **vLLM 是否实现了该模型架构**。

查 `vllm/model_executor/models/registry.py`（vLLM 0.22.1）：

**GLM‑5.2 —— 支持：**
- `config.json` → `architectures: ["GlmMoeDsaForCausalLM"]`，`model_type: glm_moe_dsa`
- registry 第 124 行：
  ```
  "GlmMoeDsaForCausalLM": ("deepseek_v2", "GlmMoeDsaForCausalLM"),
  ```
  → vLLM 用 `deepseek_v2` 的实现来跑它，**架构被支持**，所以同事的量化模型能正常加载推理。

**MiniMax‑M3 —— 不支持：**
- `config.json` → `architectures: ["MiniMaxM3SparseForConditionalGeneration"]`，`model_type: minimax_m3_vl`，且带 `auto_map`（自定义建模代码）。
- registry 里 MiniMax 系列只有：
  ```
  MiniMaxForCausalLM / MiniMaxText01ForCausalLM / MiniMaxM1ForCausalLM
  MiniMaxM2ForCausalLM / MiniMaxVL01ForConditionalGeneration
  ```
  **没有 `MiniMaxM3SparseForConditionalGeneration`，也没有 `minimax_m3_vl` 对应实现**，`model_executor/models/` 下也只有 `minimax_m2.py / minimax_text_01.py / minimax_vl_01.py`。

> 结论：在当前这台机器的 vLLM 0.22.1 上，**MiniMax‑M3 这个架构本身就没被实现**。即便量化完全正确、MoE 格式完全标准，vLLM 也会在“识别架构”这一步就失败（典型报错类似 `Model architecture MiniMaxM3SparseForConditionalGeneration is not supported / not registered`）。它还是个多模态(VL)模型，vLLM 对其视觉塔等部分也需要专门实现。

---

## 四、对你最初担忧的直接回答

| 你的担忧 | 结论 |
|---|---|
| auto-round 把 MoE 部分 module 替换了 | ✅ 确有替换，但只是**量化期内存内的 un-fuse**（3D 融合张量 → 每专家 `nn.Linear`），数值等价，不改架构 |
| 替换会不会导致保存的模型结构异常 | ❌ 不会。落盘是**标准 compressed-tensors「每专家」格式**，与可正常推理的 GLM‑5.2 产物逐字段同构 |
| llm_compressor / compressed-tensors 格式还能不能用 vLLM 推理 | ✅ 格式层面完全能（vLLM 有 `mxfp4-pack-quantized` 的 CT-MoE 加载器）。**能不能跑取决于架构是否被 vLLM 支持，而不是取决于量化/MoE 替换** |
| 为什么 GLM‑5.2 能跑 | 因为 `GlmMoeDsaForCausalLM` 在 vLLM registry 中（映射到 deepseek_v2 实现） |
| 为什么 MiniMax‑M3 可能跑不了 | 因为 `MiniMaxM3SparseForConditionalGeneration` / `minimax_m3_vl` **当前 vLLM 0.22.1 未注册/未实现**，与量化无关 |

---

## 五、建议

1. **先验证“架构支持”，再谈量化**：用很小的输入直接 `vllm serve` 或 `LLM(...)` 试加载 MiniMax‑M3 量化目录。若报 `architecture ... not supported`，即印证第三节结论。
2. **升级 / 切换 vLLM**：到 vLLM 官方仓库确认是否已有 `MiniMaxM3` / `minimax_m3_vl` 支持（main 分支或更高 release）。若已合入，升级 vLLM 即可；若尚无，需要等待官方实现或自行适配（实现 `FusedMoE` + 视觉塔 + 权重映射）。
3. **若只想验证量化正确性**（与 vLLM 解耦）：可用 transformers + compressed-tensors/llm-compressor 加载该目录做前向/简单生成，验证 MoE 权重确实可正常加载、数值合理。这能把“量化是否成功”和“vLLM 是否支持该架构”两个问题分开判断。
4. **顺带提醒一个小副作用（不影响加载，仅影响精度覆盖）**：你 `fp_layers` 里写了 `mlp.gate`，落盘 ignore 用的是正则 `re:.*mlp\.gate.*`。这个正则的 `mlp.gate` 是子串匹配，会**连带把前 3 层 dense MLP 的 `mlp.gate_up_proj` 也排除在量化外**（日志中 `layers.0/1/2.mlp.gate_up_proj` 出现在 ignore 列表即此原因）。它不会影响 vLLM 加载，只是这几层没被量化。专家里的 `experts.N.gate_proj` 不含 `mlp.gate` 子串，未被误伤，无需担心。另外 `re:.*layers\.[3-59]\.` 这种写法里的 `[3-59]` 是个写歪的字符类（等价于 3、4、5、9 单字符，并非 3~59 的范围），但因为 `re:.*mlp\.gate.*` 已经覆盖了所有 gate，所以实际效果无碍——后续如果想用类似正则按层号精确控制，要改成 `layers\.([3-9]|[1-5][0-9])\.` 这类写法。

---

## 附：复现/核对用的关键命令

```bash
# 1) 看产物的专家 key 布局
python - <<'PY'
import json
d=json.load(open('/storage/lkk/m3/MiniMax-M3_autoround_rtn_mxfp4_llmc/MiniMax-M3-mxfp-w4g32/model.safetensors.index.json'))
print([k for k in d['weight_map'] if 'layers.3.mlp.experts.0' in k])
PY

# 2) 看 vLLM 是否注册了该架构
grep -n "MiniMaxM3\|minimax_m3\|GlmMoeDsa" \
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/registry.py

# 3) 看 vLLM 是否支持 mxfp4-pack-quantized 的 MoE
ls /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/
```

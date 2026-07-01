# MiniMax-M3 MXFP4 第三个报错：融合层 gate_up_proj 量化方案不一致

> 本文续接前两篇：
> - `vllm_vision_quant_bug_analysis.md`（第一次 `fc1` KeyError，改 config 绕过）
> - `vllm_qkv_fix_and_amd_comparison.md`（第二次 `qkv_proj` KeyError，改 nvidia 版源码对齐 amd 版）
>
> 视觉塔和注意力融合层修好后，模型加载推进到了**语言模型**阶段，暴露出第三个错误。
> **和前两个不同：这一次是量化产物本身的问题，不是 vLLM 的 bug。**

---

## 0. 结论速览

- **报错**：
  ```
  ValueError: Found a different quantization schemes for ['gate_proj', 'up_proj']
  in language_model.model.layers.0.mlp.gate_up_proj. vLLM requires all to use the same scheme.
  ```
- **根因**：量化命令里的 `--ignore_layers mlp.gate` 本意是保护 MoE 路由器，但：
  - 路由器真名是 `block_sparse_moe.gate`，`mlp.gate` **匹配不到它**；
  - `mlp.gate` 却**子串命中**了 dense 层的 `mlp.gate_proj`（layer 0/1/2 共 3 个），
    把它们错误地排除量化。
  - 于是这 3 层 `gate_proj` 未量化、`up_proj` 已量化。vLLM 把二者融合成 `gate_up_proj`，
    要求两个分片量化方案一致 → 不一致 → `ValueError`。
- **修复**：**必须重新量化**（config 后处理无解）。把 `--ignore_layers` 里的
  `mlp.gate` 改成 `block_sparse_moe.gate`，重跑 auto-round。

---

## 1. 报错本身：融合层要求分片方案一致

vLLM 把 MLP 的 `gate_proj` 和 `up_proj` **融合**成一个 `gate_up_proj`
（`MergedColumnParallelLinear`）。融合的前提是两个分片用**同一种量化方案**。

判断逻辑在 `vllm/model_executor/layers/quantization/compressed_tensors/utils.py`
的 `should_ignore_layer`（前一篇分析 qkv 时引用过的同一个函数）：

```python
if proj_name in fused_mapping and layer_name not in ignore:
    shard_proj_names = fused_mapping[proj_name]        # ["gate_proj","up_proj"]
    shard_names = [layer_name.replace(proj_name, s) for s in shard_proj_names]

    should_ignore_layer = None
    for shard_name in shard_names:
        should_ignore_shard = check_equal_or_regex_match(shard_name, ignore)
        if should_ignore_layer is None:
            should_ignore_layer = should_ignore_shard      # 记录第一个分片
        elif should_ignore_shard != should_ignore_layer:   # 后续分片与第一个不一致
            raise ValueError(                              # ← 就是这里抛错
                f"Found a different quantization schemes for "
                f"{shard_proj_names} in {layer_name}. vLLM "
                "requires all to use the same scheme."
            )
```

- `gate_proj` → 在 ignore（未量化）→ `should_ignore = True`
- `up_proj`   → 不在 ignore（量化）→ `should_ignore = False`
- 两者不一致 → **ValueError**。

> 注意这和前两个 KeyError 是**不同的失败点**：前两个是"名字对不上、找不到参数"，
> 这个是"融合层两半的量化状态相互矛盾"，vLLM 主动检测并报错。

---

## 2. 模型结构：哪些是 dense、哪些是 MoE

从 `config.json` 的 `moe_layer_freq` 和 checkpoint 权重名核对得到：

```
num_hidden_layers = 60
moe_layer_freq = [0,0,0, 1,1,1,...,1]    # 前 3 层是 0=dense，其余 57 层是 1=MoE
```

| 层号 | 类型 | MLP 结构（checkpoint 里的模块名） |
|------|------|-----------------------------------|
| 0,1,2 | **dense** | `mlp.gate_proj` / `mlp.up_proj` / `mlp.down_proj` |
| 3–59 | **MoE** | `block_sparse_moe.gate`（路由器）<br>`block_sparse_moe.experts.N.w1/w2/w3`（128 个专家）<br>`block_sparse_moe.shared_experts.gate_proj/up_proj/down_proj` |

**关键点**：MoE 路由器叫 `block_sparse_moe.gate`，**dense 层的门控投影叫 `mlp.gate_proj`**。
两者名字里都有 `gate`，但完全是不同的东西。

---

## 3. `mlp.gate` 这个 ignore token 到底匹配了什么

量化命令（来自你最初的记录）：
```bash
auto-round ../MiniMax-M3/ --model_free --scheme MXFP4 \
  --ignore_layers visual,lm_head,mlp.gate,linear_attn,shared_expert_gate,embed_tokens,self_attn \
  --format llm_compressor --output_dir "./MiniMax-M3-MXFP4"
```

auto-round 的 `--ignore_layers` 是**子串匹配**。逐个看 `mlp.gate` 命中了谁：

| 目标模块 | 是否含子串 `mlp.gate` | 结果 |
|----------|----------------------|------|
| `...layers.0.mlp.gate_proj`（dense 门控） | ✅ 含 `mlp.gate` | **被误 ignore** ❌ |
| `...layers.3.block_sparse_moe.gate`（MoE 路由器） | ❌ 不含 `mlp.gate` | 未被此 token 命中 |

也就是说：
- `mlp.gate` **没能**保护 MoE 路由器（本意落空）；
- `mlp.gate` **反而**误伤了 dense 层的 `gate_proj`。

**那 MoE 路由器为什么现在还是被保护的？** 核对 config 后发现 ignore 里有 57 条
`block_sparse_moe.gate`（layer 3–59 各一条）。它们不是 `mlp.gate` 匹配来的，而是
**auto-round 对 MoE router/gate 的默认保护规则**自动加入的。所以：
- 路由器保护 = auto-round 默认行为（与 `mlp.gate` token 无关）；
- `mlp.gate` token 是**多余且有害**的，唯一效果就是把 3 个 dense `gate_proj` 拉进了 ignore。

---

## 4. 已核对的证据

| 模块 | 存储格式 | 是否在 ignore | 说明 |
|------|----------|---------------|------|
| dense `mlp.gate_proj`（layer 0/1/2，3 个） | 普通 `.weight` | ✅ 在 | **错误**：应被量化 |
| dense `mlp.up_proj`（layer 0/1/2） | 量化 `weight_packed` | ❌ 不在 | 已量化 |
| MoE 路由器 `block_sparse_moe.gate`（57 个） | 普通 `.weight` | ✅ 在 | 正确保护 |
| MoE `shared_experts.gate_proj / up_proj` | 量化 `weight_packed` | ❌ 不在 | 两半一致，无问题 |
| MoE `experts.N.w1/w2/w3` | 量化 `weight_packed` | ❌ 不在 | 已量化 |
| 所有 `self_attn.*` | 普通 `.weight` | ✅ 在 | `self_attn` token 全量保护，融合 qkv 一致 |

只有 **dense 层的 `gate_proj`（3 个）** 是"半量化"的坏点，正好落在融合对 `gate_up_proj` 上。

---

## 5. 为什么这次 config 后处理无解，必须重量化

融合层 `gate_up_proj` 要求 `gate_proj` 与 `up_proj` **方案一致**。当前 checkpoint：
- `gate_proj`：普通权重 `.weight`
- `up_proj`：打包权重 `weight_packed` + `weight_scale`

两种"改 config 让它们一致"的方向都走不通：

| 尝试 | 后果 |
|------|------|
| 把 `gate_proj` 从 ignore 移除（当作已量化） | vLLM 建量化层，去找 `gate_proj.weight_packed` → checkpoint 里只有 `.weight` → **KeyError** |
| 把 `up_proj` 加入 ignore（当作未量化） | vLLM 建普通层，去找 `up_proj.weight` → checkpoint 里只有 `weight_packed` → **KeyError** |

**缺的那份数据（gate_proj 的量化版本 / up_proj 的原始版本）在 checkpoint 里根本不存在**，
所以无法靠后处理拼出来，只能重新量化生成。

（与前两个错误的本质区别：前两个是"数据都在，只是 vLLM 名字匹配逻辑有 bug"，改代码即可；
这个是"量化产物本身缺了一致的数据"，只能重做。）

---

## 6. 修复：改量化命令后重跑

把 `--ignore_layers` 里的 `mlp.gate` 换成真正的路由器名 `block_sparse_moe.gate`：

```bash
auto-round ../MiniMax-M3/ --model_free --scheme MXFP4 \
  --ignore_layers visual,lm_head,block_sparse_moe.gate,linear_attn,shared_expert_gate,embed_tokens,self_attn \
  --format llm_compressor --output_dir "./MiniMax-M3-MXFP4"
```

改动效果：
- `block_sparse_moe.gate` 精确指向 MoE 路由器，**不会**子串命中 `mlp.gate_proj`；
- dense 层的 `gate_proj` 不再被 ignore → 和 `up_proj` 一样被量化 → 融合层方案一致 → 报错消失；
- MoE 路由器仍被保护（token 显式保护 + auto-round 默认，双保险）。

> 备注：`block_sparse_moe.gate` 作为子串也不会误伤 `block_sparse_moe.shared_experts.gate_proj`
> （后者在 `block_sparse_moe.` 之后是 `shared_experts`，不含连续子串 `block_sparse_moe.gate`）。

### 重量化后的自查（应输出 0）
```bash
python3 -c "
import json
c=json.load(open('MiniMax-M3-MXFP4/config.json'))
ig=set(c['quantization_config']['ignore'])
bad=[x for x in ig if x.endswith('mlp.gate_proj')]
print('dense gate_proj 仍被误 ignore 的数量:', len(bad), bad)
"
```

---

## 7. 三个报错的总览

| # | 报错 | 层 | 性质 | 修复 |
|---|------|----|----|------|
| 1 | KeyError `...fc1.weight` | 视觉塔 MLP | vLLM bug：mapper 键带结尾点，ignore 没被重映射 | 改 nvidia 源码 mapper（去结尾点）／曾先手改 config |
| 2 | KeyError `...qkv_proj.weight` | 视觉/语言注意力 | vLLM bug：顶层类缺 `packed_modules_mapping`，融合层 ignore 匹配失效 | 改 nvidia 源码，给两个类补 `packed_modules_mapping` |
| 3 | ValueError `gate_up_proj 方案不一致` | dense 层 MLP | **量化产物 bug**：`mlp.gate` 误伤 dense `gate_proj` | **重量化**：`mlp.gate` → `block_sparse_moe.gate` |

前两个通过对齐 amd 版源码根治；第三个是量化侧的 ignore 匹配问题，必须重跑量化。

---

## 8. 经验小结

1. **区分是"名字对不上"还是"量化状态矛盾"**：
   - KeyError → 通常是命名/映射问题，数据还在，多半能改代码或改 config 解决；
   - "different quantization schemes" 的 ValueError → 融合层两半状态矛盾，多半要重量化。
2. **量化 ignore 用子串匹配时，短 token 容易误伤**：`mlp.gate` 命中了 `mlp.gate_proj`。
   写 ignore 时尽量用能唯一定位的完整模块名（如 `block_sparse_moe.gate`），或用精确/正则匹配。
3. **注意 dense 与 MoE 层的命名差异**：本模型前 3 层 dense 用 `mlp.gate_proj`，
   其余 MoE 用 `block_sparse_moe.gate` + experts，二者门控名字相似但含义完全不同。

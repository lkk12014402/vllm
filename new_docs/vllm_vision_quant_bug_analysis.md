# MiniMax-M3 MXFP4 量化模型在 vLLM 加载报 KeyError 的原因分析

## 0. 结论速览

- **报错**：`KeyError: 'vision_model.encoder.layers.0.fc1.weight'`，导致 vLLM EngineCore 启动失败。
- **根因**：vLLM 侧的自定义 MiniMax-M3 模型把视觉塔里 `mlp.fc1 / mlp.fc2` **重命名**成了 `fc1 / fc2`（去掉了中间的 `mlp` 子模块）。这个重命名只在**加载权重**时通过 `WeightsMapper` 生效；而 compressed-tensors 判断"这一层是否要跳过量化（ignore）"时**不走** `WeightsMapper`，用的是 vLLM 内部名 `...fc1`，去 config.json 的 ignore 列表里找 `...mlp.fc1`，**匹配不上** → vLLM 误以为该层需要量化 → 把 `fc1` 建成量化层（参数是 `weight_packed`，没有 `.weight`）→ 加载 checkpoint 里的普通 `.weight` 时找不到键 → KeyError。
- **和量化脚本无关**：视觉层在 checkpoint 里本来就是未量化的普通权重，重跑 `auto-round` 得到的 ignore 列表命名一模一样，无法解决。

---

## 1. 什么叫"vLLM 侧自定义模型内部重命名"

一个 HF 模型有两套东西：

1. **checkpoint 里的权重名**（safetensors 里的 key），例如：
   ```
   vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight
   ```
   这是原始 HF 模型的模块结构：`encoder.layers[0]` 下有一个 `mlp` 子模块，`mlp` 里才有 `fc1`。

2. **vLLM 为了推理性能自己重写了一份模型代码**（`vllm/models/minimax_m3/...`）。vLLM 的模块树**不一定**和 HF 一样。对 MiniMax-M3 的视觉塔，vLLM 把 `fc1 / fc2` 直接挂在了 `EncoderLayer` 上，**没有** `mlp` 这一层。

看 vLLM 的代码 `vllm/models/minimax_m3/common/vision_tower.py`，`MiniMaxVLEncoderLayer.__init__`：

```python
class MiniMaxVLEncoderLayer(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        ...
        self.fc1 = ColumnParallelLinear(
            config.hidden_size, config.intermediate_size, bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",          # ← 注意：prefix 是 ...layers.0.fc1，没有 .mlp.
        )
        self.act = get_act_fn(...)
        self.fc2 = RowParallelLinear(
            ...,
            prefix=f"{prefix}.fc2",          # ← 同理 ...layers.0.fc2
        )
```

所以：
- **HF checkpoint 名**：`...encoder.layers.0.mlp.fc1.weight`
- **vLLM 内部模块名（prefix）**：`...encoder.layers.0.fc1`

两者差一个 `.mlp.`。这就是"内部重命名"。

---

## 2. 重命名是靠 `WeightsMapper` 在"加载权重时"抹平的

vLLM 知道自己的模块名和 checkpoint 名对不上，所以在加载时用一个映射器把 checkpoint 名改成 vLLM 名。见 `vllm/models/minimax_m3/nvidia/model.py`：

```python
hf_to_vllm_mapper = WeightsMapper(
    orig_to_new_prefix={
        "multi_modal_projector.": "vision_tower.multi_modal_projector.",
        "patch_merge_mlp.":       "vision_tower.patch_merge_mlp.",
    },
    orig_to_new_substr={
        ".mlp.fc1.": ".fc1.",     # ← 把 checkpoint 的 .mlp.fc1. 改成 vLLM 的 .fc1.
        ".mlp.fc2.": ".fc2.",     # ← 同理
    },
)
```

加载时（`load_weights`）会用它把权重名转好，再去 `params_dict` 里找对应参数：

```python
# model.py
return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

# vision_tower.py 的 load_weights（第 ~739 行）
params_dict = dict(self.named_parameters(remove_duplicate=False))
for name, loaded_weight in weights:      # name 已被 mapper 改成 vLLM 名，如 ...layers.0.fc1.weight
    ...
    param = params_dict[name]            # ← 第 761 行，就是这里 KeyError
```

**关键点**：`WeightsMapper` 只在**加载权重**这条路径上被使用。它负责"名字转换"，但它管不到"量化配置匹配"。

---

## 3. 量化的 ignore 匹配走的是另一条路，且不用 mapper

模型在**构建每一层**时，会问 compressed-tensors："这一层要不要量化？"。判断逻辑在 `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py`：

```python
def get_quant_method(self, layer, prefix):
    ...
    quant_scheme = self.get_scheme(layer=layer, layer_name=prefix)   # ← 传进去的是 vLLM 的 prefix
```

```python
def get_scheme_dict(self, layer, layer_name=None):
    # TODO (@kylesayrs): support ignore module names with ct matching utils   ← vLLM 自己也标了 TODO
    if should_ignore_layer(
        layer_name, ignore=self.ignore, fused_mapping=self.packed_modules_mapping
    ):
        return None      # 命中 ignore → 不量化，建成普通层（参数是 .weight）
    ...                  # 没命中 → 量化，建成量化层（参数是 .weight_packed / .weight_scale）
```

再看 `should_ignore_layer`（`compressed_tensors/utils.py`）：

```python
def should_ignore_layer(layer_name, ignore=(), fused_mapping=...):
    ...
    # 对非融合层，直接把 vLLM 的 layer_name 和 ignore 列表做字符串/正则比较
    should_ignore_layer = check_equal_or_regex_match(
        layer_name=layer_name,     # = "vision_tower.vision_model.encoder.layers.0.fc1"
        targets=ignore,            # 里面是 "vision_tower.vision_model.encoder.layers.0.mlp.fc1"
    )
    return should_ignore_layer
```

**注意这里传入的 `layer_name` 是 vLLM 的 `prefix`（`...fc1`），而 `ignore` 是从 config.json 原样读出来的（`...mlp.fc1`）。中间没有任何 `WeightsMapper` 的介入。**

- config.json 的 ignore 列表由 auto-round/llm_compressor 生成，用的是**源 HF 模型真实模块名**：`...encoder.layers.0.mlp.fc1`。
- vLLM 拿 `...encoder.layers.0.fc1` 去比对。
- `...fc1` ≠ `...mlp.fc1` → **匹配失败** → `should_ignore_layer` 返回 `False` → 该层被当成"要量化"。

---

## 4. 完整因果链

```
HF checkpoint（源模型真实结构）
   vision_tower.vision_model.encoder.layers.0.mlp.fc1   ← 有 .mlp.
        │
        ├─(a) auto-round 量化时，视觉层被 ignore（不量化）
        │      → checkpoint 里存的是普通 .weight（已确认）
        │      → 但 config.json 的 ignore 列表用的还是 HF 名 “...mlp.fc1”
        │
        └─(b) vLLM 自定义模型把它重命名成 “...fc1”（去掉 .mlp.）
               │
               ├─ 加载权重路径：WeightsMapper 把 checkpoint 名 .mlp.fc1 → .fc1  ✅ 正常
               │
               └─ 量化 ignore 判断路径：拿 vLLM 名 “...fc1” 去 ignore 列表找 “...mlp.fc1”
                     → 找不到 ❌ → 认为该层要量化
                     → fc1 建成量化层，参数只有 weight_packed / weight_scale，没有 .weight
                          │
                          ▼
   加载时 mapper 把权重名转成 “...fc1.weight”，去 params_dict 里找
   params_dict 里只有 “...fc1.weight_packed”，没有 “...fc1.weight”
        → KeyError: 'vision_model.encoder.layers.0.fc1.weight'   💥
```

一句话：**同一个"重命名"，加载权重时被 WeightsMapper 处理了，判断是否量化时却没被处理，两条路径不一致，导致视觉层被错误地当成需要量化，从而参数名对不上。**

---

## 5. 证据（在你机器上已核对）

**checkpoint 里视觉层是未量化的普通权重**（`.weight`，不是 `weight_packed`）：
```
vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight   ← 普通
vision_tower.vision_model.encoder.layers.0.mlp.fc2.weight   ← 普通
```
对比语言层（已量化）：
```
language_model.model.layers.1.mlp.down_proj.weight_packed
language_model.model.layers.1.mlp.down_proj.weight_scale
```

**config.json 的 ignore 列表用的是带 `.mlp.` 的 HF 名**（共 64 条 fc1/fc2）：
```
vision_tower.vision_model.encoder.layers.0.mlp.fc1
vision_tower.vision_model.encoder.layers.0.mlp.fc2
...
```

**vLLM 里对应层的 prefix 是 `...fc1 / ...fc2`（无 `.mlp.`）** —— 见 `vision_tower.py` 的 `prefix=f"{prefix}.fc1"`。

---

## 6. 为什么"改量化脚本、重新生成模型"解决不了

1. 视觉层现在**本来就没被量化**，问题不在量化与否。
2. config.json 的 ignore 名字由 auto-round 按**源模型真实模块名**写死（`...mlp.fc1`），无论 `--ignore_layers` 参数写成什么，都改变不了这个命名。
3. 错位发生在 **vLLM 侧**（内部重命名 + ignore 匹配不走 mapper），量化脚本够不到这里。

所以重跑只会得到一模一样的 ignore 列表和一模一样的报错。

---

## 7. 解决方案

### 方案 A（推荐，最快）：后处理改 config.json 的 ignore 名字
把 ignore 列表里视觉塔的 `...mlp.fc1 / ...mlp.fc2` 改成 vLLM 用的 `...fc1 / ...fc2`（共 64 条），让 ignore 匹配命中，vLLM 就会把这两层建成普通未量化层。

```python
import json, shutil
p = "/storage/lkk/m3/MiniMax-M3-MXFP4/config.json"
shutil.copy(p, p + ".bak")                       # 备份
c = json.load(open(p))
ig = c["quantization_config"]["ignore"]
c["quantization_config"]["ignore"] = [
    x.replace(".mlp.fc1", ".fc1").replace(".mlp.fc2", ".fc2") if "vision" in x else x
    for x in ig
]
json.dump(c, open(p, "w"), indent=2, ensure_ascii=False)
```
> 缺点：改后的 config 名字偏离 HF 原始命名，若还要用 transformers 等其他工具直接读这个目录，可能需要保留原备份。用于 vLLM 推理没问题。

### 方案 B（更彻底）：改 vLLM 源码
让 compressed-tensors 的 ignore 匹配也套用 `hf_to_vllm_mapper`（正是 `get_scheme_dict` 里那句 `# TODO: support ignore module names` 想做的事）。改动在框架层，而且这是个自定义 fork，维护成本更高。

### 方案 C：让 vLLM 模型命名与 HF 对齐
把 `vision_tower.py` 里 `fc1/fc2` 包回一个 `mlp` 子模块，并去掉 `WeightsMapper` 里 `.mlp.fc1.→.fc1.` 的替换。改动大、易引入回归，不推荐。

---

## 8. 附：那些 `_qutlass_C ... undefined symbol` 的 WARNING

日志开头大量：
```
WARNING ... Failed to import from vllm._qutlass_C: ImportError('.../_qutlass_C.abi3.so: undefined symbol: ...')
```
这些只是**警告**，vLLM 会回退到别的实现，**不是本次崩溃的原因**。真正致命的是第 3 节的 KeyError。它反映 `_qutlass_C` 这个扩展和当前 torch 的 ABI 不一致（多半是 torch 升级后没重新编译扩展），有精力可以顺手重编，但和本问题无关。

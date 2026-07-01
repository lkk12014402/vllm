# MiniMax-M3 MXFP4 在 vLLM(nvidia 版)加载报错的完整分析与源码修复

> 本文续接 `vllm_vision_quant_bug_analysis.md`。上一篇讲清了 `fc1/fc2` 的 KeyError；
> 本文覆盖第二个报错 `qkv_proj` 的 KeyError，并给出**根因级源码修复**，同时对照
> vLLM 里已经写对了的 **amd 版**模型，说明 nvidia 版差在哪。

---

## 0. 结论速览

- **第二次报错**：`KeyError: 'vision_model.encoder.layers.0.self_attn.qkv_proj.weight'`
- **根因（一句话）**：你用的 **nvidia 版** MiniMax-M3 模型代码是个不完整的 fork，缺了两样东西，
  导致 compressed-tensors 无法把 config.json 里的 ignore 列表正确匹配到 vLLM 的内部模块名：
  1. **顶层模型类没有定义 `packed_modules_mapping`** → 融合层 `qkv_proj` 的 ignore 匹配失效。
  2. **`hf_to_vllm_mapper` 的替换键带了结尾的点**（`.mlp.fc1.`）→ 无法命中 ignore 里的裸模块名
     `...mlp.fc1`，所以 `fc1/fc2` 的 ignore 也没被重映射（就是上一篇里我们只能手改 config 绕过的原因）。
- **对照组**：同目录下的 **amd 版**(`vllm/models/minimax_m3/amd/model.py`)这两处都是**对的**。
  修复方法就是把 nvidia 版对齐 amd 版。
- **修完后**：可以把上次手改的 config.json 还原成原始 HF 命名（已还原），模型目录保持标准。

---

## 1. 两次报错的关系

| | 第一次 (test.log) | 第二次 (test_2.log) |
|---|---|---|
| 报错键 | `...encoder.layers.0.fc1.weight` | `...encoder.layers.0.self_attn.qkv_proj.weight` |
| 层类型 | 普通层 `ColumnParallelLinear` | **融合层** `QKVParallelLinear` |
| ignore 匹配走的分支 | 直接名匹配 | **融合(fused)匹配**，依赖 `packed_modules_mapping` |
| 上次的手改 config 是否解决 | ✅ 解决了 fc1/fc2 | ❌ 解决不了 qkv（融合层是另一套逻辑） |

两次是**同一个病根**（ignore 匹配失效）的两种表现。手改 config 只治了普通层，融合层要靠
`packed_modules_mapping`，所以又冒出来 qkv 的报错。

---

## 2. 融合层为什么不一样：`qkv_proj` 的匹配逻辑

vLLM 为了性能，把注意力的 `q_proj / k_proj / v_proj` **融合**成一个 `qkv_proj`。
但 checkpoint 和 config.json 里是**没有** `qkv_proj` 的，只有分开的 `q_proj/k_proj/v_proj`。

看 `vllm/models/minimax_m3/common/vision_tower.py` 里视觉注意力的构造：

```python
self.qkv_proj = QKVParallelLinear(
    ...,
    prefix=f"{prefix}.qkv_proj",   # ← vLLM 内部名是融合后的 qkv_proj
)
```

而 config.json 的 ignore 列表里是（已核对）：
```
vision_tower.vision_model.encoder.layers.0.self_attn.q_proj
vision_tower.vision_model.encoder.layers.0.self_attn.k_proj
vision_tower.vision_model.encoder.layers.0.self_attn.v_proj
vision_tower.vision_model.encoder.layers.0.self_attn.out_proj
```
**没有 `qkv_proj`**。

vLLM 判断"融合层要不要 ignore"的逻辑在
`vllm/model_executor/layers/quantization/compressed_tensors/utils.py`：

```python
def should_ignore_layer(layer_name, ignore=(), fused_mapping=...):
    proj_name = layer_name.split(".")[-1]           # = "qkv_proj"

    # 关键：只有当 proj_name 在 fused_mapping 里，才会把 qkv_proj 拆成 q/k/v_proj 去逐个匹配
    if proj_name in fused_mapping and layer_name not in ignore:
        shard_proj_names = fused_mapping[proj_name]  # ["q_proj","k_proj","v_proj"]
        shard_names = [layer_name.replace(proj_name, s) for s in shard_proj_names]
        # → 生成 ...self_attn.q_proj / k_proj / v_proj，再逐个和 ignore 比对
        for shard_name in shard_names:
            should_ignore_shard = check_equal_or_regex_match(shard_name, ignore)
            ...
    else:
        # proj_name 不在 fused_mapping → 直接拿 "qkv_proj" 去 ignore 里找
        should_ignore_layer = check_equal_or_regex_match(layer_name, ignore)
    return should_ignore_layer
```

这里的 `fused_mapping` 就是 `self.packed_modules_mapping`（见 `compressed_tensors.py` 第 869 行
`should_ignore_layer(..., fused_mapping=self.packed_modules_mapping)`）。

- **如果 `packed_modules_mapping` 正常** = `{"qkv_proj": ["q_proj","k_proj","v_proj"], ...}`：
  `qkv_proj` 会被拆成 `q_proj/k_proj/v_proj`，逐个命中 ignore → 整层被 ignore（不量化）✅
- **如果 `packed_modules_mapping` 是空 `{}`**（当前 nvidia 版的情况）：
  `proj_name="qkv_proj"` 不在空字典里 → 走 else 分支 → 直接拿 `qkv_proj` 去 ignore 找 →
  ignore 里只有 `q_proj/k_proj/v_proj`，没有 `qkv_proj` → **匹配失败** → 该层被量化 →
  参数变成 `weight_packed`，加载 checkpoint 的 `qkv_proj.weight`（由 q/k/v 融合而来）时找不到 →
  **KeyError** 💥

---

## 3. `packed_modules_mapping` 为什么是空的

它的默认值是空字典，见 `vllm/model_executor/layers/quantization/base_config.py`：
```python
self.packed_modules_mapping: dict[str, list[str]] = dict()   # 第 90 行，默认空
```

vLLM 在加载模型时会尝试用**顶层模型类的类属性** `packed_modules_mapping` 去填充它，见
`vllm/model_executor/model_loader/utils.py`：
```python
def configure_quant_config(quant_config, model_class):
    if not issubclass(model_class, SupportsQuant):
        hf_to_vllm_mapper = getattr(model_class, "hf_to_vllm_mapper", None)
        packed_mapping   = getattr(model_class, "packed_modules_mapping", None)  # ← 读类属性

        if hf_to_vllm_mapper is not None:
            quant_config.apply_vllm_mapper(hf_to_vllm_mapper)   # 顺带把 mapper 应用到 ignore
        if packed_mapping is not None:
            quant_config.packed_modules_mapping = packed_mapping # ← 填进 quant_config
```

**问题**：nvidia 版的顶层类 `MiniMaxM3SparseForConditionalGeneration`
（以及它委托的 `MiniMaxM3SparseForCausalLM`）**根本没定义 `packed_modules_mapping` 这个类属性**，
`getattr(..., None)` 拿到 `None` → 不填充 → `quant_config.packed_modules_mapping` 保持空 `{}` →
第 2 节的融合匹配失效。

> 这也解释了为什么**语言模型的 self_attn 也是未量化的**（config 里同样把前几层 q/k/v_proj 放进了
> ignore），只是模型加载时视觉塔先加载、先崩，所以你先看到 vision 的报错。

---

## 4. `hf_to_vllm_mapper` 的"结尾点"bug（fc1/fc2 的根因）

上一篇我们发现：ignore 匹配**不走** mapper。这句话需要修正一下——其实 vLLM **是想走的**，
在 `configure_quant_config` 里调了 `quant_config.apply_vllm_mapper(hf_to_vllm_mapper)`，
它会把 ignore 列表里的名字用 mapper 转一遍。看
`compressed_tensors.py` 的 `apply_vllm_mapper`：

```python
def apply_vllm_mapper(self, hf_to_vllm_mapper):
    def _map_target(target):
        is_layer_path = "." in target and not target.startswith("re:")
        if is_layer_path:
            return hf_to_vllm_mapper._map_name(target)   # 用 mapper 转名字
        return target
    ...
    self.ignore = _apply_list(self.ignore)   # ← ignore 列表确实会被 mapper 处理
```

**那为什么 fc1 之前没被转成功？** 因为 nvidia 版原来的 mapper 写的是**带结尾点**的键：

```python
# nvidia 版原始（有问题）
orig_to_new_substr={
    ".mlp.fc1.": ".fc1.",   # ← 结尾有点
    ".mlp.fc2.": ".fc2.",
}
```

`_map_name` 做的是子串替换。它对：
- **权重名** `...encoder.layers.0.mlp.fc1.weight` → 含子串 `.mlp.fc1.`（后面跟着 `weight`），能替换 ✅
  （这就是加载权重时能对上的原因）
- **ignore 里的裸模块名** `...encoder.layers.0.mlp.fc1` → **结尾没有点**，不含子串 `.mlp.fc1.` →
  **替换不了** ❌ → ignore 里仍是 `...mlp.fc1`，而 vLLM 用 `...fc1` 去找 → 不匹配 → fc1 被量化 → KeyError。

**amd 版写的是不带结尾点的键**，两种场景都能命中：
```python
# amd 版（正确）
orig_to_new_substr={
    ".mlp.fc1": ".fc1",    # ← 无结尾点，权重名和裸模块名都能替换
    ".mlp.fc2": ".fc2",
}
```

---

## 5. amd 版长什么样（正确对照组）

文件：`vllm/models/minimax_m3/amd/model.py`

**(a) `MiniMaxM3SparseForCausalLM`（约 1051 行）——有 packed_modules_mapping：**
```python
class MiniMaxM3SparseForCausalLM(nn.Module, SupportsPP, SupportsEagle3):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
```

**(b) `MiniMaxM3SparseForConditionalGeneration`（约 1129 行）——有 packed_modules_mapping + 正确 mapper：**
```python
class MiniMaxM3SparseForConditionalGeneration(...):
    supports_encoder_tp_data = True

    packed_modules_mapping = {                          # ← nvidia 版缺这个
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "multi_modal_projector.": "vision_tower.multi_modal_projector.",
            "patch_merge_mlp.": "vision_tower.patch_merge_mlp.",
        },
        orig_to_new_substr={
            ".mlp.fc1": ".fc1",                         # ← 无结尾点（nvidia 版有结尾点）
            ".mlp.fc2": ".fc2",
        },
    )
```

**nvidia 版原始代码的差异（就是这两处 bug）：**
- `MiniMaxM3SparseForCausalLM`：**无** `packed_modules_mapping`
- `MiniMaxM3SparseForConditionalGeneration`：**无** `packed_modules_mapping`；
  且 `hf_to_vllm_mapper` 的键是 `.mlp.fc1.` / `.mlp.fc2.`（**带结尾点**）

---

## 6. 我做的源码修改（把 nvidia 版对齐 amd 版）

文件：`vllm/models/minimax_m3/nvidia/model.py`

### 修改 1：给 `MiniMaxM3SparseForCausalLM` 加 `packed_modules_mapping`
```python
class MiniMaxM3SparseForCausalLM(nn.Module, SupportsPP, SupportsEagle3):
    """MiniMax M3 (sparse/dense backbone) for causal language modeling."""

    packed_modules_mapping = {                    # ← 新增
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        ...
```

### 修改 2：给 `MiniMaxM3SparseForConditionalGeneration` 加 `packed_modules_mapping`
```python
    supports_encoder_tp_data = True

    packed_modules_mapping = {                    # ← 新增
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        ...
```

### 修改 3：把 mapper 的替换键去掉结尾点
```python
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "multi_modal_projector.": "vision_tower.multi_modal_projector.",
            "patch_merge_mlp.": "vision_tower.patch_merge_mlp.",
        },
        orig_to_new_substr={
            ".mlp.fc1": ".fc1",       # 原来是 ".mlp.fc1.": ".fc1."
            ".mlp.fc2": ".fc2",       # 原来是 ".mlp.fc2.": ".fc2."
        },
    )
```

> 为什么两个类都要加 `packed_modules_mapping`：`ConditionalGeneration` 是顶层 VL 入口，
> 它内部又通过 `init_vllm_registered_model(..., architectures=["MiniMaxM3SparseForCausalLM"])`
> 构建语言模型。视觉塔在 `ConditionalGeneration` 这层加载，语言塔在 `CausalLM` 那层加载，
> 两条路径的 `configure_quant_config` 各读各的类属性，所以两个类都得有，才能同时覆盖视觉和语言的融合层。

### 附带：还原 config.json
源码修复后，mapper 会正确地把 ignore 里的 `...mlp.fc1` 重映射成 `...fc1`，融合层也能靠
`packed_modules_mapping` 命中，所以**不再需要手改 config**。已把
`/storage/lkk/m3/MiniMax-M3-MXFP4/config.json` 从备份 `config.json.bak` 还原为原始 HF 命名，
模型目录保持标准（也便于 transformers 等其他工具读取）。

---

## 7. 验证（离线模拟已通过）

用真实 config.json + 修复后的 mapper/packed_mapping 复现匹配逻辑，vision 的 qkv/fc1/fc2 全部命中 ignore：

```
=== 修复后（mapper 无结尾点 + packed_modules_mapping 生效）===
  True <- vision_tower.vision_model.encoder.layers.0.self_attn.qkv_proj
  True <- vision_tower.vision_model.encoder.layers.0.fc1
  True <- vision_tower.vision_model.encoder.layers.0.fc2
  True <- vision_tower.vision_model.encoder.layers.31.self_attn.qkv_proj
mapper 是否把 vision ...mlp.fc1 改成 ...fc1: True
```

端到端验证：重跑 `bash eval_vllm.sh`，应能越过之前崩溃的模型加载阶段。
若还有新的 KeyError（例如 `gate_up_proj` 或语言层某个融合层命名差异），把新日志发我继续排。

---

## 8. 一图总结

```
config.json ignore（HF 名）           vLLM 内部模块名（prefix）
 ...mlp.fc1                            ...fc1              ← 普通层
 q_proj / k_proj / v_proj             ...qkv_proj         ← 融合层

要让二者对上，需要两座桥：
  桥①  hf_to_vllm_mapper：把 ignore 里的 ...mlp.fc1 → ...fc1
        └─ nvidia 版原来键带结尾点，桥断了 → fc1 KeyError（第一次）
  桥②  packed_modules_mapping：把 vLLM 的 qkv_proj 拆回 q/k/v_proj 再匹配 ignore
        └─ nvidia 版原来根本没这个属性，桥没建 → qkv_proj KeyError（第二次）

修复 = 把两座桥按 amd 版补齐。
```

# 报错5分析与修复：KV cache 块大小与稀疏注意力不匹配（No common block size for 16）

> 对应日志：`/storage/lkk/m3/test_5.log`
> 你的问题：“vllm_m3 的代码里有没有修复？” —— **没有**。
> 结论：这是 vLLM 对 MiniMax-M3「**全注意力 + 稀疏注意力混合**」结构的 KV cache
> 块大小协商缺口。`vllm_m3`（docker 拷贝）与运行时的 `/storage/lkk/xpu_vllm/vllm`
> 在稀疏后端这块**行为一致**（都要求 block=128），甚至运行时 vllm 更新一些
> （多了 SM100/MSA、fp8 indexer、ROCm 支持）。两边都没有针对这个问题的修复。

---

## 0. 先说结论：clamp 修复已生效，这是全新的第 5 个报错

报错4（`SWIGLUOAI_UNINTERLEAVE requires clamp_limit`）已被上一轮修复解决——本次日志
里 MoE 前向已经跑过去了，崩溃发生在**更后面的 KV cache 初始化 / CUDA graph 显存
profiling 阶段**：

```
determine_available_memory
  → profile_cudagraph_memory
    → _init_minimal_kv_cache_for_profiling
      → initialize_kv_cache
        → prepare_kernel_block_sizes
          → select_common_block_size
            raise ValueError("No common block size for 16. ")
```

---

## 1. 报错含义

`select_common_block_size(kv_manager_block_size=16, backends=[稀疏注意力后端])`：

- KV cache 管理器用的 **block_size = 16**（vLLM 默认）；
- 但 MiniMax-M3 稀疏注意力 / lightning-indexer 内核**只支持 block_size = 128**。

`vllm/models/minimax_m3/common/sparse_attention.py` 与 `indexer.py`：

```python
@staticmethod
def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
    # Page size == sparse block size (one sparse block per KV page).
    return [128]
```

协商逻辑（`vllm/v1/worker/utils.py:select_common_block_size`）：

```python
# Case 1: 16 被所有后端支持吗？稀疏后端只支持 128 → 否
# Case 2: 在所有后端的 int 支持尺寸里，找一个能整除 16 的
#         唯一候选是 128，但 16 % 128 != 0 → 跳过
raise ValueError(f"No common block size for {kv_manager_block_size}. ")  # 16
```

16 不是 128 的倍数，无解 → 报错。

---

## 2. 根因：MiniMax-M3 是「全注意力 + 稀疏注意力」混合结构

看你的模型 config（`text_config.sparse_attention_config`）：

```json
"sparse_block_size": 128,
"sparse_attention_freq": [0,0,0,1,1,1, ... ,1]   // 共 60 层
```

- **第 0/1/2 层**：`freq = 0` → **全注意力（dense full attention）**，用标准后端
  （FlashAttn / FlashInfer，`get_supported_kernel_block_sizes() = [MultipleOf(16)]`，
  block=16 就满足）。
- **第 3–59 层**：`freq = 1` → **稀疏注意力 + lightning indexer**，只支持 block=128。

> 顺带一提：这和报错3里「前 3 层是 dense MLP、3 层之后是 MoE」是同一批 dense 层
> ——MiniMax-M3 前 3 层整体就是 dense（全注意力 + dense MLP）。

### 块大小是怎么被定成 16 的

vLLM 自动选块大小的逻辑在 `Platform.update_block_size_for_backend`
（`vllm/platforms/interface.py`）：

```python
backend_cls = cls._find_non_ssm_backend(vllm_config)   # ← 关键
...
if not cache_config.user_specified_block_size:
    preferred = backend_cls.get_preferred_block_size(DEFAULT_BLOCK_SIZE)  # 16
    cache_config.block_size = preferred
```

而 `_find_non_ssm_backend` 只取**第一个**非 SSM 注意力后端：

```python
def _find_non_ssm_backend(cls, vllm_config):
    for layer in attn_layers.values():
        b = layer.get_attn_backend()
        if not b.is_ssm():
            return b        # ← 返回“第一个”，就停了
    return None
```

MiniMax-M3 第 0 层是**全注意力**，于是这里返回的是 FlashAttn 后端。
FlashAttn 的 `get_preferred_block_size(16)`：因为它支持 `MultipleOf(16)`，16 本身就合法，
直接返回 **16**。于是 `cache_config.block_size` 被定成 16。

**但第 3–59 层的稀疏后端需要 128**。到 KV cache 初始化时，稀疏那一组用 block=16 去
和 `[128]` 协商 → `No common block size for 16`。

一句话根因：**块大小自动选择只看了第一个（全注意力）后端，忽略了后面稀疏层要求的
128；对全+稀疏混合的模型，它选小了。**

---

## 3. 为什么 DeepSeek-V3.2 不报这个错，MiniMax-M3 报

DeepSeek-V3.2 的所有注意力层都是同一种（稀疏 MLA），`_find_non_ssm_backend` 返回的
第一个后端就是稀疏后端，`get_preferred_block_size` 直接给出它需要的大块尺寸，因此不冲突。

MiniMax-M3 特殊在**前几层是普通全注意力、后面才是稀疏**，第一个后端“骗过”了自动选择。
这也是为什么 vLLM 的 `MODELS_CONFIG_MAP` 里给 DeepSeek 注册了钩子、却**没有给
MiniMax-M3 注册任何 block_size 处理**——这个混合场景没被覆盖到。

---

## 4. 修复方案（已实施，改的是通用 config 钩子，不是模型前向代码）

vLLM 有一套 per-架构的 `verify_and_update_config` 钩子机制
（`vllm/model_executor/models/config.py` 的 `MODELS_CONFIG_MAP`，在引擎初始化早期按
架构名调用）。给 MiniMax-M3 补一个钩子，把 KV cache 块大小**钉到 sparse_block_size
（128）**，并标记为“用户已指定”，这样后续的自动选择（Phase 1）不会再把它改回 16。

### 改动 1：新增 `MiniMaxM3Config` 钩子

`vllm/model_executor/models/config.py`：

```python
class MiniMaxM3Config(VerifyAndUpdateConfig):
    @classmethod
    def verify_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        hf_config = vllm_config.model_config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        sparse_cfg = getattr(text_config, "sparse_attention_config", None)
        if sparse_cfg is None:
            return
        if isinstance(sparse_cfg, dict):
            sparse_block_size = sparse_cfg.get("sparse_block_size")
        else:
            sparse_block_size = getattr(sparse_cfg, "sparse_block_size", None)
        if not sparse_block_size:
            return

        cache_config = vllm_config.cache_config
        if cache_config.block_size != sparse_block_size:
            logger.info(
                "Setting KV cache block size to %d to match MiniMax-M3 sparse "
                "block size (was %s).",
                sparse_block_size, cache_config.block_size,
            )
        cache_config.block_size = sparse_block_size
        cache_config.user_specified_block_size = True   # 防止被自动选择重置回 16
```

### 改动 2：注册两个架构名

```python
MODELS_CONFIG_MAP = {
    ...
    "MiniMaxM3SparseForCausalLM": MiniMaxM3Config,
    "MiniMaxM3SparseForConditionalGeneration": MiniMaxM3Config,  # 你这份模型的架构
    ...
}
```

### 为什么 128 对所有层都成立

| 层类型 | 后端 | 支持的 block | 128 是否可用 |
|--------|------|-------------|-------------|
| 第 0–2 层 全注意力 | FlashAttn/FlashInfer | `MultipleOf(16)` | ✅ 128 % 16 == 0 |
| 第 3–59 层 稀疏 + indexer | MiniMax 稀疏后端 | `[128]` | ✅ 正好 128 |

于是 `select_common_block_size` 对每个 KV cache 组都能选到 128，不再报错。

### 为什么必须同时设 `user_specified_block_size = True`

块大小的“自动选择”`update_block_size_for_backend` 在 worker 里、比我们的钩子更晚运行。
它的 Phase 1 只有在 `not user_specified_block_size` 时才会覆盖 block_size。如果我们只改
block_size 不设这个标志，它会在 worker 里又把 128 改回 16（因为它只看第一个全注意力
后端）。设成 True 就跳过 Phase 1，保住 128。

---

## 5. 备选：不改代码，直接在启动参数里指定 block_size=128

如果你不想依赖代码改动，最简单的等价做法是在 `eval_vllm.sh` 的 `MODEL_ARGS` 里加一项
`block_size=128`：

```bash
MODEL_ARGS="pretrained=${MODEL},tensor_parallel_size=${NUM_GPUS},...,block_size=128"
```

vLLM 会把它当作用户显式指定（`user_specified_block_size=True`），效果和代码修复一致，
且无需改 vLLM 源码。两种方式二选一即可（代码修复的好处是开箱即用、不用记住这个参数）。

> 注意：`--block-size` 只影响 KV cache 分页粒度，不影响精度/结果；对稀疏模型这是正确
> 且必需的设置。

---

## 6. 到目前为止 5 个报错的全景

| # | 报错 | 阶段 | 性质 | 处理 |
|---|------|------|------|------|
| 1 | `KeyError: vision...fc1.weight` | 加载 | nvidia fork mapper 键带结尾点 + 视觉层误量化 | 改 `nvidia/model.py` mapper |
| 2 | `KeyError: vision...qkv_proj.weight` | 加载 | nvidia fork 缺 `packed_modules_mapping` | 补 `packed_modules_mapping` |
| 3 | `different quantization schemes [gate_proj, up_proj]` | 加载 | 量化 `--ignore_layers mlp.gate` 误伤 dense gate_proj | **重量化**（`block_sparse_moe.gate`） |
| 4 | `SWIGLUOAI_UNINTERLEAVE requires clamp_limit` | 前向 | cutlass MXFP4 MoE 漏传 clamp | 打通 clamp（config/cutlass_moe） |
| 5 | `No common block size for 16` | KV cache 初始化 | 全+稀疏混合注意力，块大小选小了 | 本文档：钉 block_size=128 |

- **报错 1、2**：nvidia fork 抄漏（模型定义）。
- **报错 3**：量化命令 bug（需重量化）。
- **报错 4、5**：vLLM 通用支持缺口——分别在「MXFP4 MoE 的钳位激活」和
  「混合注意力的 KV 块大小协商」，两份 vLLM 代码都有，`vllm_m3` 也没修。

---

## 7. 本次改动文件清单

- `vllm/model_executor/models/config.py`
  - 新增 `MiniMaxM3Config(VerifyAndUpdateConfig)`：把 KV cache `block_size` 设为
    `sparse_block_size`（128）并置 `user_specified_block_size=True`。
  - `MODELS_CONFIG_MAP` 注册 `MiniMaxM3SparseForCausalLM` 与
    `MiniMaxM3SparseForConditionalGeneration`。

> 未改模型前向 / 稀疏注意力内核，只在引擎初始化阶段修正块大小；对其它模型无影响
> （仅当架构名匹配 MiniMax-M3 且存在 `sparse_attention_config` 时才生效）。

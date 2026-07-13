# DiffusionGemma (int4-AutoRound) 在 XPU 上的报错排查与修复记录

- 模型：`Intel/diffusiongemma-26B-A4B-it-int4-AutoRound`
- 环境：Intel XPU，vLLM `v0.19.1rc1.dev2996+g8535a3527`，venv 路径 `/opt/venv`
- 复现脚本：`/sdp/lkk/eval_vllm_w4a16_ar_diffusiongemma.sh`（lm_eval + vLLM，任务 piqa）
- 对照脚本：
  - `/sdp/lkk/eval_vllm_w4a16_ar_gemma.sh`（`Intel/gemma-4-31B-it-int4-AutoRound`，正常）
  - `/sdp/lkk/eval_vllm_w4a16_ar_moe.sh`（`Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound`，正常）

按加载/运行顺序，共遇到并解决了 3 个问题。

---

## 问题 1：embedding 层量化方法选择错误（代码 bug）

### 报错
```
File ".../vllm/model_executor/models/gemma4.py", line 980, in __init__
    self.embed_tokens = VocabParallelEmbedding(...)
File ".../vllm/model_executor/layers/vocab_parallel_embedding.py", line 290, in __init__
    raise NotImplementedError
NotImplementedError: The class UnquantizedLinearMethod must implement the
'embedding' method, see UnquantizedEmbeddingMethod.
```
（日志：`/sdp/lkk/test.log`）

### 根因
`VocabParallelEmbedding.__init__` 要求 embedding 层拿到的 quant method 必须实现
`embedding()` 方法。而 INC 量化的
`vllm/model_executor/layers/quantization/inc/inc.py::get_quant_method`
第一个分支（处理 `extra_config` 里标记为“不量化” bits>=16 的层）对所有非
`RoutedExperts` 的层一律返回 `UnquantizedLinearMethod()`——它是给线性层用的，
不实现 `embedding()`，因此 embedding 层命中此分支即崩溃。

第二个分支（未量化通用路径）本来就按层类型正确分派，第一个分支漏了这套判断。

### 为什么 gemma-4-31B 不报错、diffusiongemma 报错
差异在两个模型 `config.json` 的 `quantization_config`：
- `gemma-4-31B`：没有 `extra_config` → 第一个分支的 `if prefix and self.extra_config`
  条件为假，整段跳过，embedding 走第二个分支返回 `None` → fallback 到
  `UnquantizedEmbeddingMethod`，正常。
- `diffusiongemma`：`extra_config` 里显式有
  `model.decoder.embed_tokens: {bits: 16, ...}` → embedding 命中第一个分支的 bug。

### 修复
`vllm/model_executor/layers/quantization/inc/inc.py::get_quant_method`
第一个分支补齐与第二个分支一致的类型判断：

```python
) and self.extra_config[layer_name].get("bits", 16) >= 16:
    if isinstance(layer, (LinearBase, ParallelLMHead)):
        return UnquantizedLinearMethod()
    if isinstance(layer, RoutedExperts):
        return UnquantizedFusedMoEMethod(layer.moe_config)
    return None
```
返回 `None` 后，`VocabParallelEmbedding` 会 fallback 到
`UnquantizedEmbeddingMethod`（实现了 `embedding()`），不再报错。

---

## 问题 2：MoE WNA16 的 `assert group_size >= 32`（模型形状与 TP 不兼容，非代码 bug）

### 报错
```
File ".../vllm/model_executor/layers/quantization/moe_wna16.py", line 222, in create_weights
    assert group_size >= 32
AssertionError
```
（日志：`/sdp/lkk/test_2.log`，此时 TP=4）

### 根因
`moe_wna16.py::create_weights` 要求 MoE intermediate 按 TP 切分后的每个分片能被
group_size 整除，否则把 group_size 不断减半，减到 <32 就断言失败：

```python
while intermediate_size_per_partition % group_size or hidden_size % group_size:
    group_size = group_size // 2
    group_size_div_factor *= 2
    assert group_size >= 32
```

各模型算术对比（auto-round int4）：

| | diffusiongemma（报错） | Qwen3.6（正常） |
|---|---|---|
| `moe_intermediate_size` | **704** | 512 |
| config `group_size` | 64 | 128 |
| TP=4 切分后每片 | 704/4 = **176** | 512/4 = 128 |
| 能被 group_size 整除？ | 176%64=48 ❌，减半到 32 仍 176%32=16 ❌，再减到 16 → `assert 16>=32` 崩 | 128%128=0 ✓ |

关键：`176 = 2^4 × 11`，最大 2 的幂因子只有 16，永远凑不出 ≥32 且整除 176 的
group_size；而 Qwen 的 512/4=128 = 2^7 仍能被 group_size 整除，所以通过。

### 解决办法（改 TP，非改代码）
让 `704/TP` 仍能被 ≥32 的数整除。`704 = 2^6 × 11`：
- **TP=1** → 704（÷64 ✓）
- **TP=2** → 352（÷32 ✓，group_size 自动降到 32）
- TP=4 → 176 ❌

结论：这个模型把 `tensor_parallel_size` 改为 **2**（或 1）。本次采用 TP=2。

---

## 问题 3：UVA 指针被传入 torch.compile 的 Triton kernel（XPU 移植性 bug）

### 报错
```
File ".../vllm/model_executor/models/diffusion_gemma.py", line 1310, in __call__
    scaled = _compiled_sample_step(...)
...
File ".../torch/_inductor/.../<hash>.py", line 1871, in call
    triton_poi_fused_index_0.run(arg0_1, arg15_1, buf13, s21, 1, ...)
RuntimeError: Pointer argument doesn't reference XPU device memory at 0-th argument, err=0
```
（日志：`/sdp/lkk/test_3.log`，TP=2、enforce_eager=True，仍在 warmup 阶段崩）

### 根因
DiffusionGemma 的自定义采样器把 **UVA-backed（host-pinned + 统一虚拟寻址映射）**
的张量直接喂进了 `@torch.compile` 生成的 Triton kernel。

- `self._decode_slots` / `self._decode_idx` / `self._num_logits` 是
  `UvaBackedTensor`，其 `.gpu` 实际是
  `get_xpu_view_from_cpu_tensor()` 返回的 UVA 视图，指向 pinned CPU 内存，
  **不是真正的 XPU 显存**。
- 经确认，报错 kernel 的 `arg0_1` 就是 `decode_slots`（对应
  `history_len_tensor[decode_slots]` 的 index gather）。
- 在 **CUDA** 上，UVA 指针 kernel 能透明解引用，所以没问题；在 **XPU** 上，
  Triton static launcher 会校验指针，发现是 host/UVA 而非 device 内存直接拒绝。

补充说明：
- `enforce_eager=True` 不能规避，因为出问题的
  `_compiled_sample_step` / `_compute_num_rejected` 上是模型代码里
  **硬写的 `@torch.compile` 装饰器**，不受 vLLM 图编译开关影响。
- `all_slots`、`valid_canvas_len` 走 `async_copy_to_gpu`（真实 device 显存），不受影响。

### 方案 A（快速验证，不改代码）
全局关闭 torch.compile，让 sampler 走 eager（ATen 的 index 在 XPU 上能正常读 UVA 张量，只是慢）：
```bash
export TORCHDYNAMO_DISABLE=1   # 或 TORCH_COMPILE_DISABLE=1
```

### 方案 B（推荐，保留编译）
在传入编译函数前，把 UVA-backed 张量 `.clone()` 成真实 XPU 显存。
文件 `vllm/model_executor/models/diffusion_gemma.py`，共 3 处：

```python
# 约 1205 行，喂给 _compute_num_rejected
num_rejected = _compute_num_rejected(
    self._num_logits.gpu[:num_reqs].clone(),
    num_sampled,
    input_batch.query_start_loc[: num_reqs + 1],
)

# 约 1256-1257 行，喂给 _compiled_sample_step
decode_slots = self._decode_slots.gpu[:num_decode].clone()
decode_idx = self._decode_idx.gpu[:num_decode].clone()
```
（`_query_lens.gpu` 未被读取，无需处理。）

clone 的都是 `[num_reqs]`/`[num_decode]` 小 int 张量，开销可忽略，跨平台安全。

---

## 修改文件清单

| 文件 | 改动 | 对应问题 |
|---|---|---|
| `vllm/model_executor/layers/quantization/inc/inc.py` | `get_quant_method` 第一个分支按层类型分派 | 问题 1 |
| `eval_vllm_w4a16_ar_diffusiongemma.sh` | `tensor_parallel_size` 4 → 2 | 问题 2 |
| `vllm/model_executor/models/diffusion_gemma.py` | 3 处 UVA 张量 `.clone()` | 问题 3 |

> 说明：本次为纯 Python 改动，直接同步到 `/opt/venv/.../site-packages/vllm/...`
> 已安装文件即可生效（Python 会因源码 mtime 更新重新编译 .pyc），无需重新
> `pip install`。源码树 `/workspace/vllm/...` 也已同步同样改动。
> 上述 3 处代码改动均为 vLLM 层面的 XPU / 量化兼容性修复，建议整理后向上游提交
> PR（提交前请按 vLLM 贡献规范做重复性检查并附测试结果）。

## 验证方式
```bash
cd /sdp/lkk
bash eval_vllm_w4a16_ar_diffusiongemma.sh   # TP=2
```

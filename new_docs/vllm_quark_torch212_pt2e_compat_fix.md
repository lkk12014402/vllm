# 报错8分析与修复：AMD-Quark 版 MXFP4 模型报 “amd-quark is required”（实为 torch 2.12 API 迁移）

> 对应日志：`/storage/lkk/m3/test_8.log`（测的是 **AMD quark 版** MiniMax-M3-MXFP4 模型）
> 结论：**不是 amd-quark 没装**（你已装 0.11.2）。真正原因是
> **quark 0.11.2 与 torch 2.12 的 PT2E API 迁移不兼容**，import 期就炸，被 vLLM 的
> `except ImportError` 吞掉、误报成“需要安装 amd-quark”。
> 已在运行时库 `/storage/lkk/xpu_vllm/vllm` 加一个**兼容 shim** 修好，无需改 quark、不污染全局环境。

---

## 1. 报错现象

日志里模型已成功加载（走的是 quark 量化方法 + OCP MX 仿真 MoE）：

```
[mxfp4.py:1705] Using MoEPrepareAndFinalizeNoDPEPModular
[ocp_mx_emulation_moe.py:48] Using OCP_MXQuantizationEmulationTritonExperts MOE backend.
  This will dequantize weights on the fly ...
[gpu_model_runner.py:5272] Model loading took 113.35 GiB ...
```

崩溃发生在 KV cache profiling（第一次真正跑 dequant kernel）时，worker 抛：

```
RuntimeError: Worker failed with error 'The package `amd-quark` is required to use
MX-FP4 models. Please install it with `pip install amd-quark`.'
```

而实际上 `pip show amd-quark` = **0.11.2 已安装**，`import quark` 顶层也 OK。

---

## 2. 真正的根因：torch 2.12 把 PT2E 从 `torch.ao` 迁到了 `torchao`

vLLM 需要 quark 的 on-the-fly dequant kernel（`quark.torch.kernel.mx.dq_mxfp4/qdq_mxfp4`）。
但 `import quark.torch.kernel.mx` 会先触发 `quark.torch.__init__` **eager import 整条图量化栈**
（`quark.torch.quantization.api → graph.optimization → ...`），其中多处 import 了 PyTorch 的
**私有 PT2E API**：

```python
# quark/torch/quantization/graph/optimization/utils.py
from torch.ao.quantization.pt2e.utils import _get_tensor_constant_from_node
# quark/torch/quantization/graph/processor/insert_quantizer.py
from torch.ao.quantization.pt2e.prepare import _get_edge_or_node_to_group_id, _get_edge_or_node_to_qspec
from torch.ao.quantization.quantizer import EdgeOrNode
```

**torch 2.12 已经把整个 `torch.ao.quantization.pt2e`（以及 pt2e 的 `quantizer`）删除/迁移到独立的
`torchao` 包**：

```
torch.ao.quantization.pt2e            → ModuleNotFoundError（torch 2.12 里没了）
torchao.quantization.pt2e.utils       → OK（_get_tensor_constant_from_node 在这）
torchao.quantization.pt2e.prepare     → OK（_get_edge_or_node_to_group_id / _qspec 在这）
torchao.quantization.pt2e.quantizer   → OK（EdgeOrNode 在这）
```

于是 `from quark.torch.kernel import mx` 因链式 import 抛 `ModuleNotFoundError:
No module named 'torch.ao.quantization.pt2e'`。vLLM 的代码：

```python
# vllm/model_executor/layers/quantization/utils/mxfp4_utils.py
try:
    from quark.torch.kernel import mx
except ImportError as err:
    raise ImportError("The package `amd-quark` is required to use MX-FP4 models. ...") from err
```

`ModuleNotFoundError` 是 `ImportError` 的子类 → 被这个 `except` 捕获 → **误报成“未安装 amd-quark”**。

### 为什么自量化的 llm_compressor 版模型不报这个
你自量化的模型是 **compressed-tensors** 格式，走的是 cutlass/marlin MXFP4 路径
（前 7 个报错那条链），**根本不 import quark**。只有 **AMD quark 格式**的模型才会走
`quark` 量化方法、调用 quark 的 dequant kernel，才会撞上这个 import。

### 为什么在 AMD 机器上不报
quark 是 AMD 自家的库，其发布版与 AMD ROCm 栈里的 torch 版本配套；那套 torch 仍带
`torch.ao.quantization.pt2e`。而你这套是给 vLLM 专门编译的 **torch 2.12（很新）**，PT2E 已迁走。

---

## 3. 为什么不能靠“升级/降级 quark”解决

- `pip index versions amd-quark` → 最高只有 **0.11.2**（已装），没有更新版本可升。
- 降级到 0.11/0.10 等旧版同样用 `torch.ao.quantization.pt2e` 老路径，一样炸。
- 直接 `pip install -U torch` 之类会破坏 vLLM 专编的 torch 2.12+cu130 环境（严禁，见 flashinfer 那篇）。

**关键洞察**：vLLM 推理只用 quark 的 **dequant kernel**（`quark.torch.kernel.mx`），
**根本不执行**那条 PT2E 图优化/量化路径——它只是在 **import 期**被连累。
所以只要让 import 通过即可，被 alias 的代码永不被调用。

---

## 4. 修复：PT2E 模块别名兼容 shim（已落地）

新增文件：
`vllm/model_executor/layers/quantization/utils/quark_torch_compat.py`

核心逻辑（幂等、静默、只在缺失时生效）：

```python
_QUARK_TORCH_AO_ALIASES = {
    "torch.ao.quantization.pt2e":          "torchao.quantization.pt2e",
    "torch.ao.quantization.pt2e.utils":    "torchao.quantization.pt2e.utils",
    "torch.ao.quantization.pt2e.prepare":  "torchao.quantization.pt2e.prepare",
    "torch.ao.quantization.quantizer":     "torchao.quantization.pt2e.quantizer",
}

def ensure_quark_torch_ao_compat() -> None:
    # 若 torch 本身就有 pt2e（旧 torch），直接返回，不做任何事
    try:
        importlib.import_module("torch.ao.quantization.pt2e.utils"); return
    except Exception:
        pass
    # 否则把旧路径别名到 torchao 的新路径
    for old, new in _QUARK_TORCH_AO_ALIASES.items():
        if old in sys.modules: continue
        try:
            sys.modules[old] = importlib.import_module(new)
        except Exception:
            return   # torchao 也没有就放弃，让原始 ImportError 自然报出

ensure_quark_torch_ao_compat()   # 模块 import 时执行一次
```

接入点（在每个 quark import 之前调用一次，幂等）：
- `mxfp4_utils.py`：顶部 import + `_dequant_mxfp4` / `_quant_dequant_mxfp4` 两处
- `mxfp6_utils.py`：顶部 import + `_quant_dequant_mxfp6` / `_dequant_mxfp6` 两处

改动量：新增 1 文件（compat shim）+ 2 文件各 +5 行。**不改 quark、不动全局环境、不动 torch。**

### 为什么用 vLLM 内 shim 而不是全局 sitecustomize
本环境非独占（可能与他人共享）。全局 `sitecustomize.py` 会影响该 python 里所有程序；
放进 vLLM 的 mxfp4/mxfp6 util（只有加载 MX-FP4/6 quark 模型时才 import）作用域最小、
与前 7 个报错的修复风格一致（都在 `/storage/lkk/xpu_vllm/vllm` 内）。

---

## 5. 验证（已通过）

```
pre: torch.ao pt2e MISSING -> ModuleNotFoundError          # 修前：老路径确实没了
imported vllm mxfp4_utils OK                                # import vLLM util 即装好 shim
quark.torch.kernel.mx OK; dq_mxfp4: True qdq_mxfp4: True    # quark kernel 现在能 import
_dequant_mxfp4 raised: RuntimeError number of output elements should be a multiple of 64
                                                            # ↑ 已进入真实 kernel（假输入shape报错），
                                                            #   不再是“amd-quark is required”
```

即：原“amd-quark is required”不再出现，dequant 走的是真实 quark kernel。

> `Failed to import from vllm._qutlass_C: undefined symbol ...` 是**预存在的可选扩展 ABI 警告**，
> vLLM 自动回退，与本修复及推理无关。

---

## 6. 迁移状态与后续

- 已修：**运行时库 `/storage/lkk/xpu_vllm/vllm`**（`eval_vllm.sh` 实际用的就是它）→ 直接重跑即可。
- 如需在另两个库保持一致，同样迁移 3 个改动：
  - 新增 `.../quantization/utils/quark_torch_compat.py`
  - `mxfp4_utils.py`、`mxfp6_utils.py` 各加 import + 调用（注意目标库的行号可能不同，按 quark import 点插）
  - 目标库：`/storage/lkk/m3/commit/vllm`、`/storage/lkk/m3/vllm_m3`
- 后续若 quark 出兼容 torch 2.12 的新版本，shim 会自动 no-op（检测到 `torch.ao pt2e` 存在就跳过），
  向前兼容、无需回滚。

---

## 7. 相关文件

- `vllm/model_executor/layers/quantization/utils/quark_torch_compat.py`（新增，shim）
- `vllm/model_executor/layers/quantization/utils/mxfp4_utils.py`（`_dequant_mxfp4:112`、`_quant_dequant_mxfp4:135` 前调 shim）
- `vllm/model_executor/layers/quantization/utils/mxfp6_utils.py`（同理两处）
- quark 侧失效 import 源头（仅供参考，未改）：
  `quark/torch/quantization/graph/optimization/utils.py`、
  `quark/torch/quantization/graph/processor/insert_quantizer.py`

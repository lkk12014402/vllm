
为什么精度出来是0




这是转换脚本
python3 /storage/lkk/m3/convert_native_to_original_schema.py   --src ./MiniMax-M3-MXFP4-MXFP8-auto/   --orig-config /storage/lkk/MiniMax-M3   --dst ./M3-rtn-auto-vllm



这是评估脚本

root@ip-172-31-32-47:/storage/lkk/m3# bash eval_vllm_gsm8k.sh
/usr/local/lib/python3.12/dist-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3 (2.6.3) or chardet (6.0.0.post1)/charset_normalizer (3.4.4) doesn't match a supported version!
  warnings.warn(
2026-07-08:01:29:27 INFO     [_cli.run:388] Selected Tasks: ['gsm8k']
2026-07-08:01:29:28 INFO     [evaluator:214] Setting random seed to 0 | Setting numpy seed to 1234 | Setting torch manual seed to 1234 | Setting fewshot manual seed to 1234
2026-07-08:01:29:28 INFO     [evaluator:239] Initializing vllm model, with arguments: {'pretrained': './M3-rtn-auto-vllm', 'tensor_parallel_size': 2, 'max_model_len': 8192, 'max_num_batched_tokens': 32768, 'max_num_seqs': 128, 'add_bos_token': True, 'gpu_memory_utilization': 0.8, 'dtype': 'bfloat16', 'max_gen_toks': 2048, 'enable_prefix_caching': False, 'reasoning_parser': 'minimax_m3'}
WARNING 07-08 01:29:28 [cuda.py:45] Failed to import from vllm._qutlass_C: ImportError('/storage/lkk/xpu_vllm/vllm/vllm/_qutlass_C.abi3.so: undefined symbol: _ZNR5torch7Library4_defEON3c1014FunctionSchemaEPNS1_12OperatorNameERKSt6vectorIN2at3TagESaIS8_EENS_17_RegisterOrVerifyE')
INFO 07-08 01:29:43 [api_utils.py:273] non-default args: {'dtype': 'bfloat16', 'seed': 1234, 'max_model_len': 8192, 'tensor_parallel_size': 2, 'enable_prefix_caching': False, 'gpu_memory_utilization': 0.8, 'max_num_batched_tokens': 32768, 'max_num_seqs': 128, 'disable_log_stats': True, 'reasoning_parser': 'minimax_m3', 'model': './M3-rtn-auto-vllm'}
INFO 07-08 01:29:43 [model.py:601] Resolved architecture: MiniMaxM3SparseForConditionalGeneration
INFO 07-08 01:29:43 [model.py:1731] Using max model len 8192
WARNING 07-08 01:29:43 [cuda.py:233] Failed to import from vllm._qutlass_C: ImportError('/storage/lkk/xpu_vllm/vllm/vllm/_qutlass_C.abi3.so: undefined symbol: _ZNR5torch7Library4_defEON3c1014FunctionSchemaEPNS1_12OperatorNameERKSt6vectorIN2at3TagESaIS8_EENS_17_RegisterOrVerifyE')
WARNING 07-08 01:29:43 [cuda.py:233] Failed to import from vllm._qutlass_C: ImportError('/storage/lkk/xpu_vllm/vllm/vllm/_qutlass_C.abi3.so: undefined symbol: _ZNR5torch7Library4_defEON3c1014FunctionSchemaEPNS1_12OperatorNameERKSt6vectorIN2at3TagESaIS8_EENS_17_RegisterOrVerifyE')
WARNING 07-08 01:29:43 [cuda.py:233] Failed to import from vllm._qutlass_C: ImportError('/storage/lkk/xpu_vllm/vllm/vllm/_qutlass_C.abi3.so: undefined symbol: _ZNR5torch7Library4_defEON3c1014FunctionSchemaEPNS1_12OperatorNameERKSt6vectorIN2at3TagESaIS8_EENS_17_RegisterOrVerifyE')
WARNING 07-08 01:29:43 [cuda.py:233] Failed to import from vllm._qutlass_C: ImportError('/storage/lkk/xpu_vllm/vllm/vllm/_qutlass_C.abi3.so: undefined symbol: _ZNR5torch7Library4_defEON3c1014FunctionSchemaEPNS1_12OperatorNameERKSt6vectorIN2at3TagESaIS8_EENS_17_RegisterOrVerifyE')
INFO 07-08 01:29:44 [scheduler.py:252] Chunked prefill is enabled with max_num_batched_tokens=32768.
INFO 07-08 01:29:44 [config.py:77] Setting KV cache block size to 128 to match MiniMax-M3 sparse block size (was 16).
INFO 07-08 01:29:44 [vllm.py:1006] Asynchronous scheduling is enabled.
INFO 07-08 01:29:44 [vllm.py:1094] Auto-enabling VLLM_USE_BREAKABLE_CUDAGRAPH=1. Set VLLM_USE_BREAKABLE_CUDAGRAPH=0 to opt out.
WARNING 07-08 01:29:44 [vllm.py:1100] VLLM_USE_BREAKABLE_CUDAGRAPH is set, disabling vLLM's torch.compile pipeline. Equivalent to -cc.mode=none.
WARNING 07-08 01:29:44 [vllm.py:1110] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.
INFO 07-08 01:29:44 [kernel.py:278] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['vllm_c', 'native'], fused_add_rms_norm=['vllm_c', 'native'])
INFO 07-08 01:29:47 [compilation.py:310] Enabled custom fusions: norm_quant, act_quant, allreduce_rms
/usr/local/lib/python3.12/dist-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3 (2.6.3) or chardet (6.0.0.post1)/charset_normalizer (3.4.4) doesn't match a supported version!
  warnings.warn(
WARNING 07-08 01:29:55 [cuda.py:45] Failed to import from vllm._qutlass_C: ImportError('/storage/lkk/xpu_vllm/vllm/vllm/_qutlass_C.abi3.so: undefined symbol: _ZNR5torch7Library4_defEON3c1014FunctionSchemaEPNS1_12OperatorNameERKSt6vectorIN2at3TagESaIS8_EENS_17_RegisterOrVerifyE')
WARNING 07-08 01:30:01 [cuda.py:233] Failed to import from vllm._qutlass_C: ImportError('/storage/lkk/xpu_vllm/vllm/vllm/_qutlass_C.abi3.so: undefined symbol: _ZNR5torch7Library4_defEON3c1014FunctionSchemaEPNS1_12OperatorNameERKSt6vectorIN2at3TagESaIS8_EENS_17_RegisterOrVerifyE')
WARNING 07-08 01:30:01 [cuda.py:233] Failed to import from vllm._qutlass_C: ImportError('/storage/lkk/xpu_vllm/vllm/vllm/_qutlass_C.abi3.so: undefined symbol: _ZNR5torch7Library4_defEON3c1014FunctionSchemaEPNS1_12OperatorNameERKSt6vectorIN2at3TagESaIS8_EENS_17_RegisterOrVerifyE')
WARNING 07-08 01:30:01 [cuda.py:233] Failed to import from vllm._qutlass_C: ImportError('/storage/lkk/xpu_vllm/vllm/vllm/_qutlass_C.abi3.so: undefined symbol: _ZNR5torch7Library4_defEON3c1014FunctionSchemaEPNS1_12OperatorNameERKSt6vectorIN2at3TagESaIS8_EENS_17_RegisterOrVerifyE')
WARNING 07-08 01:30:01 [cuda.py:233] Failed to import from vllm._qutlass_C: ImportError('/storage/lkk/xpu_vllm/vllm/vllm/_qutlass_C.abi3.so: undefined symbol: _ZNR5torch7Library4_defEON3c1014FunctionSchemaEPNS1_12OperatorNameERKSt6vectorIN2at3TagESaIS8_EENS_17_RegisterOrVerifyE')
(EngineCore pid=23537) INFO 07-08 01:30:01 [core.py:114] Initializing a V1 LLM engine (v0.23.1rc1.dev550+g58d6a6e60.d20260629) with config: model='./M3-rtn-auto-vllm', speculative_config=None, tokenizer='./M3-rtn-auto-vllm', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=8192, download_dir=None, load_format=auto, tensor_parallel_size=2, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=False, quantization=compressed-tensors, quantization_config=None, enforce_eager=False, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='minimax_m3', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False, jit_monitor_mode='warn', jit_monitor_verbose=False), seed=1234, served_model_name=./M3-rtn-auto-vllm, enable_prefix_caching=False, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.NONE: 0>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['all'], 'ir_enable_torch_wrap': False, 'splitting_ops': [], 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_vision_items_per_batch': 0, 'encoder_cudagraph_max_frames_per_batch': None, 'compile_sizes': [], 'compile_ranges_endpoints': [5461, 32768], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>, 'cudagraph_num_of_warmups': 1, 'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': True, 'fuse_act_quant': True, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': True, 'fuse_rope_kvcache_cat_mla': False, 'fuse_act_padding': False}, 'max_cudagraph_capture_size': 256, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': False, 'static_all_moe_layers': []}, kernel_config=KernelConfig(ir_op_priority=IrOpPriorityConfig(rms_norm=['vllm_c', 'native'], fused_add_rms_norm=['vllm_c', 'native']), enable_flashinfer_autotune=True, moe_backend='auto', linear_backend='auto')
(EngineCore pid=23537) WARNING 07-08 01:30:01 [multiproc_executor.py:1063] Reducing Torch parallelism from 96 threads to 1 to avoid unnecessary CPU contention. Set OMP_NUM_THREADS in the external environment to tune this value as needed.
(EngineCore pid=23537) INFO 07-08 01:30:01 [multiproc_executor.py:140] DP group leader: node_rank=0, node_rank_within_dp=0, master_addr=127.0.0.1, mq_connect_ip=172.31.32.47 (local), world_size=2, local_world_size=2
/usr/local/lib/python3.12/dist-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3 (2.6.3) or chardet (6.0.0.post1)/charset_normalizer (3.4.4) doesn't match a supported version!
  warnings.warn(
WARNING 07-08 01:30:02 [cuda.py:45] Failed to import from vllm._qutlass_C: ImportError('/storage/lkk/xpu_vllm/vllm/vllm/_qutlass_C.abi3.so: undefined symbol: _ZNR5torch7Library4_defEON3c1014FunctionSchemaEPNS1_12OperatorNameERKSt6vectorIN2at3TagESaIS8_EENS_17_RegisterOrVerifyE')
WARNING 07-08 01:30:08 [cuda.py:233] Failed to import from vllm._qutlass_C: ImportError('/storage/lkk/xpu_vllm/vllm/vllm/_qutlass_C.abi3.so: undefined symbol: _ZNR5torch7Library4_defEON3c1014FunctionSchemaEPNS1_12OperatorNameERKSt6vectorIN2at3TagESaIS8_EENS_17_RegisterOrVerifyE')
WARNING 07-08 01:30:08 [cuda.py:233] Failed to import from vllm._qutlass_C: ImportError('/storage/lkk/xpu_vllm/vllm/vllm/_qutlass_C.abi3.so: undefined symbol: _ZNR5torch7Library4_defEON3c1014FunctionSchemaEPNS1_12OperatorNameERKSt6vectorIN2at3TagESaIS8_EENS_17_RegisterOrVerifyE')
WARNING 07-08 01:30:08 [cuda.py:233] Failed to import from vllm._qutlass_C: ImportError('/storage/lkk/xpu_vllm/vllm/vllm/_qutlass_C.abi3.so: undefined symbol: _ZNR5torch7Library4_defEON3c1014FunctionSchemaEPNS1_12OperatorNameERKSt6vectorIN2at3TagESaIS8_EENS_17_RegisterOrVerifyE')
WARNING 07-08 01:30:08 [cuda.py:233] Failed to import from vllm._qutlass_C: ImportError('/storage/lkk/xpu_vllm/vllm/vllm/_qutlass_C.abi3.so: undefined symbol: _ZNR5torch7Library4_defEON3c1014FunctionSchemaEPNS1_12OperatorNameERKSt6vectorIN2at3TagESaIS8_EENS_17_RegisterOrVerifyE')
/usr/local/lib/python3.12/dist-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3 (2.6.3) or chardet (6.0.0.post1)/charset_normalizer (3.4.4) doesn't match a supported version!
  warnings.warn(
WARNING 07-08 01:30:10 [cuda.py:45] Failed to import from vllm._qutlass_C: ImportError('/storage/lkk/xpu_vllm/vllm/vllm/_qutlass_C.abi3.so: undefined symbol: _ZNR5torch7Library4_defEON3c1014FunctionSchemaEPNS1_12OperatorNameERKSt6vectorIN2at3TagESaIS8_EENS_17_RegisterOrVerifyE')
(Worker pid=23802) INFO 07-08 01:30:10 [parallel_state.py:1588] world_size=2 rank=0 local_rank=0 distributed_init_method=tcp://127.0.0.1:54845 backend=nccl
WARNING 07-08 01:30:16 [cuda.py:233] Failed to import from vllm._qutlass_C: ImportError('/storage/lkk/xpu_vllm/vllm/vllm/_qutlass_C.abi3.so: undefined symbol: _ZNR5torch7Library4_defEON3c1014FunctionSchemaEPNS1_12OperatorNameERKSt6vectorIN2at3TagESaIS8_EENS_17_RegisterOrVerifyE')
WARNING 07-08 01:30:16 [cuda.py:233] Failed to import from vllm._qutlass_C: ImportError('/storage/lkk/xpu_vllm/vllm/vllm/_qutlass_C.abi3.so: undefined symbol: _ZNR5torch7Library4_defEON3c1014FunctionSchemaEPNS1_12OperatorNameERKSt6vectorIN2at3TagESaIS8_EENS_17_RegisterOrVerifyE')
WARNING 07-08 01:30:16 [cuda.py:233] Failed to import from vllm._qutlass_C: ImportError('/storage/lkk/xpu_vllm/vllm/vllm/_qutlass_C.abi3.so: undefined symbol: _ZNR5torch7Library4_defEON3c1014FunctionSchemaEPNS1_12OperatorNameERKSt6vectorIN2at3TagESaIS8_EENS_17_RegisterOrVerifyE')
WARNING 07-08 01:30:16 [cuda.py:233] Failed to import from vllm._qutlass_C: ImportError('/storage/lkk/xpu_vllm/vllm/vllm/_qutlass_C.abi3.so: undefined symbol: _ZNR5torch7Library4_defEON3c1014FunctionSchemaEPNS1_12OperatorNameERKSt6vectorIN2at3TagESaIS8_EENS_17_RegisterOrVerifyE')
(Worker pid=23810) INFO 07-08 01:30:19 [parallel_state.py:1588] world_size=2 rank=1 local_rank=1 distributed_init_method=tcp://127.0.0.1:54845 backend=nccl
(Worker pid=23802) INFO 07-08 01:30:19 [pynccl.py:113] vLLM is using nccl==2.29.7
(Worker pid=23802) INFO 07-08 01:30:21 [cuda_communicator.py:245] Using ['CUSTOM', 'SYMM_MEM', 'PYNCCL'] all-reduce backends (in dispatch order) for group 'tp:0' out of potential backends: ['NCCL_SYMM_MEM', 'QUICK_REDUCE', 'FLASHINFER', 'CUSTOM', 'SYMM_MEM', 'PYNCCL'].
(Worker pid=23802) INFO 07-08 01:30:22 [cuda_communicator.py:245] Using ['PYNCCL'] all-reduce backends (in dispatch order) for group 'ep:0' out of potential backends: ['NCCL_SYMM_MEM', 'QUICK_REDUCE', 'FLASHINFER', 'CUSTOM', 'SYMM_MEM', 'PYNCCL'].
(Worker pid=23802) INFO 07-08 01:30:22 [parallel_state.py:1923] rank 0 in world size 2 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank 0, EPLB rank N/A
(Worker pid=23802) INFO 07-08 01:30:22 [topk_topp_sampler.py:55] Using FlashInfer for top-p & top-k sampling.
(Worker_TP0 pid=23802) INFO 07-08 01:30:26 [gpu_model_runner.py:5175] Starting to load model ./M3-rtn-auto-vllm...
(Worker_TP0 pid=23802) INFO 07-08 01:30:27 [cuda.py:542] Using backend AttentionBackendEnum.FLASH_ATTN for vit attention
(Worker_TP0 pid=23802) INFO 07-08 01:30:27 [mm_encoder_attention.py:373] Using AttentionBackendEnum.FLASH_ATTN for MMEncoderAttention.
(Worker_TP1 pid=23810) WARNING 07-08 01:30:27 [vllm.py:1110] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.
(Worker_TP1 pid=23810) INFO 07-08 01:30:27 [kernel.py:278] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['vllm_c', 'native'], fused_add_rms_norm=['vllm_c', 'native'])
(Worker_TP1 pid=23810) INFO 07-08 01:30:27 [selector.py:138] Using HND KV cache layout for FLASHINFER backend.
(Worker_TP0 pid=23802) INFO 07-08 01:30:27 [vllm.py:1006] Asynchronous scheduling is enabled.
(Worker_TP0 pid=23802) WARNING 07-08 01:30:27 [vllm.py:1100] VLLM_USE_BREAKABLE_CUDAGRAPH is set, disabling vLLM's torch.compile pipeline. Equivalent to -cc.mode=none.
(Worker_TP0 pid=23802) WARNING 07-08 01:30:27 [vllm.py:1110] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.
(Worker_TP0 pid=23802) INFO 07-08 01:30:27 [kernel.py:278] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['vllm_c', 'native'], fused_add_rms_norm=['vllm_c', 'native'])
(Worker_TP0 pid=23802) INFO 07-08 01:30:27 [compilation.py:310] Enabled custom fusions: norm_quant, act_quant, allreduce_rms
(Worker_TP0 pid=23802) INFO 07-08 01:30:27 [cuda.py:483] Using FLASHINFER attention backend out of potential backends: ['FLASHINFER', 'FLASH_ATTN', 'TRITON_ATTN', 'FLEX_ATTENTION'].
(Worker_TP0 pid=23802) INFO 07-08 01:30:27 [selector.py:138] Using HND KV cache layout for FLASHINFER backend.
(Worker_TP0 pid=23802) INFO 07-08 01:30:27 [__init__.py:798] Using FlashInferMxFp4LinearKernel for MXFP4 GEMM
(Worker_TP0 pid=23802) INFO 07-08 01:30:27 [sparse_attention.py:419] MiniMax M3 sparse attention selected MSA (kv_cache_dtype=auto, topk_blocks=16)
(Worker_TP0 pid=23802) INFO 07-08 01:30:27 [indexer.py:497] MiniMax M3 indexer: selected MSA (fmha_sm100 score + Triton top-k) [topk_blocks=16, indexer_kv_dtype=bf16]
(Worker_TP0 pid=23802) INFO 07-08 01:30:27 [compressed_tensors_moe_w4a4_mxfp4.py:53] Using CutlassExpertsMxfp4 for MXFP4 MoE
(Worker_TP0 pid=23802) INFO 07-08 01:30:28 [weight_utils.py:849] Filesystem type for checkpoints: EXT4. Checkpoint size: 263.01 GiB. Available RAM: 1879.95 GiB.
(Worker_TP0 pid=23802) INFO 07-08 01:30:28 [weight_utils.py:872] Auto-prefetch is disabled because the filesystem (EXT4) is not a recognized network FS (NFS/Lustre). If you want to force prefetching, start vLLM with --safetensors-load-strategy=prefetch.
Loading safetensors checkpoint shards:   0% Completed | 0/57 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:   2% Completed | 1/57 [00:00<00:21,  2.61it/s]
Loading safetensors checkpoint shards:   4% Completed | 2/57 [00:00<00:17,  3.22it/s]
Loading safetensors checkpoint shards:   5% Completed | 3/57 [00:00<00:12,  4.25it/s]
Loading safetensors checkpoint shards:   7% Completed | 4/57 [00:00<00:09,  5.34it/s]
Loading safetensors checkpoint shards:   9% Completed | 5/57 [00:01<00:08,  6.13it/s]
Loading safetensors checkpoint shards:  11% Completed | 6/57 [00:01<00:07,  6.81it/s]
Loading safetensors checkpoint shards:  12% Completed | 7/57 [00:01<00:07,  6.87it/s]
Loading safetensors checkpoint shards:  16% Completed | 9/57 [00:01<00:05,  9.41it/s]
Loading safetensors checkpoint shards:  19% Completed | 11/57 [00:01<00:04, 10.24it/s]
Loading safetensors checkpoint shards:  23% Completed | 13/57 [00:01<00:03, 11.89it/s]
Loading safetensors checkpoint shards:  26% Completed | 15/57 [00:01<00:03, 13.37it/s]
Loading safetensors checkpoint shards:  30% Completed | 17/57 [00:01<00:02, 14.42it/s]
Loading safetensors checkpoint shards:  33% Completed | 19/57 [00:02<00:02, 13.49it/s]
Loading safetensors checkpoint shards:  37% Completed | 21/57 [00:02<00:02, 13.88it/s]
Loading safetensors checkpoint shards:  40% Completed | 23/57 [00:02<00:02, 12.16it/s]
Loading safetensors checkpoint shards:  44% Completed | 25/57 [00:02<00:03,  9.57it/s]
Loading safetensors checkpoint shards:  47% Completed | 27/57 [00:02<00:03,  9.18it/s]
Loading safetensors checkpoint shards:  51% Completed | 29/57 [00:03<00:03,  9.07it/s]
Loading safetensors checkpoint shards:  53% Completed | 30/57 [00:03<00:03,  8.61it/s]
Loading safetensors checkpoint shards:  54% Completed | 31/57 [00:03<00:03,  8.58it/s]
Loading safetensors checkpoint shards:  56% Completed | 32/57 [00:03<00:02,  8.69it/s]
Loading safetensors checkpoint shards:  58% Completed | 33/57 [00:03<00:02,  8.67it/s]
Loading safetensors checkpoint shards:  60% Completed | 34/57 [00:03<00:02,  8.78it/s]
Loading safetensors checkpoint shards:  63% Completed | 36/57 [00:04<00:02,  9.28it/s]
Loading safetensors checkpoint shards:  65% Completed | 37/57 [00:04<00:02,  8.49it/s]
Loading safetensors checkpoint shards:  68% Completed | 39/57 [00:04<00:01, 10.13it/s]
Loading safetensors checkpoint shards:  72% Completed | 41/57 [00:04<00:01, 10.08it/s]
Loading safetensors checkpoint shards:  75% Completed | 43/57 [00:04<00:01,  9.02it/s]
Loading safetensors checkpoint shards:  77% Completed | 44/57 [00:04<00:01,  8.95it/s]
Loading safetensors checkpoint shards:  79% Completed | 45/57 [00:05<00:01,  8.67it/s]
Loading safetensors checkpoint shards:  81% Completed | 46/57 [00:05<00:01,  8.64it/s]
Loading safetensors checkpoint shards:  82% Completed | 47/57 [00:05<00:01,  8.67it/s]
Loading safetensors checkpoint shards:  84% Completed | 48/57 [00:05<00:01,  8.71it/s]
Loading safetensors checkpoint shards:  86% Completed | 49/57 [00:05<00:01,  7.83it/s]
Loading safetensors checkpoint shards:  88% Completed | 50/57 [00:05<00:00,  8.00it/s]
Loading safetensors checkpoint shards:  89% Completed | 51/57 [00:05<00:00,  8.13it/s]
Loading safetensors checkpoint shards:  91% Completed | 52/57 [00:05<00:00,  8.24it/s]
Loading safetensors checkpoint shards:  93% Completed | 53/57 [00:06<00:00,  7.52it/s]
Loading safetensors checkpoint shards:  95% Completed | 54/57 [00:06<00:00,  3.68it/s]
Loading safetensors checkpoint shards:  96% Completed | 55/57 [00:06<00:00,  4.20it/s]
Loading safetensors checkpoint shards:  98% Completed | 56/57 [00:06<00:00,  4.99it/s]
Loading safetensors checkpoint shards: 100% Completed | 57/57 [00:06<00:00,  8.21it/s]
(Worker_TP0 pid=23802)
(Worker_TP0 pid=23802) INFO 07-08 01:30:35 [default_loader.py:430] Loading weights took 6.94 seconds
(Worker_TP0 pid=23802) INFO 07-08 01:30:35 [mxfp4.py:1705] Using MoEPrepareAndFinalizeNoDPEPModular
(Worker_TP0 pid=23802) INFO 07-08 01:30:36 [gpu_model_runner.py:5272] Model loading took 113.04 GiB memory and 8.837664 seconds
(Worker_TP0 pid=23802) INFO 07-08 01:30:36 [breakable_cudagraph.py:288] Breakable CUDA graph enabled
(Worker_TP0 pid=23802) INFO 07-08 01:30:36 [utils.py:90] `_KV_CACHE_LAYOUT_OVERRIDE` variable detected. Setting KV cache layout to HND.
(Worker_TP0 pid=23802) INFO 07-08 01:30:36 [gpu_model_runner.py:6288] Encoder cache will be initialized with a budget of 32768 tokens, and profiled with 4 video items of the maximum feature size.
(Worker_TP1 pid=23810) /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/core.py:5791: DeprecationWarning: Use explicit `struct.scalar.ptr` for pointer instead.
(Worker_TP1 pid=23810)   warnings.warn(
(Worker_TP0 pid=23802) /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/core.py:5791: DeprecationWarning: Use explicit `struct.scalar.ptr` for pointer instead.
(Worker_TP0 pid=23802)   warnings.warn(
(Worker_TP1 pid=23810) /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/core.py:5791: DeprecationWarning: Use explicit `struct.scalar.ptr` for pointer instead.
(Worker_TP1 pid=23810)   warnings.warn(
(Worker_TP0 pid=23802) /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/core.py:5791: DeprecationWarning: Use explicit `struct.scalar.ptr` for pointer instead.
(Worker_TP0 pid=23802)   warnings.warn(
(Worker_TP1 pid=23810) /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/core.py:5791: DeprecationWarning: Use explicit `struct.scalar.ptr` for pointer instead.
(Worker_TP1 pid=23810)   warnings.warn(
(Worker_TP0 pid=23802) /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/core.py:5791: DeprecationWarning: Use explicit `struct.scalar.ptr` for pointer instead.
(Worker_TP0 pid=23802)   warnings.warn(
(Worker_TP1 pid=23810) /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/core.py:5791: DeprecationWarning: Use explicit `struct.scalar.ptr` for pointer instead.
(Worker_TP1 pid=23810)   warnings.warn(
(Worker_TP1 pid=23810) /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/core.py:5791: DeprecationWarning: Use explicit `struct.scalar.ptr` for pointer instead.
(Worker_TP1 pid=23810)   warnings.warn(
(Worker_TP0 pid=23802) /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/core.py:5791: DeprecationWarning: Use explicit `struct.scalar.ptr` for pointer instead.
(Worker_TP0 pid=23802)   warnings.warn(
(Worker_TP0 pid=23802) /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/core.py:5791: DeprecationWarning: Use explicit `struct.scalar.ptr` for pointer instead.
(Worker_TP0 pid=23802)   warnings.warn(
(Worker_TP1 pid=23810) /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/core.py:5791: DeprecationWarning: Use explicit `struct.scalar.ptr` for pointer instead.
(Worker_TP1 pid=23810)   warnings.warn(
(Worker_TP0 pid=23802) /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/core.py:5791: DeprecationWarning: Use explicit `struct.scalar.ptr` for pointer instead.
(Worker_TP0 pid=23802)   warnings.warn(
(Worker_TP1 pid=23810) /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/core.py:5791: DeprecationWarning: Use explicit `struct.scalar.ptr` for pointer instead.
(Worker_TP1 pid=23810)   warnings.warn(
(Worker_TP0 pid=23802) /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/core.py:5791: DeprecationWarning: Use explicit `struct.scalar.ptr` for pointer instead.
(Worker_TP0 pid=23802)   warnings.warn(
(Worker_TP1 pid=23810) /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/core.py:5791: DeprecationWarning: Use explicit `struct.scalar.ptr` for pointer instead.
(Worker_TP1 pid=23810)   warnings.warn(
(Worker_TP1 pid=23810) /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/core.py:5791: DeprecationWarning: Use explicit `struct.scalar.ptr` for pointer instead.
(Worker_TP1 pid=23810)   warnings.warn(
(Worker_TP0 pid=23802) /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/core.py:5791: DeprecationWarning: Use explicit `struct.scalar.ptr` for pointer instead.
(Worker_TP0 pid=23802)   warnings.warn(
(Worker_TP0 pid=23802) /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/core.py:5791: DeprecationWarning: Use explicit `struct.scalar.ptr` for pointer instead.
(Worker_TP0 pid=23802)   warnings.warn(
(Worker_TP1 pid=23810) /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/core.py:5791: DeprecationWarning: Use explicit `struct.scalar.ptr` for pointer instead.
(Worker_TP1 pid=23810)   warnings.warn(
(Worker_TP1 pid=23810) /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/core.py:5791: DeprecationWarning: Use explicit `struct.scalar.ptr` for pointer instead.
(Worker_TP1 pid=23810)   warnings.warn(
(Worker_TP0 pid=23802) /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/core.py:5791: DeprecationWarning: Use explicit `struct.scalar.ptr` for pointer instead.
(Worker_TP0 pid=23802)   warnings.warn(
(Worker_TP0 pid=23802) /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/core.py:5791: DeprecationWarning: Use explicit `struct.scalar.ptr` for pointer instead.
(Worker_TP0 pid=23802)   warnings.warn(
(Worker_TP1 pid=23810) /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/core.py:5791: DeprecationWarning: Use explicit `struct.scalar.ptr` for pointer instead.
(Worker_TP1 pid=23810)   warnings.warn(
(Worker_TP1 pid=23810) /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/core.py:5791: DeprecationWarning: Use explicit `struct.scalar.ptr` for pointer instead.
(Worker_TP1 pid=23810)   warnings.warn(
(Worker_TP0 pid=23802) /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/core.py:5791: DeprecationWarning: Use explicit `struct.scalar.ptr` for pointer instead.
(Worker_TP0 pid=23802)   warnings.warn(
(Worker_TP0 pid=23802) /usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages/cutlass/cute/core.py:5791: DeprecationWarning: Use explicit `struct.scalar.ptr` for pointer instead.
(Worker_TP0 pid=23802)   warnings.warn(
(Worker_TP1 pid=23810) INFO 07-08 01:30:58 [gpu_model_runner.py:6500] Profiling CUDA graph memory: PIECEWISE=35 (largest=256), FULL=19 (largest=128)
(Worker_TP0 pid=23802) INFO 07-08 01:30:58 [gpu_model_runner.py:6500] Profiling CUDA graph memory: PIECEWISE=35 (largest=256), FULL=19 (largest=128)
(Worker_TP0 pid=23802) INFO 07-08 01:30:58 [flashinfer_all_reduce.py:112] Auto-selected flashinfer allreduce backend: trtllm
(Worker_TP0 pid=23802) /usr/local/lib/python3.12/dist-packages/torch/distributed/c10d_logger.py:83: UserWarning: barrier(): using the device under current context. You can specify `device_id` in `init_process_group` to mute this warning.
(Worker_TP0 pid=23802)   return func(*args, **kwargs)
(Worker_TP0 pid=23802) INFO 07-08 01:30:59 [flashinfer_all_reduce.py:152] Initialized FlashInfer Allreduce norm fusion workspace with backend=trtllm
(Worker_TP0 pid=23802) INFO 07-08 01:31:10 [flashinfer.py:489] Using TRTLLM attention (--attention-config.use_trtllm_attention is set to 1)
(Worker_TP1 pid=23810) INFO 07-08 01:31:11 [custom_all_reduce.py:213] Registering 114 cuda graph addresses
(Worker_TP0 pid=23802) INFO 07-08 01:31:11 [custom_all_reduce.py:213] Registering 114 cuda graph addresses
(Worker_TP0 pid=23802) INFO 07-08 01:31:12 [gpu_model_runner.py:6605] Estimated CUDA graph memory: 1.97 GiB total
(Worker_TP1 pid=23810) INFO 07-08 01:31:12 [gpu_model_runner.py:6605] Estimated CUDA graph memory: 1.97 GiB total
(Worker_TP0 pid=23802) INFO 07-08 01:31:12 [gpu_worker.py:515] Available KV cache memory: 16.81 GiB
(Worker_TP0 pid=23802) INFO 07-08 01:31:12 [gpu_worker.py:530] CUDA graph memory profiling is enabled (default since v0.21.0). The current --gpu-memory-utilization=0.8000 is equivalent to --gpu-memory-utilization=0.7890 without CUDA graph memory profiling. To maintain the same effective KV cache size as before, increase --gpu-memory-utilization to 0.8110. To disable, set VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0.
(Worker_TP1 pid=23810) INFO 07-08 01:31:12 [gpu_worker.py:530] CUDA graph memory profiling is enabled (default since v0.21.0). The current --gpu-memory-utilization=0.8000 is equivalent to --gpu-memory-utilization=0.7890 without CUDA graph memory profiling. To maintain the same effective KV cache size as before, increase --gpu-memory-utilization to 0.8110. To disable, set VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0.
(EngineCore pid=23537) INFO 07-08 01:31:12 [kv_cache_utils.py:2146] GPU KV cache size: 237,312 tokens
(EngineCore pid=23537) INFO 07-08 01:31:12 [kv_cache_utils.py:2147] Maximum concurrency for 8,192 tokens per request: 28.97x
(Worker_TP0 pid=23802) INFO 07-08 01:31:12 [deep_gemm.py:175] deep_gemm not found in site-packages, trying vendored vllm.third_party.deep_gemm
(Worker_TP0 pid=23802) INFO 07-08 01:31:12 [deep_gemm.py:202] DeepGEMM PDL enabled on vllm.third_party.deep_gemm.
(Worker_TP0 pid=23802) INFO 07-08 01:31:12 [minimax_m3_msa_warmup.py:34] Warming up MiniMax M3 MSA kernels.
(Worker_TP1 pid=23810) INFO 07-08 01:31:12 [minimax_m3_msa_warmup.py:34] Warming up MiniMax M3 MSA kernels.
(Worker_TP0 pid=23802) 2026-07-08 01:31:16,227 - INFO - autotuner.py:651 - flashinfer.jit: [Autotuner]: Autotuning process starts ...
(Worker_TP1 pid=23810) 2026-07-08 01:31:16,227 - INFO - autotuner.py:651 - flashinfer.jit: [Autotuner]: Autotuning process starts ...
[AutoTuner]: Tuning fp4_gemm:   0%|                                                                 | 0/23 [00:00<?, ?profile/s][AutoTuner]: Tuning fp4_gemm:   4%|██▍                                                      | 1/23 [00:27<10:12, 27.85s/profile](EngineCore pid=23537) INFO 07-08 01:32:13 [shm_broadcast.py:705] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization).
(EngineCore pid=23537) INFO 07-08 01:33:13 [shm_broadcast.py:705] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization).
(EngineCore pid=23537) INFO 07-08 01:34:13 [shm_broadcast.py:705] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization).
[AutoTuner]: Tuning fp4_gemm:   9%|████▊                                                   | 2/23 [03:14<38:18, 109.44s/profile]







(EngineCore pid=23537) INFO 07-08 01:35:14 [shm_broadcast.py:705] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization).
[AutoTuner]: Tuning fp4_gemm:  17%|█████████▉                                               | 4/23 [04:02<17:39, 55.77s/profile](EngineCore pid=23537) INFO 07-08 01:36:14 [shm_broadcast.py:705] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization).




(EngineCore pid=23537) INFO 07-08 01:37:14 [shm_broadcast.py:705] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization).
(EngineCore pid=23537) INFO 07-08 01:38:14 [shm_broadcast.py:705] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization).



[AutoTuner]: Tuning fp4_gemm:  35%|███████████████████▊                                     | 8/23 [07:52<12:36, 50.43s/profile](EngineCore pid=23537) INFO 07-08 01:39:14 [shm_broadcast.py:705] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization).
[AutoTuner]: Tuning fp4_gemm: 100%|████████████████████████████████████████████████████████| 23/23 [08:31<00:00, 22.23s/profile]
[AutoTuner]: Tuning fp4_gemm: 100%|████████████████████████████████████████████████████████| 23/23 [08:34<00:00, 22.36s/profile]
[AutoTuner]: Tuning fp4_gemm:   4%|██▍                                                      | 1/23 [00:10<03:45, 10.26s/profile](EngineCore pid=23537) INFO 07-08 01:40:14 [shm_broadcast.py:705] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization).
[AutoTuner]: Tuning fp4_gemm:  35%|███████████████████▊                                     | 8/23 [01:22<01:46,  7.10s/profile](EngineCore pid=23537) INFO 07-08 01:41:14 [shm_broadcast.py:705] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization).
[AutoTuner]: Tuning fp4_gemm:  39%|██████████████████████▎                                  | 9/23 [01:27<02:00,  8.58s/profile]


[AutoTuner]: Tuning fp4_gemm:  43%|████████████████████████▎                               | 10/23 [01:57<03:02, 14.01s/profile](EngineCore pid=23537) INFO 07-08 01:42:14 [shm_broadcast.py:705] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization).
[AutoTuner]: Tuning fp4_gemm: 100%|████████████████████████████████████████████████████████| 23/23 [03:08<00:00,  8.18s/profile]
[AutoTuner]: Tuning fp4_gemm: 100%|████████████████████████████████████████████████████████| 23/23 [03:08<00:00,  8.21s/profile]
[AutoTuner]: Tuning fp4_gemm: 100%|████████████████████████████████████████████████████████| 23/23 [00:13<00:00,  1.70profile/s]
[AutoTuner]: Tuning fp4_gemm:  78%|███████████████████████████████████████████▊            | 18/23 [00:15<00:04,  1.11profile/s](EngineCore pid=23537) INFO 07-08 01:43:14 [shm_broadcast.py:705] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes are hanging or doing some time-consuming work (e.g. compilation, weight/kv cache quantization).
[AutoTuner]: Tuning fp4_gemm: 100%|████████████████████████████████████████████████████████| 23/23 [00:19<00:00,  1.18profile/s]
(Worker_TP1 pid=23810) 2026-07-08 01:43:19,750 - INFO - autotuner.py:674 - flashinfer.jit: [Autotuner]: Autotuning process ends
(Worker_TP0 pid=23802) 2026-07-08 01:43:19,757 - INFO - autotuner.py:674 - flashinfer.jit: [Autotuner]: Autotuning process ends
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|██████████████████████████████████| 35/35 [00:10<00:00,  3.41it/s]
Capturing CUDA graphs (decode, FULL): 100%|█████████████████████████████████████████████████████| 19/19 [00:06<00:00,  3.11it/s]
(Worker_TP0 pid=23802) INFO 07-08 01:43:42 [custom_all_reduce.py:213] Registering 2451 cuda graph addresses
(Worker_TP1 pid=23810) INFO 07-08 01:43:42 [custom_all_reduce.py:213] Registering 2451 cuda graph addresses
(Worker_TP0 pid=23802) INFO 07-08 01:43:43 [gpu_model_runner.py:6673] Graph capturing finished in 24 secs, took 1.32 GiB
(Worker_TP0 pid=23802) INFO 07-08 01:43:43 [gpu_worker.py:748] CUDA graph pool memory: 1.32 GiB (actual), 1.97 GiB (estimated), difference: 0.65 GiB (49.2%).
(Worker_TP1 pid=23810) INFO 07-08 01:43:43 [gpu_worker.py:748] CUDA graph pool memory: 1.32 GiB (actual), 1.97 GiB (estimated), difference: 0.65 GiB (49.2%).
(Worker_TP0 pid=23802) INFO 07-08 01:43:43 [jit_monitor.py:71] Kernel JIT monitor activated; monitored JIT compilations during inference will use mode=warn.
(Worker_TP1 pid=23810) INFO 07-08 01:43:43 [jit_monitor.py:71] Kernel JIT monitor activated; monitored JIT compilations during inference will use mode=warn.
(EngineCore pid=23537) INFO 07-08 01:43:43 [core.py:344] init engine (profile, create kv cache, warmup model) took 787.17 s
(EngineCore pid=23537) INFO 07-08 01:43:49 [vllm.py:1006] Asynchronous scheduling is enabled.
(EngineCore pid=23537) WARNING 07-08 01:43:49 [vllm.py:1100] VLLM_USE_BREAKABLE_CUDAGRAPH is set, disabling vLLM's torch.compile pipeline. Equivalent to -cc.mode=none.
(EngineCore pid=23537) WARNING 07-08 01:43:49 [vllm.py:1110] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.
(EngineCore pid=23537) INFO 07-08 01:43:49 [kernel.py:278] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['vllm_c', 'native'], fused_add_rms_norm=['vllm_c', 'native'])
(EngineCore pid=23537) INFO 07-08 01:43:49 [compilation.py:310] Enabled custom fusions: norm_quant, act_quant, allreduce_rms
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
2026-07-08:01:43:53 INFO     [evaluator_utils:446] Selected tasks:
2026-07-08:01:43:53 INFO     [evaluator_utils:480] Task: gsm8k (gsm8k/gsm8k.yaml)
2026-07-08:01:43:53 INFO     [evaluator:314] gsm8k: Using gen_kwargs: {'until': ['Question:', '</s>', '<|im_end|>'], 'do_sample': False, 'temperature': 0.0}
2026-07-08:01:43:53 INFO     [api.task:312] Building contexts for gsm8k on rank 0...
100%|██████████████████████████████████████████████████████████████████████████████████████| 1319/1319 [00:04<00:00, 286.42it/s]
2026-07-08:01:43:57 INFO     [evaluator:585] Running generate_until requests
Running generate_until requests:   0%|                                                                 | 0/1319 [00:00<?, ?it/s](Worker_TP0 pid=23802) WARNING 07-08 01:44:00 [jit_monitor.py:127] Triton kernel JIT compilation during inference: _compute_slot_mapping_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.
Running generate_until requests:   3%|█▍                                                      | 33/1319 [00:54<30:16,  1.41s/it]

Running generate_until requests:  10%|█████▍                                                 | 129/1319 [02:12<17:44,  1.12it/s]


Running generate_until requests:  22%|████████████                                           | 289/1319 [04:45<17:12,  1.00s/it](Worker_TP0 pid=23802) WARNING 07-08 01:48:46 [jit_monitor.py:127] Triton kernel JIT compilation during inference: _gqa_sparse_decode_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.
Running generate_until requests:  27%|██████████████▋                                        | 353/1319 [05:56<17:00,  1.06s/it](Worker_TP0 pid=23802) WARNING 07-08 01:49:57 [jit_monitor.py:127] Triton kernel JIT compilation during inference: _topk_index_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.




Running generate_until requests:  36%|████████████████████                                   | 481/1319 [08:21<15:41,  1.12s/it]


Running generate_until requests:  44%|████████████████████████                               | 577/1319 [10:22<15:07,  1.22s/it]



Running generate_until requests:  53%|█████████████████████████████▍                         | 705/1319 [13:21<13:34,  1.33s/it]





























Running generate_until requests:  63%|██████████████████████████████████▋                    | 833/1319 [16:19<11:37,  1.44s/it]



Running generate_until requests:  83%|████████████████████████████████████████████▌         | 1089/1319 [22:19<05:24,  1.41s/it]



Running generate_until requests:  85%|█████████████████████████████████████████████▉        | 1121/1319 [23:07<04:43,  1.43s/it]


Running generate_until requests: 100%|██████████████████████████████████████████████████████| 1319/1319 [27:08<00:00,  1.23s/it]
fatal: not a git repository (or any parent up to mount point /)
Stopping at filesystem boundary (GIT_DISCOVERY_ACROSS_FILESYSTEM not set).
2026-07-08:02:11:12 INFO     [loggers.evaluation_tracker:247] Saving results aggregated
vllm ({'pretrained': './M3-rtn-auto-vllm', 'tensor_parallel_size': 2, 'max_model_len': 8192, 'max_num_batched_tokens': 32768, 'max_num_seqs': 128, 'add_bos_token': True, 'gpu_memory_utilization': 0.8, 'dtype': 'bfloat16', 'max_gen_toks': 2048, 'enable_prefix_caching': False, 'reasoning_parser': 'minimax_m3'}), gen_kwargs: ({}), limit: None, num_fewshot: None, batch_size: 32
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |    0|±  |     0|
|     |       |strict-match    |     5|exact_match|↑  |    0|±  |     0|

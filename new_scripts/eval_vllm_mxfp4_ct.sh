export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ZE_AFFINITY_MASK=0,1,2,3
export ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3

NUM_GPUS=4
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}

#MODEL=Qwen3-8B_autoround_mxfp4_rtn_ct/Qwen3-8B-mxfp-w4g32/
MODEL=Qwen3-8B_autoround_mxfp4_rtn_ct/Qwen3-8B-mxfp-w4g32/
#MODEL_ARGS="pretrained=${MODEL},tensor_parallel_size=${NUM_GPUS},max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,add_bos_token=True,gpu_memory_utilization=0.8,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False,enforce_eager=True"
MODEL_ARGS="pretrained=${MODEL},tensor_parallel_size=${NUM_GPUS},max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,add_bos_token=True,gpu_memory_utilization=0.8,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False"

lm_eval \
  --model vllm \
  --model_args $MODEL_ARGS \
  --tasks piqa \
  --batch_size 8 \
  --output_path lm_eval_results

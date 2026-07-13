export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ZE_AFFINITY_MASK=0,1
export ONEAPI_DEVICE_SELECTOR=level_zero:0,1

NUM_GPUS=2
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}

#MODEL=Qwen3-30B-A3B_autoround_w4a16_rtn_ar/Qwen3-30B-A3B-w4g32/
MODEL=Intel/diffusiongemma-26B-A4B-it-int4-AutoRound
#MODEL_ARGS="pretrained=${MODEL},tensor_parallel_size=${NUM_GPUS},max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,add_bos_token=True,gpu_memory_utilization=0.8,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False,enforce_eager=True"
MODEL_ARGS="pretrained=${MODEL},tensor_parallel_size=${NUM_GPUS},max_model_len=4096,max_num_batched_tokens=8192,max_num_seqs=8,add_bos_token=True,gpu_memory_utilization=0.9,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False,enforce_eager=True"

lm_eval \
  --model vllm \
  --model_args $MODEL_ARGS \
  --tasks piqa \
  --batch_size 2 \
  --output_path lm_eval_results

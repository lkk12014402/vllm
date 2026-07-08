export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

NUM_GPUS=2
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}

MODEL=/storage/lkk/m3/MiniMax-M3-MXFP4
MODEL_ARGS="pretrained=${MODEL},tensor_parallel_size=${NUM_GPUS},max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,add_bos_token=True,gpu_memory_utilization=0.9,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False,reasoning_parser=minimax_m3"

lm_eval \
  --model vllm \
  --model_args $MODEL_ARGS \
  --tasks piqa,mmlu,gsm8k,hellaswag \
  --batch_size 64 \
  --output_path lm_eval_results



MODEL=/storage/lkk/m3/MiniMax-M3-MXFP4-ignore-mlp-all
MODEL_ARGS="pretrained=${MODEL},tensor_parallel_size=${NUM_GPUS},max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,add_bos_token=True,gpu_memory_utilization=0.9,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False,reasoning_parser=minimax_m3"

lm_eval \
  --model vllm \
  --model_args $MODEL_ARGS \
  --tasks piqa,mmlu,gsm8k,hellaswag \
  --batch_size 64 \
  --output_path lm_eval_results

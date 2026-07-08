export CUDA_VISIBLE_DEVICES=1,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn

NUM_GPUS=2
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}


MODEL=/storage/lkk/m3/MiniMax-M3-MXFP4-ignore-mlp-all
#MODEL=/storage/lkk/MiniMax-M3-amd
#MODEL=/storage/lkk/m3/MiniMax-M3-MXFP4-ignore-mlp-all-tuning/MiniMax-M3-mxfp-w4g32
#MODEL_ARGS="pretrained=${MODEL},tensor_parallel_size=${NUM_GPUS},max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,add_bos_token=True,gpu_memory_utilization=0.9,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False,reasoning_parser=minimax_m3,moe_backend=emulation"
MODEL=./M3-rtn-native-vllm/
MODEL=./M3-rtn-auto-vllm
MODEL=./MiniMax-M3-MXFP4MoE-MXFP8-conservative/
MODEL=./MiniMax-M3-MXFP4MoE-MXFP8-attn
MODEL_ARGS="pretrained=${MODEL},tensor_parallel_size=${NUM_GPUS},max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,add_bos_token=True,gpu_memory_utilization=0.8,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False,reasoning_parser=minimax_m3"

lm_eval \
  --model vllm \
  --model_args $MODEL_ARGS \
  --tasks gsm8k \
  --batch_size 64 \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --output_path lm_eval_results

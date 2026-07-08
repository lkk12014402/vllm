
export CUDA_VISIBLE_DEVICES=2,3

vllm serve \
  /storage/lkk/m3/MiniMax-M3-MXFP4MoE-MXFP8-conservative \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --tool-call-parser minimax_m3 \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice \
  --port 8008

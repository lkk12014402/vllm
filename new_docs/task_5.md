对minimax-m3，这是模型路径：/storage/lkk/MiniMax-M3，我们之前讨论走model-free的量化命令是auto-round ../MiniMax-M3/ --model_free --scheme MXFP4 \
  --ignore_layers vision_tower,lm_head,block_sparse_moe.gate,embed_tokens,self_attn,patch_merge_mlp,multi_modal_projector,mlp.gate_proj,mlp.up_proj,mlp.down_proj \
  --format llm_compressor --output_dir "./MiniMax-M3-MXFP4-ignore-mlp-all"，你帮我再次确认ignore_layers的配置是否正确？？

因为这种方式量化的模型精度稍微差一些，根绝以往的经验，moe的部分量化为mxfp4，其它的量化为mxfp8，这是一个示例auto-round deepseek-ai/DeepSeek-V4-Pro  \
  --model_free \
  --scheme MXFP8 \
  --ignore_layers compressor,indexer.weights_proj \
  --layer_config "{ffn.experts:{bits:4,data_type:mx_fp}} \
  --format llm_compressor \
  --output_dir "./DeepSeek-V4-Pro-MXFP4-Mixed"

所以你能帮我把Minimax-m3的量化，也改成类似这样的量化命令吗

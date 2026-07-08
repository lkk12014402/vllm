

## use model-free path

auto-round ../MiniMax-M3/ --model_free --scheme MXFP4 \
  --ignore_layers vision_tower,lm_head,block_sparse_moe.gate,embed_tokens,self_attn,patch_merge_mlp,multi_modal_projector,mlp.gate_proj,mlp.up_proj,mlp.down_proj \
  --format llm_compressor --output_dir "./MiniMax-M3-MXFP4-ignore-mlp-all"


## not use model-free path

root@ip-172-31-32-47:/storage/lkk/m3# auto-round ../MiniMax-M3/ --scheme MXFP4 --iters 0  --ignore_layers vision_tower,lm_head,block_sparse_moe.gate,embed_tokens,self_attn,patch_merge_mlp,multi_modal_projector,mlp.gate_proj,mlp.up_proj,mlp.down_proj   --format llm_compressor --output_dir "./MiniMax-M3-MXFP4-ignore-mlp-all-rtn"

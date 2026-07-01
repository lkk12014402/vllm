
我们的量化

auto-round ../MiniMax-M3/ --model_free --scheme MXFP4 \
  --ignore_layers visual,lm_head,block_sparse_moe.gate,linear_attn,shared_expert_gate,embed_tokens,self_attn \
  --format llm_compressor --output_dir "./MiniMax-M3-MXFP4"

amd的量化

from quark.torch import LLMTemplate, ModelQuantizer

# --- Register template ---
minimax_m3_vl_template = LLMTemplate(
    model_type="minimax_m3_vl",
    kv_layers_name=["*language_model.*k_proj", "*language_model.*v_proj"],
    q_layer_name="*language_model.*q_proj",
    exclude_layers_name=[
        "*lm_head",
        "*vision_tower*",
        "*multi_modal_projector*",
        "*patch_merge_mlp*",
        "*block_sparse_moe.gate",
        "*self_attn*",
    ],
)
LLMTemplate.register_template(minimax_m3_vl_template)
print(f"[INFO]: Registered template '{minimax_m3_vl_template.model_type}'")

# --- Configuration ---
model_dir = "MiniMaxAI/MiniMax-M3"
output_dir = "amd/MiniMax-M3-MXFP4"
quant_scheme = "mxfp4"
exclude_layers = [
    "*lm_head",
    "*vision_tower*",
    "*multi_modal_projector*",
    "*patch_merge_mlp*",
    "*block_sparse_moe.gate",
    "*self_attn*",
    "*mlp.gate_proj",
    "*mlp.up_proj",
    "*mlp.down_proj",
]

# --- Build quant config from template ---
template = LLMTemplate.get("minimax_m3_vl")
quant_config = template.get_config(scheme=quant_scheme, exclude_layers=exclude_layers)

# --- File-to-file quantization (memory-efficient, no full model loading) ---
quantizer = ModelQuantizer(quant_config)
quantizer.direct_quantize_checkpoint(
    pretrained_model_path=model_dir,
    save_path=output_dir,
)
print(f"[INFO]: Quantization complete. Output saved to {output_dir}")



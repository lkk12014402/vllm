from auto_round import AutoRound

# Load a model (supports FP8/BF16/FP16/FP32)
#model_name_or_path = "Qwen/Qwen3-8B"
model_name_or_path = "Qwen/Qwen3-30B-A3B"
output_dir = "./Qwen3-30B-A3B_autoround_w4a16_rtn_ct"

ar = AutoRound(model_name_or_path, scheme="W4A16", iters=0, device_map="auto", low_gpu_mem_usage=True)


ar.quantize_and_save(output_dir=output_dir, format="llm_compressor")

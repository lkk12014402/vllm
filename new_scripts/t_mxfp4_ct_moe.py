from auto_round import AutoRound

# Load a model (supports FP8/BF16/FP16/FP32)
model_name_or_path = "Qwen/Qwen3-30B-A3B"
output_dir = "./Qwen3-30B-A3B_autoround_mxfp4_rtn_ct"

ar = AutoRound(model_name_or_path, scheme="MXFP4", iters=0, device_map="auto", low_gpu_mem_usage=True)


ar.quantize_and_save(output_dir=output_dir, format="llm_compressor")

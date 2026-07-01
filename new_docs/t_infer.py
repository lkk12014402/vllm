import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.image_utils import load_image


model = AutoModelForImageTextToText.from_pretrained(
    "/storage/lkk/MiniMax-M3", dtype=torch.bfloat16, device_map="auto",
)
processor = AutoProcessor.from_pretrained("/storage/lkk/MiniMax-M3", trust_remote_code=True)

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image briefly."},
        ],
    }
]
text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = processor(images=[image], text=text, return_tensors="pt").to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])

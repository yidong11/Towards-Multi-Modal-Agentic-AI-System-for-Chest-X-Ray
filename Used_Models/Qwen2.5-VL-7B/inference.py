import io
import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import warnings
import csv
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)

# Or the accelerated version with flash attention 2
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

def qwen(model, processor, img_path):
    prompt = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_path,
                },
                {"type": "text", "text": "Please generate report for this chest X-ray image."},
            ],
        }
    ]

    text = processor.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(prompt)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text


# Use Samples:
base_path = 'IMAGE_PATH'  # Base path for images
input_csv_path = 'INPUT_CSV'  # Path to the input CSV file
output_csv_path = 'OUTPUT_CSV'

os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

start_line = 0
max_lines = None  # Maximum number of lines to process (None means process all)

with open(input_csv_path, mode='r', encoding='utf-8') as infile, \
     open(output_csv_path, mode='a', newline='', encoding='utf-8') as outfile:

    reader = csv.DictReader(infile)
    writer = csv.writer(outfile)

    if os.stat(output_csv_path).st_size == 0:
        writer.writerow(['img_path', 'report'])

    for _ in range(start_line):
        next(reader, None)

    for idx, row in enumerate(reader):
        if max_lines is not None and idx >= max_lines:
            break

        img_paths = row['img_path'].split(';')  # Split img_path into multiple paths

        for img_path in img_paths:
            img_path = img_path.strip()

            full_img_path = os.path.join(base_path, img_path)
            classification_result = qwen(model, processor, full_img_path)
            classification_result = classification_result.replace('\n', ' ')
            writer.writerow([img_path, classification_result])
            print(f"Processed image: {full_img_path}")

print(f"Results appended to [{output_csv_path}]")

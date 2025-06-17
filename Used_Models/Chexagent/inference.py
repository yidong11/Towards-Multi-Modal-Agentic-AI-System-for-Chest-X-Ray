import io
import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import warnings
import csv
import os
torch.cuda.empty_cache()

def cheXagent_generate(images, prompt, processor, model, device, dtype, generation_config):
    inputs = processor(
        images=images[:2], text=f" USER: <s>{prompt} ASSISTANT: <s>", return_tensors="pt"
    ).to(device=device, dtype=dtype)
    output = model.generate(**inputs, generation_config=generation_config)[0]
    response = processor.tokenizer.decode(output, skip_special_tokens=True) 
    return response


def cheXagent(local_image_path, processor, model, device, dtype, generation_config):

    images = [Image.open(local_image_path).convert("RGB")]

    anatomies = [
        "Airway", "Breathing", "Cardiac", "Diaphragm",
        "Everything else (e.g., mediastinal contours, bones, soft tissues, tubes, valves, and pacemakers)"
    ]

    all_responses = ""
    for anatomy in anatomies:
        prompt = f'Describe "{anatomy}"'
        response = cheXagent_generate(images, prompt, processor, model, device, dtype, generation_config)
        all_responses += f"[{anatomy}]:\n{response}\n"
    return all_responses

# Use Samples:
base_path = 'IMAGE_PATH'  # Base path for images
input_csv_path = 'INPUT_CSV'  # Path to the input CSV file
output_csv_path = 'OUTPUT_CSV'

device = "cuda"
dtype = torch.float16
processor = AutoProcessor.from_pretrained("StanfordAIMI/CheXagent-8b", trust_remote_code=True)
generation_config = GenerationConfig.from_pretrained("StanfordAIMI/CheXagent-8b")
model = AutoModelForCausalLM.from_pretrained(
    "StanfordAIMI/CheXagent-8b", torch_dtype=dtype, trust_remote_code=True
).to(device)

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
            classification_result = cheXagent(full_img_path, processor, model, device, dtype, generation_config)
            classification_result = classification_result.replace('\n', ' ')
            writer.writerow([img_path, classification_result])
            print(f"Processed image: {full_img_path}")

print(f"Results appended to [{output_csv_path}]")

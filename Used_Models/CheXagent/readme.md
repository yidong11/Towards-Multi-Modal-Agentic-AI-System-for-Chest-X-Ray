# Chexagent Implementation

## Chexagent: 

- **Overview**  
 CheXagent is an instruction-tuned FM capable of analyzing and summarizing CXRs. We apply it to our dataset and generate corresponding anatomical report. Source: [https://huggingface.co/StanfordAIMI/CheXagent-2-3b](https://huggingface.co/StanfordAIMI/CheXagent-2-3b)

- **Start to use** 

  You can start to generate a anatomical report from a CXR image by using the following python code. If you need to batch process the data you can refer to the code in `inference.py`.

```python

import io

import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

# step 1: Setup constant
model_name = "StanfordAIMI/CheXagent-2-3b"
dtype = torch.bfloat16
device = "cuda"

# step 2: Load Processor and Model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
model = model.to(dtype)
model.eval()

# step 3: Inference
query = tokenizer.from_list_format([*[{'image': path} for path in paths], {'text': prompt}])
conv = [{"from": "system", "value": "You are a helpful assistant."}, {"from": "human", "value": query}]
input_ids = tokenizer.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt")
output = model.generate(
    input_ids.to(device), do_sample=False, num_beams=1, temperature=1., top_p=1., use_cache=True,
    max_new_tokens=512
)[0]
response = tokenizer.decode(output[input_ids.size(1):-1])

```

- ## Reference

```
@article{chexagent-2024,
  title={CheXagent: Towards a Foundation Model for Chest X-Ray Interpretation},
  author={Chen, Zhihong and Varma, Maya and Delbrouck, Jean-Benoit and Paschali, Magdalini and Blankemeier, Louis and Veen, Dave Van and Valanarasu, Jeya Maria Jose and Youssef, Alaa and Cohen, Joseph Paul and Reis, Eduardo Pontes and Tsai, Emily B. and Johnston, Andrew and Olsen, Cameron and Abraham, Tanishq Mathew and Gatidis, Sergios and Chaudhari, Akshay S and Langlotz, Curtis},
  journal={arXiv preprint arXiv:2401.12208},
  url={https://arxiv.org/abs/2401.12208},
  year={2024}
}

```
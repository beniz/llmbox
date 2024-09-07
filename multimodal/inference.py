from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import torch

model = PaliGemmaForConditionalGeneration.from_pretrained("full2-448/checkpoints/checkpoint-2629/").to("cuda")
processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-448")

custom_prompt = """Below is an email message as input. Decide whether it is spam or ham. In response, if spam write as spam otherwise write ham."""
img_path = "spam/probaly_spam_llmbox_email_162.png"

import torchvision.transforms as transforms
transform_rs = transforms.Resize((448, 448))

import sys
image = Image.open(img_path)
image = transform_rs(image)
inputs = processor(text=custom_prompt, images=image, return_tensors="pt").to("cuda")
linputs = inputs['input_ids'].shape[1]

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=1, do_sample=False, output_scores=True, return_dict_in_generate=True)
print(type(generate_ids))
print(generate_ids.keys())
print('scores=', generate_ids['scores'])
print('len(scores)=', len(generate_ids['scores']))
print('scores shape=', generate_ids['scores'][0].shape)
print('sequences=', generate_ids['sequences'])
out = processor.batch_decode(generate_ids['sequences'][:, linputs:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(out)
confidences = generate_ids['scores'][0]
confidences = torch.nn.functional.softmax(confidences, dim=-1)
confidence = torch.max(confidences, dim=-1)[0]
print('confidence=', confidence.item())

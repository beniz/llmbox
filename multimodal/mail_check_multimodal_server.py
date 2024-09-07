from fastapi import FastAPI, File, UploadFile
from typing_extensions import Annotated
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import torch

import math
import argparse

arg_parser = argparse.ArgumentParser(description='spam multimodal LLM server')
arg_parser.add_argument('--host', type=str, default="localhost", help='server host')
arg_parser.add_argument('--port', type=int, default=8236, help='server port')
arg_parser.add_argument('--model-name', type=str, default="model-448/checkpoints/checkpoint-1000/", help="model location to use for spam classification")
arg_parser.add_argument('--use-crops', action='store_true', help="use crops for spam classification (resize is the default)")
arg_parser.add_argument('--no-resize', action='store_true', help="do not resize images")
args = arg_parser.parse_args()

# load model
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
model = PaliGemmaForConditionalGeneration.from_pretrained("full2-448/checkpoints/checkpoint-2629/").to("cuda")
processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-448")
custom_prompt = """Below is an email message as input. Decide whether it is spam or ham. In response, if spam write as spam otherwise write ham."""

import torchvision.transforms as transforms
transform_rs = transforms.Resize(448)

import torchvision.transforms.v2 as transformsv2
transform_v2_rc = transformsv2.RandomCrop((224,224), pad_if_needed=True, padding_mode='constant')

# API server
app = FastAPI()

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float

no_crops = not args.use_crops
no_resize = args.no_resize
    
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # read the file content into memory
    image_bytes = await file.read()
    
    # convert bytes into PIL image
    image = Image.open(BytesIO(image_bytes))

    # resize or crop image
    if no_crops and not no_resize:
        image = transform_rs(image)
    elif not no_crops:
        image = transform_v2_rc(image)
    
    # process inputs 
    inputs = processor(text=custom_prompt, images=image, return_tensors="pt").to("cuda")
    prediction = model.generate(**inputs, temperature=1.0, do_sample=False, max_new_tokens=1, top_k=5, output_scores=True, return_dict_in_generate=True)
    linputs = inputs['input_ids'].shape[1]

    # class prediction
    decoded_text = processor.batch_decode(prediction['sequences'][:,linputs:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    #print('decoded_text=', decoded_text)

    # confidence
    confidences = prediction['scores'][0]
    confidences = torch.nn.functional.softmax(confidences, dim=-1)
    conf = torch.max(confidences, dim=-1)[0].item()
    
    return PredictionResponse(prediction=decoded_text, confidence=conf)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

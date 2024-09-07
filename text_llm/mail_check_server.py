from fastapi import FastAPI
from pydantic import BaseModel
import math
import argparse

arg_parser = argparse.ArgumentParser(description='spam LLM server')
arg_parser.add_argument('--host', type=str, default="localhost", help='server host')
arg_parser.add_argument('--port', type=int, default=8235, help='server port')
arg_parser.add_argument('--model-name', type=str, default="gemma2mail1/lora", help="model location to use for spam classification")
args = arg_parser.parse_args()

# load model
from unsloth import FastLanguageModel

max_seq_length = 8192
load_in_4bit = True
dtype = None

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = args.model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

app = FastAPI()

class PredictionRequest(BaseModel):
    input: str

class PredictionResponse(BaseModel):
    prediction: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    
    max_length = len(request.input)
    inputs = tokenizer(request.input, return_tensors = 'pt', padding='max_length', max_length=max_length)
    prediction = model.generate(**inputs, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)

    decoded_text = tokenizer.decode(prediction[0][-1], skip_special_tokens=True).rstrip().lstrip()
    #print('decoded_text=', decoded_text)
    
    return PredictionResponse(prediction=decoded_text)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

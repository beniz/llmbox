from fastapi import FastAPI
from pydantic import BaseModel
import math

max_seq_length = 8192
dtype = None

from llama_cpp import Llama

llm = Llama(
    model_path="qwen0.5_orig/ollama/unsloth.F16.gguf",
    n_ctx=max_seq_length, # Uncomment to increase the context window
    temperature=1,
    prompt_template='',
)

app = FastAPI()

class PredictionRequest(BaseModel):
    input: str

class PredictionResponse(BaseModel):
    prediction: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    
    prediction = llm(request.input, max_tokens=1, echo=False)
    print('prediction=', prediction)
    
    decoded_text = prediction['choices'][0]['text'].rstrip().lstrip()
    print(decoded_text)
    return PredictionResponse(prediction=decoded_text)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8250)

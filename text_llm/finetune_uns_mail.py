import torch
from unsloth import FastLanguageModel
#import wandb
#wandb.init(mode="disabled")
import os
os.environ['WANDB_DISABLED'] = 'true'

import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+

arg_parser = argparse.ArgumentParser(description='spam filter training')
arg_parser.add_argument('--model', type=str, help='model name to finetune', default="unsloth/Meta-Llama-3.1-8B")
arg_parser.add_argument('--train-data', help='json file to train data', required=True)
arg_parser.add_argument('--test-data', help='json file to test data', required=True)
arg_parser.add_argument('--context-size', type=int, default=16384, help='model context size')
arg_parser.add_argument('--not-4bit', action='store_true', help='not using 4bit model')
arg_parser.add_argument('--batch-size', type=int, default=1, help='per device batch size')
arg_parser.add_argument('--iter-size', type=int, default=64, help='gradient accumulation')
arg_parser.add_argument('--eval-steps', type=int, default=5, help='number of steps between evaluations')
arg_parser.add_argument('--save-steps', type=int, default=5, help='number of steps the model is saved')
arg_parser.add_argument('--nepochs', type=int, default=1, help='number of training epochs')
arg_parser.add_argument('--output-dir', required=True, type=str, help='output dir')
arg_parser.add_argument('--resume', action='store_true', help='whether to resume training')
args = arg_parser.parse_args()

checkpoints_dir = args.output_dir + '/outputs'
lora_dir = args.output_dir + '/lora'
try:
    os.mkdir(checkpoints_dir)
    os.mkdir(lora_dir)
except:
    print('model dirs already exists')
    pass

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
# full model list: https://huggingface.co/unsloth
fourbit_models = [
"unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 15 trillion tokens model 2x faster!
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # We also uploaded 4bit for 405b!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # New Mistral 12b 2x faster!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",          # Phi-3 2x faster!d
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
    "unsloth/gemma-2b-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = args.model,
    max_seq_length = args.context_size,
    dtype = dtype,
    load_in_4bit = True if not args.not_4bit else False
)

# adding LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = 'unsloth',
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# metrics
def preprocess_logits_for_metrics(logits, labels):
    # argmax to get the token ids
    lgm = logits.argmax(dim=-1)
    return lgm

def compute_metrics(pred):

    preds, labels = pred
    labels = labels[:, 1:]
    preds = preds[:, :-1]

    # -100 is a default value for ignore_index used by DataCollatorForCompletionOnlyLM
    mask = labels == -100
    # replace -100 with a value that the tokenizer can decode
    labels[mask] = tokenizer.pad_token_id
    preds[mask] = tokenizer.pad_token_id
    
    # we have to translate from token ids to text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    decoded_labels_int = []
    for dl in decoded_labels:
        if dl == ' Ham':
            decoded_labels_int.append(0)
        elif dl == ' Spam':
            decoded_labels_int.append(1)
        else:
            decoded_labels_int.append(2) # never occurs
    decoded_preds_int = []
    for dp in decoded_preds:
        if dp == ' Ham':
            decoded_preds_int.append(0)
        elif dp == ' Spam':
            decoded_preds_int.append(1)
        else:
            decoded_preds_int.append(2)
    
    
    em = sum([1 if p == l else 0 for p, l in zip(decoded_preds, decoded_labels)]) / len(decoded_labels) # exact match
    accuracy = accuracy_score(decoded_labels_int, decoded_preds_int)
    precision = precision_score(decoded_labels_int, decoded_preds_int, average='weighted')
    recall = recall_score(decoded_labels_int, decoded_preds_int, average='weighted')
    f1_1 = 2 * (precision * recall) / (precision + recall)
    f1_2 = f1_score(decoded_labels_int, decoded_preds_int, average='weighted')
    
    return {
        'f1_1': f1_1,
        'f1_2': f1_2,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'em': em
    }

# Dataset
# custom dataset
benimail_prompt = """Below is an email message as input. Decide whether it is spam or ham. In response, if spam write as Spam otherwise write Ham.

### Input: {}

### Response: {}"""

def formatting_prompts_func(examples):
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for input, output in zip(inputs, outputs):
        text = benimail_prompt.format(input, output)
        texts.append(text)
    return texts

from datasets import load_dataset
dataset = load_dataset("json", data_files=args.train_data, split = "train")
test_dataset = load_dataset("json", data_files=args.test_data, split = "train")

from trl import DataCollatorForCompletionOnlyLM
response_template = "### Response:"
collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

# Training
from trl import SFTTrainer
from transformers import TrainingArguments, Seq2SeqTrainingArguments
import transformers

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    formatting_func = formatting_prompts_func,
    data_collator = collator,
    #dataset_text_field = "text",
    compute_metrics = compute_metrics,
    preprocess_logits_for_metrics = preprocess_logits_for_metrics,
    eval_dataset = test_dataset,
    max_seq_length = args.context_size,
    dataset_num_proc = 8,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = args.batch_size,
        gradient_accumulation_steps = args.iter_size,
        warmup_steps = 5,
        num_train_epochs = args.nepochs,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        eval_strategy = 'steps',
        eval_steps = args.eval_steps,
        output_dir = checkpoints_dir,
        resume_from_checkpoint = checkpoints_dir if args.resume else None,
        save_steps = args.save_steps,
        report_to=None,
    ),
)

# GPU stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# Training loop
trainer_stats = trainer.train(resume_from_checkpoint=args.resume)

# save lora
model.save_pretrained(lora_dir)
tokenizer.save_pretrained(lora_dir)


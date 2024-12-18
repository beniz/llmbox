import sys
import os
os.environ['WANDB_DISABLED'] = 'true'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
arg_parser = argparse.ArgumentParser(description='ft multimodal spamfilter')
arg_parser.add_argument('--model-id', type=str, default="google/paligemma-3b-mix-448")
arg_parser.add_argument('--batch-size', type=int, default=2)
arg_parser.add_argument('--iter-size', type=int, default=64, help='gradient accumulation')
arg_parser.add_argument('--eval-steps', type=int, default=10, help='number of steps between evaluations')
arg_parser.add_argument('--save-steps', type=int, default=10, help='number of steps the model is saved')
arg_parser.add_argument('--nepochs', type=int, default=1, help='number of training epochs')
arg_parser.add_argument('--output-dir', required=True, type=str, help='output dir')
arg_parser.add_argument('--resume', type=int, default=-1, help='whether to resume training, must indicate checkpoint iteration')
arg_parser.add_argument('--min-img-size', type=int, default=448)
arg_parser.add_argument('--use-crops', action='store_true', help='whether to use crops (otherwise image is resized)')
arg_parser.add_argument('--no-resize', action='store_true', help='whether to resize images')
args = arg_parser.parse_args()

checkpoints_dir = args.output_dir + '/checkpoints/'
try:
    os.mkdir(args.output_dir)
    os.mkdir(checkpoints_dir)
except:
    print('model dirs already exist')
    pass

import numpy as np
from transformers import PaliGemmaProcessor 
#model_id = "google/paligemma-3b-pt-224"
#model_id = "google/paligemma-3b-pt-448"
#model_id = "google/paligemma-3b-pt-896"
model_id = args.model_id
processor = PaliGemmaProcessor.from_pretrained(model_id)

import torch
device = torch.cuda.current_device()

image_token = processor.tokenizer.convert_tokens_to_ids("<image>")

from datasets import load_dataset, Image
ds = load_dataset('KagglingFace/vit-cats-dogs', split='train')
shuffled_ds = ds.shuffle(seed=42)
split_dataset = shuffled_ds.train_test_split(test_size=0.01)
train_ds = split_dataset['train']
val_ds = split_dataset['test']
#train_ds = load_dataset('KagglingFace/vit-cats-dogs', split='train[0%:99%]')
#val_ds = load_dataset('KagglingFace/vit-cats-dogs', split='train[99%:]')

print('train_ds.shape:', train_ds.shape)
print('val_ds.shape:', val_ds.shape)

custom_prompt = """<image>Determine whether the image depicts a cat or a dog.<bos>""" #<bos>

import torchvision.transforms as transforms
transform_rs = transforms.Resize(args.min_img_size)

if args.no_resize:
    tr = None # original size
else:
    tr = transform_rs # resize is default

def collate_fn(examples):
    texts = [custom_prompt for example in examples]
    
    images = [example["image"] for example in examples]
    labels = [example["label"] for example in examples]
    labels = ['dog' if label == 1 else 'cat' for label in labels]
    
    images = [image.convert("RGB") for image in images]

    if tr is not None:
        images = [tr(image) for image in images]
    
    tokens = processor(text=texts, images=images, suffix=labels,
                       return_tensors="pt", padding="longest")
    tokens = tokens.to(torch.bfloat16).to(device)
    return tokens

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def preprocess_logits_for_metrics(logits, labels):
    # argmax to get the token ids
    lgm = logits[0].argmax(dim=-1)
    return lgm

def compute_metrics(pred):
    labels = pred.label_ids[:, 1:]
    #print('labels=', labels)
    # -100 is a default value for ignore_index used by DataCollatorForCompletionOnlyLM
    mask = labels == -100
    # replace -100 with a value that the tokenizer can decode
    labels[mask] = processor.tokenizer.pad_token_id
    decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

    preds = pred.predictions[:,:-1]
    preds[mask] = processor.tokenizer.pad_token_id
    decoded_preds = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    decoded_labels_int = []
    for dl in decoded_labels:
        if dl == 'cat':
            decoded_labels_int.append(0)
        elif dl == 'dog':
            decoded_labels_int.append(1)
        else:
            decoded_labels_int.append(2) # never occurs

    decoded_preds_int = []
    for dp in decoded_preds:
        if dp == 'cat':
            decoded_preds_int.append(0)
        elif dp == 'dog':
            decoded_preds_int.append(1)
        else:
            decoded_preds_int.append(2)

    print('decoded labels: ', decoded_labels)
    print('decoded preds: ', decoded_preds)
            
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
    
# load model
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

# optimization / lora
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    #bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

lora_config = LoraConfig(
	r=8, 
	target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
	task_type="CAUSAL_LM",
)
#from accelerate import PartialState
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, quantization_config=bnb_config, device_map='auto', torch_dtype=torch.bfloat16)#.to(device)#, device_map={"": PartialState().process_index})
model.train()
for param in model.vision_tower.parameters():
    param.requires_grad = False
for param in model.multi_modal_projector.parameters():
    param.requires_grad = False
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# training
from transformers import TrainingArguments
args_training=TrainingArguments(
    num_train_epochs=args.nepochs,
    remove_unused_columns=False,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=args.iter_size,
    warmup_steps=2,
    learning_rate=2e-5,
    weight_decay=1e-6,
    adam_beta2=0.999,
    logging_steps=100,
    optim="adamw_8bit",
    save_strategy="steps",
    save_steps=args.save_steps,
    eval_strategy='steps',
    eval_steps=args.eval_steps,
    push_to_hub=False,
    save_total_limit=10,
    bf16 = True,
    dataloader_pin_memory=False,
    output_dir = checkpoints_dir,
    resume_from_checkpoint = checkpoints_dir + 'checkpoint-' + str(args.resume) if args.resume > 0 else None,
)

if args.resume > 0:
    print('resuming from checkpoint: ' + checkpoints_dir + 'checkpoint-' + str(args.resume))
    if not os.path.exists(checkpoints_dir + 'checkpoint-' + str(args.resume)):
        print('resume from checkpoint failed, path does not exist')
        exit(1)

from transformers import Trainer
trainer = Trainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics = preprocess_logits_for_metrics,
    args=args_training
)
trainer.train(resume_from_checkpoint=(args.resume > 0))

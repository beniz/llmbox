import sys
import os

os.environ['WANDB_DISABLED'] = 'true'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
arg_parser = argparse.ArgumentParser(description='ft multimodal spamfilter')
arg_parser.add_argument('--data-dir', type=str, required=True)
arg_parser.add_argument('--model-id', type=str, default="Qwen/Qwen2-VL-2B-Instruct")
arg_parser.add_argument('--batch-size', type=int, default=2)
arg_parser.add_argument('--iter-size', type=int, default=64, help='gradient accumulation')
arg_parser.add_argument('--eval-steps', type=int, default=10, help='number of steps between evaluations')
arg_parser.add_argument('--save-steps', type=int, default=10, help='number of steps the model is saved')
arg_parser.add_argument('--nepochs', type=int, default=1, help='number of training epochs')
arg_parser.add_argument('--output-dir', required=True, type=str, help='output dir')
arg_parser.add_argument('--resume', type=int, default=-1, help='whether to resume training, must indicate checkpoint iteration')
arg_parser.add_argument('--min-img-size', type=int, default=224)
arg_parser.add_argument('--use-crops', action='store_true', help='whether to use crops (otherwise image is resized)')
arg_parser.add_argument('--no-resize', action='store_true', help='whether to resize images')
args = arg_parser.parse_args()

checkpoints_dir = args.output_dir + '/checkpoints'
try:
    os.mkdir(args.output_dir)
    os.mkdir(checkpoints_dir)
except:
    print('model dirs already exist')
    pass

import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

model_id = args.model_id
processor = AutoProcessor.from_pretrained(model_id)

import torch
device = torch.cuda.current_device()

#image_token = processor.tokenizer.convert_tokens_to_ids("<image>")

from datasets import load_dataset, Image
ds = load_dataset('imagefolder', data_dir=args.data_dir)
#ds.train_test_split(test_size=0.1)
train_ds = ds["train"]
val_ds = ds["test"]

custom_prompt = """Below is an email message as input. Decide whether it is spam or ham. In response, if spam write as spam otherwise write ham."""

import torchvision.transforms as transforms
transform_rs = transforms.Resize(args.min_img_size)

import torchvision.transforms.v2 as transformsv2
transform_v2_rc = transformsv2.RandomCrop((args.min_img_size,args.min_img_size),
                                          pad_if_needed=True, padding_mode='constant')

if args.use_crops:
    tr = transform_v2_rc
elif args.no_resize:
    tr = None # original size
else:
    tr = transform_rs # resize is default

def find_assistant_content_sublist_indexes(l):
    # (Pdb++) processor.tokenizer.encode("<|im_start|>assistant")
    # [151644, 77091]
    # (Pdb++) processor.tokenizer.encode("<|im_end|>")
    # [151645]

    start_indexes = []
    end_indexes = []

    # Iterate through the list to find starting points
    for i in range(len(l) - 1):
        # Check if the current and next element form the start sequence
        if l[i] == 151644 and l[i + 1] == 77091:
            start_indexes.append(i)
            # Now look for the first 151645 after the start
            for j in range(i + 2, len(l)):
                if l[j] == 151645:
                    end_indexes.append(j)
                    break  # Move to the next start after finding the end

    return list(zip(start_indexes, end_indexes))
    
from PIL import Image
def collate_fn(examples):

    images = [example["image"] for example in examples]
    labels = [example["label"] for example in examples]
    labels = ['spam' if label == 1 else 'ham' for label in labels]
    
    images = [image.convert("RGB") for image in images]

    if tr is not None:
        images = [tr(image) for image in images]

    b_image_inputs = []
    b_texts = []
    messages = []

    message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "",
                },
                {"type": "text", "text": custom_prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "",
                }
            ]
        }
    ]

    i = 0
    for image in images:
        image.save("/tmp/tmp.jpg") # ugly
        img_str = "file:///tmp/tmp.jpg"

        msg = message.copy()
        msg[0]["content"][0]['image'] = img_str
        msg[1]["content"][0]['text'] = labels[i]
        messages.append(msg)
        
        texts = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(msg)

        b_image_inputs.append(image_inputs)
        b_texts.append(texts)
        i += 1

    inputs = processor(text=b_texts, images=b_image_inputs, 
                       return_tensors="pt", padding=True)

    inputs = inputs.to(device)

    input_ids = inputs['input_ids']
    input_ids_lists = inputs['input_ids'].tolist()
    assert len(messages) == len(input_ids_lists)

    labels_list = []
    for ids_list in input_ids_lists:
        label_ids = [-100] * len(ids_list)
        for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
            label_ids[begin_end_indexs[0]+2:begin_end_indexs[1]+1] = ids_list[begin_end_indexs[0]+2:begin_end_indexs[1]+1]
        labels_list.append(label_ids)

    labels_ids = torch.tensor(labels_list, dtype=torch.int64)
    
    #return inputs, labels_ids
    return dict(
        input_ids=input_ids,
        attention_mask=input_ids.ne(processor.tokenizer.pad_token_id),
        labels=labels_ids
    )

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def preprocess_logits_for_metrics(logits, labels):
    # argmax to get the token ids
    lgm = logits.argmax(dim=-1)
    return lgm

def compute_metrics(pred):
    labels = pred.label_ids[:, 1:]
    # -100 is a default value for ignore_index used by DataCollatorForCompletionOnlyLM
    mask = labels == -100
    # replace -100 with a value that the tokenizer can decode
    labels[mask] = processor.tokenizer.pad_token_id
    decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_labels = [l.replace('\n', '') for l in decoded_labels]
    
    preds = pred.predictions[:, :-1]    
    preds[mask] = processor.tokenizer.pad_token_id
    decoded_preds = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_preds = [p.replace('\n', '') for p in decoded_preds]
    
    #print('decoded_labels=', decoded_labels)
    #print('decoded_preds=', decoded_preds)
    
    decoded_labels_int = []
    for dl in decoded_labels:
        if dl == 'ham':
            decoded_labels_int.append(0)
        elif dl == 'spam':
            decoded_labels_int.append(1)
        else:
            decoded_labels_int.append(2) # never occurs

    decoded_preds_int = []
    for dp in decoded_preds:
        if dp == 'ham':
            decoded_preds_int.append(0)
        elif dp == 'spam':
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
    
# optimization / lora
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

lora_config = LoraConfig(
	r=8, 
	target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
	task_type="CAUSAL_LM",
)
from accelerate import PartialState
model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, quantization_config=bnb_config, device_map={"": PartialState().process_index})
for param in model.visual.parameters():
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
    logging_steps=args.eval_steps,
    optim="adamw_8bit",
    #optim="adamw_hf",
    save_strategy="steps",
    save_steps=args.save_steps,
    eval_strategy='steps',
    eval_steps=args.eval_steps,
    push_to_hub=False,
    save_total_limit=10,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    #report_to=["tensorboard"],
    dataloader_pin_memory=False,
    output_dir = checkpoints_dir,
    resume_from_checkpoint = checkpoints_dir + '/checkpoints/checkpoint-' + str(args.resume) if args.resume > 0 else None,
)

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

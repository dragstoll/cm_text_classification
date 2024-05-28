#!/usr/bin/env python
# coding: utf-8


#!conda activate huggingface_env
#conda info
#ls
# import packages
import sys
import pandas as pd
import numpy as np
import sklearn
from matplotlib import pyplot as plt
import re
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig 
from transformers import GPT2Tokenizer, TrainingArguments, Trainer, GPT2LMHeadModel, AutoTokenizer, GPTJForCausalLM, GPTJConfig, GPTNeoXForCausalLM, GPTNeoXTokenizerFast, AutoModelForCausalLM, AutoConfig
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from tqdm import trange
import random
from ml_things import plot_dict, plot_confusion_matrix, fix_text
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, PeftModel
import os
os.environ['WANDB_DISABLED'] = 'true'
pd.options.display.max_colwidth = 1000

from pathlib import Path    
import os

import torch
import torch.nn as nn
import torch.nn.functional as F




cwd = os.getcwd()
from huggingface_hub import login
pd.options.display.max_colwidth = 1000
access_token = os.getenv("HUGFACE_TOKEN")
login(token=access_token)

train_path = os.path.expanduser('/cfs/earth/scratch/stdg/python/include/train_data_checked_20231022.csv')
test_path = os.path.expanduser('/cfs/earth/scratch/stdg/python/include/test_data_20231017.xlsx')

#read test_data_en.csv to a dataframe test_data
# test_data = pd.read_csv(test_path, sep=',', encoding='utf-8')
#reat the test_data.xlsx to a dataframe test_data
test_data = pd.read_excel(test_path,sheet_name='Sheet1')
print(test_data.columns)
# sys.exit()
#drop columns "Unnamed: 0", "sentence"
test_data=test_data.drop(['Unnamed: 0',], axis=1)
#rename column "sentence_en" to "sentence"
# test_data=test_data.rename(columns={'sentence_en': 'sentence'})
#read train_data_en.csv to a dataframe train_data
train_data = pd.read_csv(train_path, sep=',', encoding='utf-8')
#change the label column to int
train_data['label'] = train_data['label'].astype(int)
#drop columns "Unnamed: 0", "sentence"
# train_data=train_data.drop(['Unnamed: 0', 'sentence'], axis=1)
#rename column "sentence_en" to "sentence"
# train_data=train_data.rename(columns={'sentence_en': 'sentence'})
#print column names of train_data
print(train_data.columns)
#print count values of label column in train_data
print(pd.crosstab(index=train_data['label'], columns='count'))
#print column names of test_data
print(test_data.columns)
#print count values of label column in test_data
print(pd.crosstab(index=test_data['label'], columns='count'))
#print first 10 rows of test_data sentence column
# print(test_data['sentence'].head(10))
#print first 10 rows of train_data sentence column
# print(train_data['sentence'].head(10))



#drop na values
train_data=train_data.dropna()
test_data=test_data.dropna()
train=train_data

# sys.exit()



# from collections import Counter
# from imblearn.over_sampling import RandomOverSampler
# from imblearn.under_sampling import RandomUnderSampler 

# rus = RandomUnderSampler(random_state=42)
# x = pd.array(train['sentence']).reshape(-1,1)
# y = (train['label'])
# variablex_rus, y_rus = rus.fit_resample(x,y)
# print('Original dataset shape', Counter(y))
# print('Undersample dataset shape', Counter(y_rus))
# variablex_rus=pd.DataFrame(variablex_rus)
# y_rus = pd.DataFrame(y_rus)
# train_under=pd.concat([variablex_rus, y_rus], axis=1)
# train_under=train_under.rename(columns={0 :'sentence'})
# # train_under.head() 

# # sys.exit()  
# # define dataset
# # X, y = make_classification(n_samples=10000, weights=[0.99], flip_y=0)
# sampling_strategy1 = {0: 678, 1: 912, 2: 772, 3: 224, 4: 1233}
# # {1: 855, 0: 721, 2: 690, 3: 250}

# x = pd.array(train['sentence']).reshape(-1,1)
# y = (train['label'])
# # summarize class distribution
# print(Counter(y))
# # define oversampling strategy
# over = RandomOverSampler(sampling_strategy=sampling_strategy1)
# # fit and apply the transform
# x_over, y_over = over.fit_resample(x, y)
# # summarize class distribution
# print(Counter(y_over))
# # define undersampling strategy
# sampling_strategy2 = {0: 678, 1: 678, 2: 678, 3: 224, 4: 678}
# under = RandomUnderSampler(sampling_strategy=sampling_strategy2)
# # fit and apply the transform
# x_under, y_under = under.fit_resample(x_over, y_over)
# # summarize class distribution
# print(Counter(y_under))
# x_under=pd.DataFrame(x_under)
# y_under = pd.DataFrame(y_under)
# train_overunder=pd.concat([x_under, y_under], axis=1)
# train_overunder=train_overunder.rename(columns={0 :'sentence'})
# # train_overunder.head() 



# from sklearn.utils import shuffle

# train_overunder = train_overunder.sample(frac=1).reset_index(drop=True)


from sklearn.utils import shuffle

gewalt_string = [ "Klient hat Gewalt erfahren", "Klient hat häusliche Gewalt erlebt", "Klient hat Gewalt ausgeübt",  "Klient hat sexuelle Gewalt erlebt", "Klient hat keine Gewalt erlebt"]

# # Model Implementierung
from transformers import AutoModelForSequenceClassification
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoConfig
import aiohttp
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu") if torch.cuda.is_available() else torch.device("cpu")
CUDA_LAUNCH_BLOCKING=1
print(device)

from datasets import load_dataset

import aiohttp

# model_name = "tiiuae/falcon-180B"
model_name = "mistralai/Mixtral-8x7B-v0.1"

#print model name
print("Model name:", model_name)



## Load model and data
#--------
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os


# model_name = "EleutherAI/gpt-neox-20b"
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    # load_in_8bit=True,
    # llm_int8_enable_fp32_cpu_offload=True,
)
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory


# model_name= "ZurichNLP/swissbert"
# model_name = "GroNLP/mdebertav3-subjectivity-german"
# Get model configuration.
id2label={
    
    "0": "Klient hat Gewalt erfahren",
    "1": "Klient hat häusliche Gewalt erlebt",
    "2": "Klient hat Gewalt ausgeübt",
    "3": "Klient hat sexuelle Gewalt erlebt",
    "4": "Klient hat keine Gewalt erlebt"
  }

label2id={
    
    "Klient hat Gewalt erfahren": 0,
    "Klient hat häusliche Gewalt erlebt": 1,
    "Klient hat Gewalt ausgeübt": 2,
    "Klient hat sexuelle Gewalt erlebt": 3,
    "Klient hat keine Gewalt erlebt": 4
  }, 


# model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name, 
#                                           #  label2id={}, id2label={}, num_labels=0,
# num_labels=5, 
# id2label=id2label, 
# label2id=label2id, 

# # hidden_dropout_prob=0.1, 
# # attention_probs_dropout_prob=0,
# # summary_last_dropout=0.1,
# # cls_drop_out=0.1,
# output_attentions=False, 
#  return_dict=False, 
#  load= False,
# )

#print model configuration
# print(model_config)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
   quantization_config=quant_config, 
                                            #  device_map={"":0},
                                            device_map='auto',
                                            num_labels=5, 
                                            id2label=id2label, label2id=label2id,
                                            trust_remote_code=True,
                                            return_dict=True,
                                            num_experts_per_tok = 2,
                                                                )
# .half().to(device)

# model.set_default_language("de_CH")

#print model 
# print(model)

# model.gradient_checkpointing_enable()
model.config.use_cache = False
# model.resize_token_embeddings(len(tokenizer))
# from transformers import AutoModelForSequenceClassification
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftConfig, PeftModel, PeftModelForSequenceClassification, TaskType

#show current directory
print(os.getcwd())
# peft_model_id = "dragstoll/falcon180b_seq_20231130_2327"
peft_model_id = "./results_mixtral/checkpoint-1300"
# peft_model_id = "dragstoll/mixtral_seq"
config = PeftConfig.from_pretrained(peft_model_id)

print("Loading model from Huggingface Hub")
print("Model ID:", peft_model_id)

# Get the name of the last module (to un-freeze the last layer).
# named_modules = []
# for module in model.named_modules():
#     module_name, module_type = module
#     named_modules.append({str(module_name): str(module_type)})
# last_module_name = list(named_modules[-1].keys())[0]

# config = LoraConfig(
#     r=8,
#     lora_alpha=16,
#      target_modules=
#     #  modules,
#      [
        
#         "query_key_value",
#         "dense",                
        
#         "dense_h_to_4h",
        
#         "dense_4h_to_h",        
        
#     ],
#     lora_dropout=0.05,
#     bias="none",
#     task_type=TaskType.SEQ_CLS,
#     modules_to_save=[last_module_name],
#     inference_mode=True
# )

# model = get_peft_model(model, config)
# model.load_state_dict(torch.load("falcon180b_seq.pt"))
# model = PeftModel.from_pretrained(model, peft_model_id)
model = PeftModelForSequenceClassification.from_pretrained(model, peft_model_id)


# model = model.merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                          do_lower_case = True,
                                         use_fast=True,
                                         eos_token='###', pad_token='[PAD]',
                                         )

    



tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# tokenizer.add_special_tokens({'eos_token': '###'})
tokenizer.pad_token = tokenizer.eos_token

from functools import wraps

org_call_one = tokenizer._call_one

@wraps(org_call_one)
def _call_one_wrapped(*x, **y):
    y['return_token_type_ids'] = False
    return org_call_one(*x, **y)

tokenizer._call_one = _call_one_wrapped


# model = prepare_model_for_kbit_training(model)

# import bitsandbytes as bnb
# # COPIED FROM https://github.com/artidoro/qlora/blob/main/qlora.py
# def find_all_linear_names(model):
#     lora_module_names = set()
#     for name, module in model.named_modules():
#         if isinstance(module, bnb.nn.Linear4bit):
#             names = name.split(".")
#             lora_module_names.add(names[0] if len(names) == 1 else names[-1])

#     if "lm_head" in lora_module_names:  # needed for 16-bit
#         lora_module_names.remove("lm_head")
#     return list(lora_module_names)

# # get lora target modules
# modules = find_all_linear_names(model)
# print(f"Found {len(modules)} modules to quantize: {modules}")

# config = LoraConfig(
#     r=8,
#     lora_alpha=32,
#      target_modules=
#     #  modules,
#      [
        
#         "query_key_value",
#         "dense",                
        
#         "dense_h_to_4h",
        
#         "dense_4h_to_h",        
        
#     ],
#     lora_dropout=0.05,
#     bias="none",
#     task_type="SEQ_CLS,",
# )

# model = get_peft_model(model, config)

# # sys.exit()
# #freeze everything except classifier and last two transformer layers
# for name, param in model.named_parameters():
#     if 'classifier' not in name and '36' and '35' not in name:
#         param.requires_grad = False
#     else:
#         param.requires_grad = True

# from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType

# model = prepare_model_for_kbit_training(model)

# import bitsandbytes as bnb
# # COPIED FROM https://github.com/artidoro/qlora/blob/main/qlora.py
# def find_all_linear_names(model):
#     lora_module_names = set()
#     for name, module in model.named_modules():
#         if isinstance(module, bnb.nn.Linear4bit):
#             names = name.split(".")
#             lora_module_names.add(names[0] if len(names) == 1 else names[-1])

#     # if "lm_head" in lora_module_names:  # needed for 16-bit
#     #     lora_module_names.remove("lm_head")
#     return list(lora_module_names)

# # get lora target modules
# modules = find_all_linear_names(model)
# print(f"Found {len(modules)} modules to quantize: {modules}")


# # Define LoRA Config
# lora_config = LoraConfig(
#  r=256,
#  lora_alpha=512,
#  target_modules=
# #   modules,
#  ["query", "key", "value", "dense", "out_proj"],
#  lora_dropout=0.05,
#  bias="none",
#  task_type=TaskType.SEQ_CLS, # this is necessary
#  inference_mode=True
# )

# add LoRA adaptor
# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters() # see % trainable parameters



from datasets import Dataset
# tokenizer = AutoTokenizer.from_pretrained(checkpoint, do_lower_case = True)




#mit overrepresentation
# train_dataset = Dataset.from_pandas(train_over)
#mit underrepresentation
# train_dataset = Dataset.from_pandas(train_under)
#mit overunderrepresentation
# train_dataset = Dataset.from_pandas(train_overunder)
# #ohne overrepr
# # train_dataset = Dataset.from_pandas(train)
# valid_dataset = Dataset.from_pandas(valid)

# test_dataset = Dataset.from_pandas(valid)

train_dataset = Dataset.from_pandas(train_data)
#ohne overrepr
# train_dataset = Dataset.from_pandas(train)
valid_dataset = Dataset.from_pandas(test_data)

test_dataset = Dataset.from_pandas(test_data)

test_dataset = test_dataset.remove_columns([ "__index_level_0__" ])

valid_dataset = valid_dataset.remove_columns([ "__index_level_0__"] )
# train_dataset = train_dataset.remove_columns([ "Unnamed: 0", "__index_level_0__"] )

from datasets import DatasetDict
datasets = DatasetDict({
    # "train": train_dataset,
    "valid": valid_dataset,
    # "test": test_dataset
    })
print(datasets)
                                          
# sys.exit()

def tokenize_function(example):
    return tokenizer(example["sentence"] ,  truncation=True, max_length=450, padding=True, return_tensors="pt")

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

tokenized_datasets = datasets.map(tokenize_function, batched=True)

# resize model embedding to match new tokenizer
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = model.config.eos_token_id

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")



# from datasets import load_metric

# metric = load_metric("accuracy")
import evaluate
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
# #     print(logits)
# #     print(logits[0])
# #     print(labels)
# #     print(logits[1])
#     predictions = np.argmax(logits[0], axis = -1)
#     # print(predictions)
# #     print(labels)
#     return metric.compute(predictions=predictions, references=labels)


print('Loading configuration...')


# from transformers import TrainingArguments

# training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")

# training_args = TrainingArguments(output_dir='results_flcnseq', 
#                                 #   num_train_epochs=12, 
#                                   logging_steps=400, 
#                                   save_steps=400,
#                                 load_best_model_at_end=False, 
#                                 #  save_total_limit=8,
#                                 #  save_strategy="epoch", evaluation_strategy="epoch",
#                                 save_strategy="steps", evaluation_strategy="steps",
# #                                      save_steps=10, 
#                                     eval_steps=400,
#                                     max_steps=2000,
#                                  eval_accumulation_steps=1,
#                                  per_device_train_batch_size=8, 
#                                  per_device_eval_batch_size=8, 
#                                  gradient_accumulation_steps=1,
#                                  gradient_checkpointing=True, 
# #                                  optim="adafactor",
#                                 #   optim="adamw_hf",
#                                 optim="paged_adamw_32bit",
#                                 #  warmup_steps=100,
#                                   warmup_ratio=0.1,
#                                   weight_decay=0.001, 
#                                   logging_dir='logs', 
#                                 #   learning_rate=1e-04,
#                                 # lr_scheduler_type="linear",
#                                  fp16 = True,
#                                  remove_unused_columns=False,
#                                 #  deepspeed="deepspeed_config.json",
#                                  )
  
# # start training
# trainer =Trainer(model=model, args=training_args, 
#                  train_dataset=tokenized_datasets['train'],                  
#                  eval_dataset=tokenized_datasets['valid'],  
#                 #  compute_metrics=compute_metrics,
#                  tokenizer=tokenizer,
#                  data_collator=data_collator,
#                  )


        
# trainer.train()

# train


from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def plot_confusion_matrix(y_preds, y_true, labels, title2="Normalized confusion matrix", path='plot_confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_preds , normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=True, xticks_rotation='vertical') 
    # plt.title("Normalized confusion matrix")
    plt.title(title2)
    plt.grid(False)
    plt.show()
    # plt.savefig(path)


eval_dataloader = DataLoader(
    tokenized_datasets["valid"], batch_size=4, collate_fn=data_collator
)


metric = evaluate.load("accuracy")


# #mode mocel to gpu
# model.to(device)
model.config.use_cache = True  # silence the warnings. Please re-enable for inference!
model.eval()
predictions_sum=[]





for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        # print( model(**batch))
        logits = model(**batch).logits
    # print(logits)
    
    predictions = torch.argmax(logits, dim=-1)
    predictions_sum+=predictions
    metric.add_batch(predictions=predictions, references=batch["labels"])

#print accuracy score
print(metric.compute())
predictions_sum = torch.Tensor(predictions_sum)
predictions_sum.cpu()
# len(predictions_sum)
plot_confusion_matrix(predictions_sum, tokenized_datasets["valid"]["labels"], gewalt_string, title2="Normalized confusion matrix\n Validation Dataset", path='validation_plot_confusion_matrix.png')

y_pred = predictions_sum
y_valid = pd.Series(tokenized_datasets["valid"]["labels"])

from sklearn.metrics import classification_report
my_tags = gewalt_string
print(classification_report(y_valid, y_pred,target_names=my_tags))
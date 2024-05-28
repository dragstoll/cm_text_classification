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


train_path = os.path.expanduser('/.')
test_path = os.path.expanduser('/.')

#read test_data_en.csv to a dataframe test_data
# test_data = pd.read_csv(test_path, sep=',', encoding='utf-8')
#read the test_data.xlsx to a dataframe test_data
# test_data = pd.read_excel(test_path,sheet_name='Sheet1')
# print(test_data.columns)
# sys.exit()
#drop columns "Unnamed: 0", "sentence"
# test_data=test_data.drop(['Unnamed: 0',], axis=1)
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
# print(test_data.columns)
#print count values of label column in test_data
# print(pd.crosstab(index=test_data['label'], columns='count'))
#print first 10 rows of test_data sentence column
# print(test_data['sentence'].head(10))
#print first 10 rows of train_data sentence column
# print(train_data['sentence'].head(10))



#drop na values
train_data=train_data.dropna()
# test_data=test_data.dropna()

# split train data into train and valid data
# 0 labels
label_0=train_data['label']==0
data3_0=train_data[label_0]
train_0, test_0 = train_test_split(data3_0, test_size=50, random_state=42)
# 1 labels
label_1=train_data['label']==1
data3_1=train_data[label_1]
train_1, test_1 = train_test_split(data3_1, test_size=50, random_state=42)
# 2 labels
label_2=train_data['label']==2
data3_2=train_data[label_2]
train_2, test_2 = train_test_split(data3_2, test_size=50, random_state=42)
# 3 labels
label_3=train_data['label']==3
data3_3=train_data[label_3]
train_3, test_3 = train_test_split(data3_3, test_size=50, random_state=42)
# 4 labels
label_4=train_data['label']==4
data3_4=train_data[label_4]
train_4, test_4 = train_test_split(data3_4, test_size=50, random_state=42)


train_data = pd.concat([train_0, train_1, train_2, train_3, train_4], axis=0)

valid_data = pd.concat([test_0, test_1, test_2, test_3, test_4], axis=0)




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


model_name = "mistralai/Mixtral-8x7B-v0.1"
#print model name
print("Model name:", model_name)



tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                          do_lower_case = True,
                                         use_fast=True,
                                         eos_token='###', pad_token='[PAD]',
                                         )

    



tokenizer.add_special_tokens({'pad_token': '[PAD]'})

tokenizer.pad_token = tokenizer.eos_token

from functools import wraps

org_call_one = tokenizer._call_one

@wraps(org_call_one)
def _call_one_wrapped(*x, **y):
    y['return_token_type_ids'] = False
    return org_call_one(*x, **y)

tokenizer._call_one = _call_one_wrapped

## Load model and data
#--------
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os



quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,

)
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory



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




model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
   quantization_config=quant_config, 
                                            #  device_map={"":0},
                                            device_map='auto',
                                            num_labels=5, 
                                            id2label=id2label, label2id=label2id,
                                            trust_remote_code=True,
                                            return_dict=True,
                                           num_experts_per_tok = 8,
                                                                )






#print model 
print(model)

# model.gradient_checkpointing_enable()
model.config.use_cache = False
model.resize_token_embeddings(len(tokenizer))

from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType

model = prepare_model_for_kbit_training(model)

import bitsandbytes as bnb

def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)

# get lora target modules
modules = find_all_linear_names(model)
print(f"Found {len(modules)} modules to quantize: {modules}")

# Get the name of the last module (to un-freeze the last layer).
named_modules = []
for module in model.named_modules():
    module_name, module_type = module
    named_modules.append({str(module_name): str(module_type)})
last_module_name = list(named_modules[-1].keys())[0]

config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    modules_to_save=[  'score' ],
    r=32,
    lora_alpha=64,
     target_modules=
    #  modules,
    ['v_proj', 'q_proj', 'k_proj',  'o_proj','gate', 'w2', 'w3', 'w1','score'],

    lora_dropout=0.05,
    bias="none",
    

    inference_mode=False,
)

model = get_peft_model(model, config)





from datasets import Dataset


train_dataset = Dataset.from_pandas(train_data)

valid_dataset = Dataset.from_pandas(valid_data)

# test_dataset = Dataset.from_pandas(test_data)

# test_dataset = test_dataset.remove_columns([ "__index_level_0__" ])

valid_dataset = valid_dataset.remove_columns([ "__index_level_0__"] )
# train_dataset = train_dataset.remove_columns([ "Unnamed: 0", "__index_level_0__"] )

from datasets import DatasetDict
datasets = DatasetDict({
    "train": train_dataset,
    "valid": valid_dataset,

    })
print(datasets)
                                          
# sys.exit()

def tokenize_function(example):
    return tokenizer(example["sentence"] ,  truncation=True, max_length=512, padding=True, return_tensors="pt")

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

tokenized_datasets = datasets.map(tokenize_function, batched=True)

# resize model embedding to match new tokenizer
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = model.config.eos_token_id

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")

# from datasets import load_metric

import evaluate
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)    
    # print(labels)    
    # print(predictions)
    return metric.compute(predictions=predictions, references=labels)



print('Loading configuration...')




training_args = TrainingArguments(output_dir='./results_mixtral', 

                                  logging_steps=100, 
                                  save_steps=100,
                                load_best_model_at_end=True, 

                                save_strategy="steps", evaluation_strategy="steps",

                                    eval_steps=100,
                                    max_steps=2200,
                                 eval_accumulation_steps=1,
                                 per_device_train_batch_size=16, 
                                 per_device_eval_batch_size=16, 
                                 gradient_accumulation_steps=1,
                                 gradient_checkpointing=True, 
                                optim="paged_adamw_32bit",

                                  warmup_ratio=0.1,
                                  weight_decay=0.001, 
                                  logging_dir='logs', 
                                 fp16 = True,
                                 remove_unused_columns=False,

                                 )
   
# start training
trainer =Trainer(model=model, args=training_args, 
                 train_dataset=tokenized_datasets['train'],                  
                 eval_dataset=tokenized_datasets['valid'],  
                 compute_metrics=compute_metrics,
                 tokenizer=tokenizer,
                 data_collator=data_collator,
                 )


trainer.train()
# trainer.train("./results_mixtral/checkpoint-1500")




# print("Model pushed to hub")



model_name_save="mixtral_seq"
import datetime
now = datetime.datetime.now()
date_time_stamp = now.strftime("%Y%m%d_%H%M")
model_date_time_stamp = model_name_save+"_"+date_time_stamp



model.save_pretrained(model_name_save)
# save tokenizer
tokenizer.save_pretrained(model_name_save)

model.push_to_hub(model_name_save, use_auth_token=True, private=True)
# push tokenizer to hub
tokenizer.push_to_hub(model_name_save, use_auth_token=True, private=True)

model_date_time_stamp2="dragstoll/"+model_name_save




predictions, labels, metrics = trainer.predict(tokenized_datasets['valid'])


y_pred = np.argmax(predictions, axis = -1)




from sklearn.metrics import accuracy_score, classification_report
print(accuracy_score(tokenized_datasets['valid']['labels'], y_pred))



# Create the evaluation report.
gewalt_string = [ "Klient hat Gewalt erfahren", "Klient hat häusliche Gewalt erlebt", "Klient hat Gewalt ausgeübt",  "Klient hat sexuelle Gewalt erlebt", "Klient hat keine Gewalt erlebt"]
evaluation_report = classification_report(tokenized_datasets['valid']['labels'], y_pred,  target_names=gewalt_string)
# evaluation_report = classification_report(tokenized_datasets['test']['label'], np.argmax(predictions[0][0], axis=-1), labels=tokenized_datasets['test']['label'], target_names=tokenized_datasets['test']['label'])
# Show the evaluation report.
print(evaluation_report)



# eval
from huggingface_hub import Repository
with open('evaluation_report.txt', 'w') as f:
    f.write(evaluation_report)

from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="./evaluation_report.txt",
    path_in_repo="evaluation_report.txt",
    repo_id=model_date_time_stamp2,
    repo_type="model",
    
)

from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import re
from dotenv import load_dotenv  #  pip3 install python-dotenv
from configparser import ConfigParser, ExtendedInterpolation
'''
https://www.kaggle.com/code/changyeop/how-to-fine-tune-gpt-2-for-beginners/notebook
'''
def cleaning(s):
    s = str(s)
    s = re.sub(r'\s\W',' ',s)
    s = re.sub(r'\W,\s',' ',s)
    s = re.sub(r"\d+", "", s)
    s = re.sub(r'\s+',' ',s)
    s = re.sub('[!@#$_]', '', s)
    s = s.replace("co","")
    s = s.replace("https","")
    s = s.replace("[\\w*"," ")
    return s


load_dotenv()
config_file = os.environ['CONFIG_FILE']
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read(f"../config/{config_file}")

def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model

def load_tokenizer(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

_tokenizer = None
def tokenize_sample_data(data):
    global _tokenizer
    if _tokenizer is None:                       # load the tokenizer once, not for every batch
        _tokenizer = load_tokenizer(config['casualLM']['model_name'])
    tokenizer = _tokenizer
    # input_feature = tokenizer(data["review_body"])
    input_feature = tokenizer(data[config['casualLM']['colunm_name']], truncation=True, max_length=int(config['casualLM']['max_length']), padding=True)
    return {
        "input_ids": input_feature["input_ids"],
        "attention_mask": input_feature["attention_mask"],
    }

def load_data(data, batch_size):
    tokenized_ds = data.map(
        tokenize_sample_data,
        batched=True,
        batch_size=batch_size)
    return tokenized_ds

def load_data_collator(tokenizer, mlm=False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm,
    )
    return data_collator




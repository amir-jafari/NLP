from transformers import Seq2SeqTrainer, TrainingArguments
from transformers import TextDataset, DataCollatorForSeq2Seq
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Text2TextGenerationPipeline
from datasets import load_dataset
import torch
import os
from dotenv import load_dotenv  #  pip3 install python-dotenv
from configparser import ConfigParser, ExtendedInterpolation

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Read config.ini file
load_dotenv()
config_file = os.environ['CONFIG_FILE']
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read(f"../config/{config_file}")

def load_model(model_path):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    return model

def load_tokenizer(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def tokenize_sample_data(data):
    tokenizer = load_tokenizer(config['seq2seq']['model_name'])
    input_feature = tokenizer(data[config['seq2seq']['input_feature']], truncation=True, max_length=int(config['seq2seq']['max_length']), padding='max_length')
    label = tokenizer(data[config['seq2seq']['label']], truncation=True, max_length=int(config['seq2seq']['max_length']), padding='max_length')

    return {
        "input_ids": input_feature["input_ids"],
        "attention_mask": input_feature["attention_mask"],
        "labels": label["input_ids"],
    }

def load_data(data, batch_size):
    tokenized_ds = data.map(
        tokenize_sample_data,
        batched=True,
        batch_size=batch_size)
    return tokenized_ds


def load_data_collator(tokenizer, model):
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )
    return data_collator



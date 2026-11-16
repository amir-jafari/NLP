from transformers import DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import Dataset
import pandas as pd
import re


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

def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token     # GPT-2 has no pad token; the data collator needs one
    return tokenizer


def load_dataset(file_path, tokenizer, block_size=128, column="review_body"):
    """Language-modeling dataset: read the text (.txt file, or one column of a .csv),
    tokenize it, and cut the long token stream into blocks of `block_size` tokens."""
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path, encoding="ISO-8859-1").dropna(subset=[column])
        text = " ".join(df[column].astype(str))
    else:
        text = open(file_path, encoding="utf-8").read()
    ids = tokenizer(text)["input_ids"]
    blocks = [ids[i:i + block_size] for i in range(0, len(ids) - block_size + 1, block_size)]
    return Dataset.from_dict({"input_ids": blocks})


def load_data_collator(tokenizer, mlm=False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm,
    )
    return data_collator


from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model
from transformers import Trainer, TrainingArguments
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer

import pandas as pd
import numpy as np
import re
import os
import torch
from dotenv import load_dotenv  #  pip3 install python-dotenv
from configparser import ConfigParser, ExtendedInterpolation
'''
https://www.kaggle.com/code/changyeop/how-to-fine-tune-gpt-2-for-beginners/notebook
'''
def cleaning(s):
    s = str(s)
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W,\s',' ',s)
    s = re.sub("\d+", "", s)
    s = re.sub('\s+',' ',s)
    s = re.sub('[!@#$_]', '', s)
    s = s.replace("co","")
    s = s.replace("https","")
    s = s.replace("[\w*"," ")
    return s

def load_dataset(file_path, tokenizer, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )
    return dataset


def load_data_collator(tokenizer, mlm=False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm,
    )
    return data_collator

def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer


def generate_text(model_path, sequence, max_length):
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=100,
        pad_token_id=model.config.eos_token_id,
        top_k=0,
        top_p=0.9,
    )
    # print(tokenizer.decode(final_outputs, skip_special_tokens=True))
    return tokenizer.decode(final_outputs[0], skip_special_tokens=True)


if __name__=='__main__':
    load_dotenv()
    config_file = os.environ['CONFIG_FILE']
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(f"../config/{config_file}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_file_path = config['LMHeadModel']['file_path']

    model_name = config['LMHeadModel']['model_name']
    model_path = config['LMHeadModel']['model_path']
    overwrite_output_dir = True
    per_device_train_batch_size = int(config['LMHeadModel']['batch_size'])
    num_train_epochs = int(config['LMHeadModel']['epochs'])
    save_steps = int(config['LMHeadModel']['save_steps'])
    allfiles = True

    if allfiles==True:
        df = pd.read_csv(train_file_path, encoding="ISO-8859-1")
        seq_index = [int(x) for x in np.linspace(0, 199, 10).tolist()]
        sentences = df[config['LMHeadModel']['colunm_name']][seq_index]
        gpt2_output = []
        finetune_output = []
        for sentence in sentences:
            len_sen = len(sentence.split(" "))
            sequence = " ".join(sentence.split(" ")[:6])
            text_output1 = generate_text(model_name, sequence, len_sen)  # change it to model_path if using fine-tuning model
            text_output2 = generate_text(model_path, sequence, len_sen)
            gpt2_output.append(text_output1)
            finetune_output.append(text_output2)

        df_out = pd.DataFrame(
            dict(
                orig=sentences,
                gpt2=gpt2_output,
                finetune = finetune_output
            )
        )
        df_out.to_csv(config['LMHeadModel']['out_file'])
        print(df_out)
    # # inference
    else:
        sequence = input("Please input the start of sentence:")  # this product is
        max_len = int(input("Please input the maximum length:"))  # 20
        generate_text(model_path, sequence, max_len)
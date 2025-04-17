from utils import *
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
torch.cuda.memory_summary(device=None, abbreviated=False)
torch.cuda.empty_cache()

import os
from dotenv import load_dotenv  #  pip3 install python-dotenv
from configparser import ConfigParser, ExtendedInterpolation

'''
https://www.kaggle.com/code/changyeop/how-to-fine-tune-gpt-2-for-beginners/notebook
'''
# Read config.ini file
load_dotenv()
config_file = os.environ['CONFIG_FILE']
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read(f"../config/{config_file}")

print(config['instance']['user'])

if __name__=='__main__':


    PATH = os.getcwd()
    # you need to set parameters

    train_file_path = config['casualLM']['file_path']
    # train_file_path = '../../../../../../Data/dolly-15k_train.csv'


    # model_name = "sangjeedondrub/tibetan-roberta-causal-base"
    model_name = config['casualLM']['model_name']
    model_path = config['casualLM']['model_path']
    overwrite_output_dir = True
    per_device_train_batch_size = 4
    num_train_epochs = int(config['casualLM']['epochs'])
    save_steps = int(config['casualLM']['save_steps'])
    training = True
    tokenizer = load_tokenizer(model_name)
    # train_dataset = load_dataset(train_file_path, tokenizer)

    print("################# load dataset ###################")
    data = load_dataset("csv", data_files=train_file_path)
    train_dataset = load_data(data, per_device_train_batch_size)
    data_collator = load_data_collator(tokenizer)
    tokenizer.save_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.save_pretrained(model_path)

    print("################# start train #################")
    training_args = TrainingArguments(
        output_dir=model_path,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        save_steps=save_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
        train_dataset=train_dataset['train'],
    )

    trainer.train()
    trainer.save_model()
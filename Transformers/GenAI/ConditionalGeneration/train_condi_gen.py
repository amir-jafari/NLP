from transformers import T5ForConditionalGeneration

from transformers import TrainingArguments, Trainer
from utils import load_model, load_data, load_data_collator, load_tokenizer
from transformers import AutoTokenizer
from datasets import load_dataset
import os
import torch
import os
from dotenv import load_dotenv  #  pip3 install python-dotenv
from configparser import ConfigParser, ExtendedInterpolation


if __name__=='__main__':
    # Read config.ini file
    load_dotenv()
    config_file = os.environ['CONFIG_FILE']
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(f"../config/{config_file}")

    print(config['instance']['user'])

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_name = config['condi_gen']['model_name']
    tokenizer = load_tokenizer(model_name)

    train_file_path = config['condi_gen']['file_path']

    data = load_dataset("csv", data_files=train_file_path)

    model_path = config['condi_gen']['model_path']
    per_device_train_batch_size = int(config['condi_gen']['batch_size'])
    num_train_epochs = int(config['condi_gen']['epochs'])
    save_steps = int(config['condi_gen']['save_steps'])
    overwrite_output_dir = True
    training = True

    checkpoint = model_name
    print("################# load dataset ###################")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    train_dataset = load_data(data, per_device_train_batch_size)
    data_collator = load_data_collator(tokenizer, checkpoint)

    tokenizer.save_pretrained(model_path)

    model = load_model(checkpoint)
    model.save_pretrained(model_path)

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
        train_dataset=train_dataset['train'],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    print("################# start train #################")
    trainer.train()
    trainer.save_model()
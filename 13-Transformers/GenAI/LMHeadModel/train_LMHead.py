from transformers import Trainer, TrainingArguments
from utils import load_dataset, load_model, load_data_collator, load_tokenizer
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
    print(config['LMHeadModel']['file_path'])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_name = config['LMHeadModel']['model_name']
    train_file_path = config['LMHeadModel']['file_path']
    model_path = config['LMHeadModel']['model_path']
    overwrite_output_dir = True
    per_device_train_batch_size = int(config['LMHeadModel']['batch_size'])
    num_train_epochs = int(config['LMHeadModel']['epochs'])
    save_steps = int(config['LMHeadModel']['save_steps'])

    tokenizer = load_tokenizer(model_name)
    train_dataset = load_dataset(train_file_path, tokenizer, column=config['LMHeadModel']['colunm_name'])
    data_collator = load_data_collator(tokenizer)

    tokenizer.save_pretrained(model_path)

    model = load_model(model_name)
    # model = GPT2Model.from_pretrained(model_name)

    model.save_pretrained(model_path)

    training_args = TrainingArguments(
        output_dir=model_path,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        save_steps=save_steps,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model()

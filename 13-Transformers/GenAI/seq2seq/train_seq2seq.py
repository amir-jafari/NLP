from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import TextDataset, DataCollatorForSeq2Seq
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Text2TextGenerationPipeline
from datasets import load_dataset
from utils import load_data, load_data_collator, load_tokenizer
import os
import torch
import os
from dotenv import load_dotenv  #  pip3 install python-dotenv
from configparser import ConfigParser, ExtendedInterpolation

device = "cuda:0" if torch.cuda.is_available() else "cpu"

if __name__=='__main__':
    # Read config.ini file
    load_dotenv()
    config_file = os.environ['CONFIG_FILE']
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(f"../config/{config_file}")
    print(config['instance']['user'])

    # you need to set parameters
    model_name = config['seq2seq']['model_name']
    tokenizer = load_tokenizer(model_name)

    train_file_path = config['seq2seq']['file_path']

    print("################# load dataset ###################")
    data = load_dataset("csv", data_files=train_file_path)

    print("################# start train #################")
    model_path = config['seq2seq']['model_path']
    overwrite_output_dir = True
    per_device_train_batch_size = int(config['seq2seq']['batch_size'])
    num_train_epochs = int(config['seq2seq']['epochs'])
    save_steps = int(config['seq2seq']['save_steps'])
    training = True
    checkpoint = model_name

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    train_dataset = load_data(data, per_device_train_batch_size)
    data_collator = load_data_collator(tokenizer, checkpoint)
    tokenizer.save_pretrained(model_path)

    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
    model.save_pretrained(model_path)

    training_args = Seq2SeqTrainingArguments(
        output_dir=model_path,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        save_steps=save_steps,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset['train'],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model()
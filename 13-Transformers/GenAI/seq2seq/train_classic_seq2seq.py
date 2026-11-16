from transformers import T5ForConditionalGeneration
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from utils import load_data, load_data_collator, load_tokenizer, load_model
from transformers import AutoTokenizer
from datasets import load_dataset
import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
import os
from dotenv import load_dotenv  #  pip3 install python-dotenv
from configparser import ConfigParser, ExtendedInterpolation

class Dataset(torch.utils.data.Dataset):
    """Input = keywords, target = review text (both tokenized to 32 tokens)."""
    def __init__(self, df):
        self.texts = [tokenizer(str(text), padding='max_length', max_length=32, truncation=True,
                                return_tensors="pt") for text in df['keywords']]
        self.labels = [tokenizer(str(text), padding='max_length', max_length=32, truncation=True,
                                 return_tensors="pt") for text in df['review_body']]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


if __name__=='__main__':
    # Read config.ini file
    load_dotenv()
    config_file = os.environ['CONFIG_FILE']
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(f"../config/{config_file}")
    print(config['instance']['user'])

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_name = config['seq2seq']['model_name']
    tokenizer = load_tokenizer(model_name)

    train_file_path = config['seq2seq']['file_path']

    # data = load_dataset("csv", data_files=train_file_path)
    df= pd.read_csv(train_file_path)
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)          # shuffle, then 80/10/10 split
    n_train, n_val = int(0.8 * len(df)), int(0.9 * len(df))
    df_train, df_val, df_test = df.iloc[:n_train], df.iloc[n_train:n_val], df.iloc[n_val:]

    print(len(df_train), len(df_val), len(df_test))

    batch_size = 4
    train, val = Dataset(df_train), Dataset(df_val)

    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = config['seq2seq']['model_path']
    per_device_train_batch_size = int(config['seq2seq']['batch_size'])
    num_train_epochs = int(config['seq2seq']['epochs'])
    save_steps = int(config['seq2seq']['save_steps'])
    training = True

    checkpoint = model_name
    print("################# load dataset ###################")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    tokenizer.save_pretrained(model_path)

    model = load_model(model_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    model = model.to(device)

    for epoch in range(num_train_epochs):
        total_loss_train = 0
        model.train()
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label['input_ids'].squeeze(1).to(device)
            train_label[train_label == tokenizer.pad_token_id] = -100       # ignore padding in the loss
            mask = train_input['attention_mask'].squeeze(1).to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            model.zero_grad()

            # T5 creates decoder_input_ids by shifting the LABELS right (teacher forcing)
            output = model(input_ids=input_id, attention_mask=mask, labels=train_label)

            batch_loss = output.loss
            total_loss_train += batch_loss.item()
            batch_loss.backward()
            optimizer.step()

        total_loss_val = 0

        model.eval()
        with torch.no_grad():

            for val_input, val_label in val_dataloader:
                val_label = val_label['input_ids'].squeeze(1).to(device)
                val_label[val_label == tokenizer.pad_token_id] = -100       # ignore padding in the loss
                mask = val_input['attention_mask'].squeeze(1).to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                # T5 creates decoder_input_ids by shifting the LABELS right (teacher forcing)
                output = model(input_ids=input_id, attention_mask=mask, labels=val_label)

                batch_loss = output.loss
                total_loss_val += batch_loss.item()

            print(f"Epochs: {epoch + 1} | Train Loss: {total_loss_train / len(train_dataloader): .3f} "
                  f"| Val Loss: {total_loss_val / len(val_dataloader): .3f}")


    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
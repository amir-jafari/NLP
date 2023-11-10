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
    def __init__(self, df):
        self.labels = [tokenizer(text,
                                padding='max_length',
                                max_length=32,
                                truncation=True,
                                return_tensors="pt") for text in df['review_body']]
        self.texts = [tokenizer(" ".join(text),
                                padding='max_length',
                                max_length=32,
                                truncation=True,
                                return_tensors="pt") for text in df['keywords']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        texts = self.texts[idx]
        # y = np.array(self.labels[idx])
        y = self.labels[idx]
        return texts, y


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
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=123),
                                         [int(0.8 * len(df)), int(0.9 * len(df))])

    print(len(df_train), len(df_val), len(df_test))

    batch_size = 4
    train, val = Dataset(df_train), Dataset(df_val)

    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=batch_size)

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    model_path = config['seq2seq']['model_path']
    overwrite_output_dir = True
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
    criterion = torch.nn.CrossEntropyLoss()

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch in range(num_train_epochs):
        total_acc_train = 0
        total_loss_train = 0
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label['input_ids'].squeeze(1).to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input["input_ids"].squeeze(1).to(device)
            decoder_input_ids = model._shift_right(input_id).squeeze(1).to(device)

            model.zero_grad()

            output = model(input_ids=input_id, decoder_input_ids=decoder_input_ids, attention_mask=mask, labels=train_label)

            batch_loss = output.loss
            total_loss_train += batch_loss.item()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:
                val_label = val_label['input_ids'].squeeze(1).to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)
                decoder_input_ids = model._shift_right(input_id).squeeze(1).to(device)

                output = model(input_ids=input_id, decoder_input_ids=decoder_input_ids, attention_mask=mask, labels=val_label)

                batch_loss = output.loss
                total_loss_val += batch_loss.item()

            print(
                f"Epochs: {epoch + 1} | Train Loss: {total_loss_train / len(df_train): .3f} \
                | Val Loss: {total_loss_val / len(df_val): .3f} \
                | Val Accuracy: {total_acc_val / len(df_val): .3f}")


    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
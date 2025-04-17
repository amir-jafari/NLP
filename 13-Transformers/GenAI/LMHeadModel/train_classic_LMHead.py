from transformers import T5ForConditionalGeneration
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from utils import load_dataset, load_model, load_data_collator, load_tokenizer
from transformers import AutoTokenizer
import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm


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

    PATH = os.getcwd()
    train_file_path = '../../../../../../Data/Amazon_Review_sm.txt'

    model_name = 'gpt2'
    model_path = os.path.join(PATH, '../../../../../../NLG_results_sm/LM')
    overwrite_output_dir = True
    per_device_train_batch_size = 4
    num_train_epochs = 1
    save_steps = 5000

    tokenizer = load_tokenizer(model_name)
    train_dataset = load_dataset(train_file_path, tokenizer)
    data_collator = load_data_collator(tokenizer)

    dataloader = DataLoader(train_dataset, batch_size=per_device_train_batch_size, shuffle=True, collate_fn=data_collator)

    tokenizer.save_pretrained(model_path)

    model = load_model(model_name)
    # model = GPT2Model.from_pretrained(model_name)

    model.save_pretrained(model_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    # Set model to training mode
    model.train()

    for epoch in range(num_train_epochs):
        total_acc_train = 0
        total_loss_train = 0
        for train_input in dataloader:
            train_label = train_input['input_ids'].to(device)
            input_id = train_input["input_ids"].to(device)

            model.zero_grad()

            output = model(input_ids=input_id, labels=train_label)
            batch_loss = output.loss
            total_loss_train += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
        f"Epochs: {epoch + 1} | Train Loss: {total_loss_train / len(dataloader): .3f}"
        total_acc_val = 0
        total_loss_val = 0

        # with torch.no_grad():
        #
        #     for val_input, val_label in val_dataloader:
        #         val_label = val_label['input_ids'].squeeze(1).to(device)
        #         mask = val_input['attention_mask'].to(device)
        #         input_id = val_input['input_ids'].squeeze(1).to(device)
        #         decoder_input_ids = model._shift_right(input_id).squeeze(1).to(device)
        #
        #         output = model(input_ids=input_id, attention_mask=mask, labels=val_label)
        #
        #         batch_loss = output.loss
        #         total_loss_val += batch_loss.item()
        #
        #     print(
        #         f"Epochs: {epoch + 1} | Train Loss: {total_loss_train / len(df_train): .3f} \
        #         | Val Loss: {total_loss_val / len(df_val): .3f} \
        #         | Val Accuracy: {total_acc_val / len(df_val): .3f}")


    model.save_pretrained(model_path)

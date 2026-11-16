from torch.utils.data import DataLoader
from utils import load_dataset, load_model, load_data_collator, load_tokenizer
import os
import torch
from dotenv import load_dotenv  #  pip3 install python-dotenv
from configparser import ConfigParser, ExtendedInterpolation

# "Classic" training: the same GPT-2 fine-tuning as train_LMHead.py, but with our own PyTorch loop instead of Trainer

if __name__=='__main__':
    load_dotenv()
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(f"../config/{os.environ['CONFIG_FILE']}")

    train_file_path = config['LMHeadModel']['file_path']
    model_name = config['LMHeadModel']['model_name']
    model_path = config['LMHeadModel']['model_path'].rstrip('/') + '_classic/'
    per_device_train_batch_size = int(config['LMHeadModel']['batch_size'])
    num_train_epochs = int(config['LMHeadModel']['epochs'])

    tokenizer = load_tokenizer(model_name)
    train_dataset = load_dataset(train_file_path, tokenizer, column=config['LMHeadModel']['colunm_name'])
    data_collator = load_data_collator(tokenizer)

    dataloader = DataLoader(train_dataset, batch_size=per_device_train_batch_size, shuffle=True, collate_fn=data_collator)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # Set model to training mode
    model.train()

    for epoch in range(num_train_epochs):
        total_acc_train = 0
        total_loss_train = 0
        for train_input in dataloader:
            train_label = train_input['labels'].to(device)     # = input_ids; the model shifts them by one internally
            input_id = train_input["input_ids"].to(device)

            model.zero_grad()

            output = model(input_ids=input_id, labels=train_label)
            batch_loss = output.loss
            total_loss_train += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
        print(f"Epochs: {epoch + 1} | Train Loss: {total_loss_train / len(dataloader): .3f}")

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print('Saved fine-tuned model to', model_path)

#%% --------------------------------------------------------------------------------------------------------------------
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% --------------------------------------------------------------------------------------------------------------------
dataset = load_dataset("ag_news")
train_ds = dataset["train"].shuffle().select(range(500))
val_ds = dataset["test"].shuffle().select(range(125))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4).to(device)

#%% --------------------------------------------------------------------------------------------------------------------
train_ds = train_ds.map(tokenize_function, batched=True)
val_ds = val_ds.map(tokenize_function, batched=True)
train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

#%% --------------------------------------------------------------------------------------------------------------------
model.train()
for epoch in range(1):
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

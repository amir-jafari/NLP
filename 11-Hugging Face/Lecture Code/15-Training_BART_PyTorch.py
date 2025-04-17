#%% --------------------------------------------------------------------------------------------------------------------
from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

def preprocess(examples):
    inputs = tokenizer(examples['text'], max_length=64, truncation=True, padding='max_length')
    labels = tokenizer(examples['text'], max_length=32, truncation=True, padding='max_length')
    inputs["labels"] = labels["input_ids"]
    return inputs

#%% --------------------------------------------------------------------------------------------------------------------
dataset = load_dataset("ag_news", split="train[:200]").shuffle(seed=42)
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
tokenized_ds = dataset.map(preprocess, batched=True)
tokenized_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
train_loader = DataLoader(tokenized_ds, batch_size=32, shuffle=True, pin_memory=True)
optimizer = optim.AdamW(model.parameters(), lr=3e-5)

#%% --------------------------------------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

#%% --------------------------------------------------------------------------------------------------------------------
for epoch in range(1):
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

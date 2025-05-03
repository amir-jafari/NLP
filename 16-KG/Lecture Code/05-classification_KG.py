#%% --------------------------------------------------------------------------------------------------------------------
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from datasets import load_dataset
from sklearn.metrics import classification_report, accuracy_score
#%% --------------------------------------------------------------------------------------------------------------------
dataset_tr = load_dataset("yelp_polarity", split="train[:1000]")
dataset_te = load_dataset("yelp_polarity", split="test[:200]")
train_texts = dataset_tr["text"]
train_labels = dataset_tr["label"]
train_facts = []
pos_words = {"good","great","excellent","amazing","love","wonderful","best","awesome"}
neg_words = {"bad","terrible","poor","hate","awful","worst"}
for text in train_texts:
    words = text.lower().split()
    pos = sum(w in pos_words for w in words)
    neg = sum(w in neg_words for w in words)
    train_facts.append([f"pos_count:{pos}", f"neg_count:{neg}"])
#%% --------------------------------------------------------------------------------------------------------------------
test_texts = dataset_te["text"]
test_labels = dataset_te["label"]
test_facts = []
for text in test_texts:
    words = text.lower().split()
    pos = sum(w in pos_words for w in words)
    neg = sum(w in neg_words for w in words)
    test_facts.append([f"pos_count:{pos}", f"neg_count:{neg}"])
#%% --------------------------------------------------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
device = "cuda" if torch.cuda.is_available() else "cpu"
#%% --------------------------------------------------------------------------------------------------------------------
class ClassificationDataset(Dataset):
    def __init__(self, texts, labels, kg_facts, tokenizer, max_len=128, use_kg=False):
        self.texts, self.labels = texts, labels
        self.kg_facts, self.tokenizer = kg_facts, tokenizer
        self.max_len, self.use_kg = max_len, use_kg
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        txt = self.texts[idx]
        if self.use_kg:
            facts = " ".join(self.kg_facts[idx])
            txt = f"{txt} Facts:{facts}"
        enc = self.tokenizer(txt, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze().to(device),
                "attention_mask": enc["attention_mask"].squeeze().to(device),
                "labels": torch.tensor(self.labels[idx]).to(device)}
#%% --------------------------------------------------------------------------------------------------------------------
def train(model, loader, optimizer):
    model.train()
    for b in loader:
        optimizer.zero_grad()
        loss = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"]).loss
        loss.backward()
        optimizer.step()
def evaluate(model, loader):
    model.eval(); preds, trues = [], []
    with torch.no_grad():
        for b in loader:
            logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            trues.extend(b["labels"].cpu().numpy())
    return trues, preds
for use_kg in [False, True]:
    ds_tr = ClassificationDataset(train_texts, train_labels, train_facts, tokenizer, use_kg=use_kg)
    ds_te = ClassificationDataset(test_texts, test_labels, test_facts, tokenizer, use_kg=use_kg)
    lt = DataLoader(ds_tr, batch_size=16, shuffle=True)
    le = DataLoader(ds_te, batch_size=16)
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)
    opt = AdamW(model.parameters(), lr=2e-5)
    train(model, lt, opt)
    true, pred = evaluate(model, le)
    print(f"\n=== {'With' if use_kg else 'Without'} KG ===")
    print(f"Acc: {accuracy_score(true, pred):.2f}")
    print(classification_report(true, pred, zero_division=0))

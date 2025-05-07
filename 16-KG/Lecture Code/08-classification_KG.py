#%% --------------------------------------------------------------------------------------------------------------------
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from datasets import load_dataset
from sklearn.metrics import classification_report, accuracy_score
import networkx as nx
from pyvis.network import Network
import webbrowser
import os
#%% --------------------------------------------------------------------------------------------------------------------
dataset_tr = load_dataset("yelp_polarity", split="train[:1000]")
dataset_te = load_dataset("yelp_polarity", split="test[:200]")

train_texts, train_labels = dataset_tr["text"], dataset_tr["label"]
test_texts,  test_labels  = dataset_te["text"],  dataset_te["label"]

pos_words = {"good","great","excellent","amazing","love","wonderful","best","awesome"}
neg_words = {"bad","terrible","poor","hate","awful","worst"}
#%% --------------------------------------------------------------------------------------------------------------------
def compute_facts(texts):
    facts = []
    for t in texts:
        words = t.lower().split()
        pos = sum(w in pos_words for w in words)
        neg = sum(w in neg_words for w in words)
        facts.append([f"pos_count:{pos}", f"neg_count:{neg}"])
    return facts

train_facts = compute_facts(train_texts)
test_facts  = compute_facts(test_texts)
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
            # Inject KG facts into the text
            txt = f"{txt} Facts: {' '.join(self.kg_facts[idx])}"
        enc = self.tokenizer(txt, padding="max_length", truncation=True,
                             max_length=self.max_len, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze().to(device),
            "attention_mask": enc["attention_mask"].squeeze().to(device),
            "labels": torch.tensor(self.labels[idx]).to(device)
        }

def train(model, loader, optimizer):
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()

def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"]).logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            trues.extend(batch["labels"].cpu().numpy())
    return trues, preds
#%% --------------------------------------------------------------------------------------------------------------------
# Train & evaluate both without and with KG facts
for use_kg in [False, True]:
    ds_tr = ClassificationDataset(train_texts, train_labels, train_facts, tokenizer, use_kg=use_kg)
    ds_te = ClassificationDataset(test_texts,  test_labels,  test_facts,  tokenizer, use_kg=use_kg)
    loader_tr = DataLoader(ds_tr, batch_size=16, shuffle=True)
    loader_te = DataLoader(ds_te, batch_size=16)

    model = AutoModelForSequenceClassification \
        .from_pretrained("distilbert-base-uncased", num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    train(model, loader_tr, optimizer)
    true_labels, pred_labels = evaluate(model, loader_te)

    print(f"\n=== {'With' if use_kg else 'Without'} KG ===")
    print(f"Accuracy: {accuracy_score(true_labels, pred_labels):.2f}")
    print(classification_report(true_labels, pred_labels, zero_division=0))

#%% --------------------------------------------------------------------------------------------------------------------
# Build a small KG graph and visualize it
G = nx.Graph()
for i, facts in enumerate(test_facts[:10]):
    doc_node = f"doc_{i}"
    G.add_node(doc_node, label=doc_node)
    for fact in facts:
        G.add_node(fact, label=fact)
        G.add_edge(doc_node, fact)

net = Network(height="400px", width="100%", bgcolor="#ffffff", font_color="black")
net.from_nx(G)
net.write_html("classification_kg_graph.html", open_browser=True)
print("Generated classification_kg_graph.html")
#%% --------------------------------------------------------------------------------------------------------------------

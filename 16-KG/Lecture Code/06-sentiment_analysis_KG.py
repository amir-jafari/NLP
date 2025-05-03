#%% --------------------------------------------------------------------------------------------------------------------
import torch
import numpy as np
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from transformers import pipeline
from scipy.sparse import hstack
#%% --------------------------------------------------------------------------------------------------------------------
device = 0 if torch.cuda.is_available() else -1
#%% --------------------------------------------------------------------------------------------------------------------
ds = load_from_disk("./yelp_polarity")
texts = np.array(ds["train"]["text"][:5000])
labels = np.array(ds["train"]["label"][:5000])
test_texts = np.array(ds["test"]["text"][:1000])
test_labels = np.array(ds["test"]["label"][:1000])
#%% --------------------------------------------------------------------------------------------------------------------
pos_words = {"good","great","excellent","amazing","love","wonderful","best","awesome"}
neg_words = {"bad","terrible","poor","hate","awful","worst"}
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device
)

def compute_kg_features(txts):
    feats = []
    for t in txts:
        w = t.lower().split()
        pos = sum(x in pos_words for x in w)
        neg = sum(x in neg_words for x in w)
        ratio = pos/(pos+neg+1e-6)
        score = sentiment_analyzer(t[:512])[0]
        val = score["score"] if score["label"]=="POSITIVE" else -score["score"]
        feats.append([pos, neg, ratio, val])
    return np.array(feats)

kg_train = compute_kg_features(texts)
kg_test  = compute_kg_features(test_texts)
#%% --------------------------------------------------------------------------------------------------------------------
vec = TfidfVectorizer(max_features=10000, stop_words="english")
X_tr_text = vec.fit_transform(texts)
X_te_text = vec.transform(test_texts)
#%% --------------------------------------------------------------------------------------------------------------------
scaler = StandardScaler()
kg_tr_s = scaler.fit_transform(kg_train)
kg_te_s = scaler.transform(kg_test)
#%% --------------------------------------------------------------------------------------------------------------------
def run(use_kg):
    if use_kg:
        Xtr = hstack([X_tr_text, kg_tr_s])
        Xte = hstack([X_te_text, kg_te_s])
    else:
        Xtr, Xte = X_tr_text, X_te_text

    gs = GridSearchCV(LogisticRegression(max_iter=1000, solver="liblinear"),
                      {"C":[0.1,1,10]}, cv=3, scoring="accuracy", n_jobs=-1)
    gs.fit(Xtr, labels)
    p = gs.predict(Xte)
    print(f"\n=== {'With' if use_kg else 'Without'} KG ===")
    print("Best C:", gs.best_params_["C"])
    print("Acc:", accuracy_score(test_labels,p))
    print(classification_report(test_labels,p,zero_division=0))

run(False)
run(True)

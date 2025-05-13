#%% --------------------------------------------------------------------------------------------------------------------
import logging
from gensim import corpora, similarities
from collections import defaultdict
import os
import torch
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#%% --------------------------------------------------------------------------------------------------------------------
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()
#%% --------------------------------------------------------------------------------------------------------------------
# 1. Prepare a small example corpus
documents = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey",
]
print(f"Using example corpus with {len(documents)} documents")
#%% --------------------------------------------------------------------------------------------------------------------
# 2. Preprocess text (tokenize, remove stopwords, filter rare tokens)
print("\n#%% Step 2: Preprocess (tokenize, remove stopwords, filter rare tokens)")
stoplist = set('for a of the and to in'.split())
texts = [[word for word in doc.lower().split() if word not in stoplist] for doc in documents]
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
processed_texts = [[token for token in text if frequency[token] > 1] for text in texts]
print(f"Sample processed text[0]: {processed_texts[0]}")
#%% --------------------------------------------------------------------------------------------------------------------
# 3. Build Dictionary and Corpus
print("\n#%% Step 3: Dictionary & Corpus (Optional)")
dictionary = corpora.Dictionary(processed_texts)
dictionary.filter_extremes(no_below=2, no_above=0.5)
print(f"Dictionary size after filtering: {len(dictionary)} tokens")
corpus = [dictionary.doc2bow(text) for text in processed_texts]
print(f"First document BOW: {corpus[0]}")
#%% --------------------------------------------------------------------------------------------------------------------
# 4. Load a Transformers Model & Generate Document Embeddings
print("\n#%% Step 4: Transformers Embeddings")
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
embeddings = [get_bert_embedding(doc) for doc in documents]
print(f"Generated {len(embeddings)} embeddings of size {len(embeddings[0])} each (for BERT-base).")
#%% --------------------------------------------------------------------------------------------------------------------
# Convert each dense embedding into a list of (index, value) pairs
embedding_size = len(embeddings[0])
embedding_corpus = []
for emb in embeddings:
    embedding_corpus.append(list(enumerate(emb)))
#%% --------------------------------------------------------------------------------------------------------------------
# 5. Build Similarity Index on Dense Embeddings
print("\n#%% Step 5: Build Similarity Index")
index = similarities.MatrixSimilarity(embedding_corpus, num_features=embedding_size)
#%% --------------------------------------------------------------------------------------------------------------------
# 6. Example: Similarity Query using Transformers
print("\n#%% Step 6: Similarity Query")
query_text = "Human computer interface"
query_emb = get_bert_embedding(query_text)
query_vector = list(enumerate(query_emb))
sims = index[query_vector]
print(f"Similarity scores for query '{query_text}':")
for doc_id, score in sorted(enumerate(sims), key=lambda x: -x[1])[:5]:
    print(f"Document {doc_id}\tScore: {score:.3f}")
#%% --------------------------------------------------------------------------------------------------------------------
# 7. Save all relevant artifacts to 'gensim_transformers'
print("\n#%% Step 7: Save all generated items")
os.makedirs("gensim_transformers", exist_ok=True)
dictionary.save("gensim_transformers/dictionary.dict")
corpora.MmCorpus.serialize("gensim_transformers/corpus.mm", corpus)
index.save("gensim_transformers/bert_similarity.index")


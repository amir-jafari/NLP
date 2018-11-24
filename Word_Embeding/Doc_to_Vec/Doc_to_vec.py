# sudo pip install --ignore-installed scipy
# sudo pip install --ignore-installed gensim
# sudo pip3 install --ignore-installed gensim
# -------------------------------------------------------------------------------------
import gensim.models as g
import logging
# -------------------------------------------------------------------------------------
vector_size = 300
window_size = 15
min_count = 1
sampling_threshold = 1e-5
negative_size = 5
train_epoch = 100
dm = 0
worker_count = 1
# -------------------------------------------------------------------------------------
pretrained_emb = "pretrained_word_embeddings.txt"
train_corpus = "train_docs.txt"
saved_path = "model.txt"
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
docs = g.doc2vec.TaggedLineDocument(train_corpus)
model = g.Doc2Vec(docs, size=vector_size, window=window_size, min_count=min_count, sample=sampling_threshold, workers=worker_count, hs=0, dm=dm, negative=negative_size, dbow_words=1, dm_concat=1, pretrained_emb=pretrained_emb, iter=train_epoch)
# -------------------------------------------------------------------------------------
model.save(saved_path)
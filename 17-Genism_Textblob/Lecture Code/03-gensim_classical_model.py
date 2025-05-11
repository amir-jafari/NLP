#%% --------------------------------------------------------------------------------------------------------------------
import logging
from gensim import corpora, models, similarities
from collections import defaultdict
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
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
# 2. Preprocess text
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
print("\n#%% Step 3: Dictionary & Corpus")
dictionary = corpora.Dictionary(processed_texts)
dictionary.filter_extremes(no_below=2, no_above=0.5)
print(f"Dictionary size after filtering: {len(dictionary)} tokens")
corpus = [dictionary.doc2bow(text) for text in processed_texts]
print(f"First document BOW: {corpus[0]}")
#%% --------------------------------------------------------------------------------------------------------------------
# 4. TF-IDF Transformation
print("\n#%% Step 4: TF-IDF Transformation")
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
print(f"TF-IDF vector for doc 0: {list(corpus_tfidf)[0]}")
#%% --------------------------------------------------------------------------------------------------------------------
# 5. Train LSI Model
print("\n#%% Step 5: Train LSI Model")
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=5)
#%% --------------------------------------------------------------------------------------------------------------------
# 6. Display LSI Topics
print("\n#%% Step 6: Display LSI Topics")
for idx, topic in lsi.print_topics(num_topics=5):
    print(f"Topic #{idx}: {topic}")
#%% --------------------------------------------------------------------------------------------------------------------
# 7. Similarity Query
print("\n#%% Step 7: Similarity Query")
index = similarities.MatrixSimilarity(lsi[corpus_tfidf], num_features=len(dictionary))
query = "Human computer interface"
vec_bow = dictionary.doc2bow(query.lower().split())
vec_lsi = lsi[tfidf[vec_bow]]
sims = index[vec_lsi]
print(f"Similarity scores for query '{query}':")
for doc_id, score in sorted(enumerate(sims), key=lambda x: -x[1])[:5]:
    print(f"Document {doc_id}\tScore: {score:.3f}")
#%% --------------------------------------------------------------------------------------------------------------------
# Save all generated documents to the 'gensim_classical' folder
os.makedirs("gensim_classical", exist_ok=True)
dictionary.save("gensim_classical/dictionary.dict")
corpora.MmCorpus.serialize("gensim_classical/corpus.mm", corpus)
tfidf.save("gensim_classical/tfidf.model")
lsi.save("gensim_classical/lsi_lee.model")
index.save("gensim_classical/lsi_index.index")
print("All items have been saved to the 'gensim_classical' folder.")

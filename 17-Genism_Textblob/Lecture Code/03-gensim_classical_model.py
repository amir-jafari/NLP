#%% --------------------------------------------------------------------------------------------------------------------
import gensim.downloader as api
from gensim import corpora
from gensim.models import LdaModel
#%% --------------------------------------------------------------------------------------------------------------------
dataset = api.load("text8")
data = [d for d in dataset]
#%% --------------------------------------------------------------------------------------------------------------------
texts = data
#%% --------------------------------------------------------------------------------------------------------------------
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
#%% --------------------------------------------------------------------------------------------------------------------
lda_model = LdaModel(corpus=corpus,id2word=dictionary,num_topics=5,passes=5)
#%% --------------------------------------------------------------------------------------------------------------------
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx}:\n{topic}\n")
#%% --------------------------------------------------------------------------------------------------------------------
new_doc = "human computer interaction".split()
bow_vector = dictionary.doc2bow(new_doc)
topic_distribution = lda_model.get_document_topics(bow_vector)
print("New doc topic distribution:", topic_distribution)

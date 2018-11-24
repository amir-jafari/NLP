# sudo pip install --ignore-installed scipy
# sudo pip install --ignore-installed gensim
# sudo pip3 install --ignore-installed gensim
# -------------------------------------------------------------------------------------
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec
# -------------------------------------------------------------------------------------

data = ["I love machine learning. Its awesome.",
        "I love coding in python",
        "I love building chatbots",
        "they chat amagingly well"]
# -------------------------------------------------------------------------------------
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]
# -------------------------------------------------------------------------------------
max_epochs = 100
vec_size = 20
alpha = 0.025
# -------------------------------------------------------------------------------------
model = Doc2Vec(size=vec_size, alpha=alpha, min_alpha=0.00025, min_count=1, dm=1)

model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    model.alpha -= 0.0002
    model.min_alpha = model.alpha
# -------------------------------------------------------------------------------------
model.save("d2v.model")
print("Model Saved")
# -------------------------------------------------------------------------------------
model= Doc2Vec.load("d2v.model")
test_data = word_tokenize("I love chatbots".lower())
v1 = model.infer_vector(test_data)
print("V1_infer", v1)
# -------------------------------------------------------------------------------------
similar_doc = model.docvecs.most_similar('1')
print(similar_doc)
print(model.docvecs['1'])
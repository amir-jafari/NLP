doc1 = "Data Science Machine Learning"
doc2 = "Money fun Family Kids home"
doc3 = "Programming Java Data Structures"
doc4 = "Love food health games energy fun"
doc5 = "Algorithms Data Computers"

doc_complete = [doc1, doc2, doc3, doc4, doc5]

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X =vectorizer.fit_transform(doc_complete)

from sklearn.decomposition import TruncatedSVD
lsa = TruncatedSVD(n_components=2,n_iter=100)
lsa.fit(X)
terms = vectorizer.get_feature_names()

for i,comp in enumerate(lsa.components_):
    termsInComp = zip(terms,comp)
    sortedterms = sorted(termsInComp, key=lambda x: x[1],reverse=True)[:10]
    print("Concept %d:" % i)
    for term in sortedterms:
        print(term[0])
    print(" ")

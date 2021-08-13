from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics
import numpy as np

'''
The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents, 
partitioned (nearly) evenly across 20 different newsgroups. To the best of our knowledge, 
it was originally collected by Ken Lang, probably for his paper “Newsweeder: Learning 
to filter netnews,” though he does not explicitly mention this collection. The 20 
newsgroups collection has become a popular data set for experiments in text applications
of machine learning techniques, such as text classification and text clustering.
'''

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

print(twenty_train.target_names)
print(len(twenty_train.data))
print(len(twenty_train.filenames))


tfidf= TfidfVectorizer()
tfidf.fit(twenty_train.data)
X_train_tfidf =tfidf.transform(twenty_train.data)
print(X_train_tfidf.shape)

clf = LogisticRegression().fit(X_train_tfidf, twenty_train.target)
X_test_tfidf = tfidf.transform(twenty_test.data)
predicted = clf.predict(X_test_tfidf)

print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))
print(metrics.confusion_matrix(twenty_test.target, predicted))
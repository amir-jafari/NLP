# -------------------------------------------------------------
# Gradient Boosting using scikit-learn on 20 Newsgroups
# -------------------------------------------------------------
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix


categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']


train_data = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42)
test_data  = fetch_20newsgroups(subset='test',  categories=categories,
                                shuffle=True, random_state=42)


tfidf = TfidfVectorizer()
tfidf.fit(train_data.data)

X_train = tfidf.transform(train_data.data)
y_train = train_data.target

X_test  = tfidf.transform(test_data.data)
y_test  = test_data.target

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
model.fit(X_train, y_train)


y_train_pred = model.predict(X_train)
y_test_pred  = model.predict(X_test)

print("Gradient Boosting (Classification) on 20 Newsgroups (4 categories):")
print("Training Classification Report:")
print(classification_report(y_train, y_train_pred, target_names=train_data.target_names))

print("Test Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=test_data.target_names))

print("Test Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

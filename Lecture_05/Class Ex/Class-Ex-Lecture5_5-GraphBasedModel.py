import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# Load data.csv
#  - Two columns: 'text' and 'label'
# ============================================================

"""
    Recommend you should try to load the data and split the data into train and test set by yourself.
    And also try to use the tfidf learned on the previous class!

"""

# df = pd.read_csv('data.csv')
#
#
# X_text = df['text'].values
# y = df['label'].values
#
# X_train_text, X_test_text, y_train, y_test = train_test_split(
#     X_text, y, test_size=0.3, random_state=42
# )
#
# tfidf = TfidfVectorizer()
# X_train_tfidf = tfidf.fit_transform(X_train_text)
# X_test_tfidf  = tfidf.transform(X_test_text)


# ============================================================
# Class_Ex1:
# Train a Decision Tree and measure accuracy on the test set.
# ------------------------------------------------------------
print(20 * '-' + 'Begin Q1' + 20 * '-')



print(20 * '-' + 'End Q1' + 20 * '-')

# ============================================================
# Class_Ex2:
# Adjust max_depth in Decision Tree to see if accuracy changes.
# Print accuracies for depths 1 to 5.
# ------------------------------------------------------------
print(20 * '-' + 'Begin Q2' + 20 * '-')



print(20 * '-' + 'End Q2' + 20 * '-')

# ============================================================
# Class_Ex3:
# Train a Random Forest and compare its accuracy to the Decision Tree.
# ------------------------------------------------------------
print(20 * '-' + 'Begin Q3' + 20 * '-')



print(20 * '-' + 'End Q3' + 20 * '-')

# ============================================================
# Class_Ex4:
# Train a Gradient Boosting Classifier (scikit-learn) and measure accuracy.
# ------------------------------------------------------------
print(20 * '-' + 'Begin Q4' + 20 * '-')



print(20 * '-' + 'End Q4' + 20 * '-')

# ============================================================
# Class_Ex5:
# Compare the accuracies of all three models on data.csv.
# ------------------------------------------------------------
print(20 * '-' + 'Begin Q5' + 20 * '-')



print(20 * '-' + 'End Q5' + 20 * '-')

# ============================================================
# (Optional) Class_Ex6:
# Compare scikit-learn's GB with scratch code.
# ------------------------------------------------------------
"""
Recommend you to build your personal scratch code using the theory in the class. 
And compare the result with the sklearn package.

"""



print(20 * '-' + 'Begin Q6' + 20 * '-')


"""
scratch_clf = GradientBoostingBinaryClassifierScratch(
    n_estimators=5,
    reg_lambda=1.0,
    learning_rate=0.1
)
# We'll need to transform the text data into numeric form
scratch_clf.fit(X_train_tfidf, y_train)
y_pred_scratch = scratch_clf.predict(X_test_tfidf)
acc_scratch = accuracy_score(y_test, y_pred_scratch)
print(f"From-Scratch Gradient Boosting Accuracy: {acc_scratch:.3f}")

print("Comparing scikit-learn GB vs. scratch GB:")
print(f"   sklearn GB Acc:  {acc_gb:.3f}")
print(f"   scratch GB Acc:  {acc_scratch:.3f}")

"""
print(20 * '-' + 'End Q6' + 20 * '-')



import pandas as pd

df_train = pd.read_csv('imdb_train.csv')
df_test = pd.read_csv('imdb_test.csv')


from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Official LIME
from lime.lime_text import LimeTextExplainer

pipeline = make_pipeline(
    TfidfVectorizer(stop_words='english'),
    LogisticRegression(max_iter=200)
)
pipeline.fit(df_train['text'], df_train['label'])

test_idx = 0
test_text  = df_test['text'].iloc[test_idx]
true_label = df_test['label'].iloc[test_idx]

explainer = LimeTextExplainer(class_names=["Negative", "Positive"])
exp = explainer.explain_instance(
    test_text,
    pipeline.predict_proba,
    num_features=10
)

#%%
# ==========================================================================
# The Below code is to create the bar chart for the word weight, the top one
# word is the most important word
# ==========================================================================
import matplotlib.pyplot as plt

# Use the explanation for the single test instance you already have:
lime_explanation = exp.as_list()

# Separate words and their corresponding weights
words, weights = zip(*lime_explanation)

# Create a simple bar chart
plt.figure(figsize=(8, 5))
colors = ['green' if w > 0 else 'red' for w in weights]
plt.barh(words, weights, color=colors)
plt.title("LIME Explanation for Test Instance #0")
plt.xlabel("Weight")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


#%%
import shap
import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\aaron\PycharmProjects\NLP\Other Lectures\Explainability\Lecture Code\Customer Churn.csv')
df.head()
# Dropping all irrelevant columns
df.drop(columns=['customerID'], inplace = True)

df.dropna(inplace=True)

#%%

# Label Encoding features
categorical_feat =list(df.select_dtypes(include=["object"]))

# Using label encoder to transform string categories to integer labels
le = LabelEncoder()
for feat in categorical_feat:
    df[feat] = le.fit_transform(df[feat]).astype('int')
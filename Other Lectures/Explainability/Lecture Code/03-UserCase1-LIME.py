import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lime.lime_text import LimeTextExplainer
from sklearn.metrics import accuracy_score
#%%---------------------------------------------------------------------------------------------------------------------
df_train = pd.read_csv('imdb_train.csv')
df_test = pd.read_csv('imdb_test.csv')
X_train, y_train = df_train['text'], df_train['label']
X_test, y_test   = df_test['text'], df_test['label']
#%%---------------------------------------------------------------------------------------------------------------------
pipeline_lr = make_pipeline(TfidfVectorizer(stop_words='english'),LogisticRegression(max_iter=200))
pipeline_rf = make_pipeline(TfidfVectorizer(stop_words='english'),RandomForestClassifier(n_estimators=100, random_state=42))
pipeline_xgb = make_pipeline(TfidfVectorizer(stop_words='english'),XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))

pipeline_lr.fit(X_train, y_train)
pipeline_rf.fit(X_train, y_train)
pipeline_xgb.fit(X_train, y_train)

#%%---------------------------------------------------------------------------------------------------------------------
pred_lr  = pipeline_lr.predict(X_test)
pred_rf  = pipeline_rf.predict(X_test)
pred_xgb = pipeline_xgb.predict(X_test)
acc_lr  = accuracy_score(y_test, pred_lr)
acc_rf  = accuracy_score(y_test, pred_rf)
acc_xgb = accuracy_score(y_test, pred_xgb)

print("Model Accuracy on Test Set:")
print("---------------------------")
print(f"Logistic Regression: {acc_lr:.3f}")
print(f"Random Forest:       {acc_rf:.3f}")
print(f"XGBoost:            {acc_xgb:.3f}")

#%%---------------------------------------------------------------------------------------------------------------------
test_idx = 0
test_text  = X_test.iloc[test_idx]
true_label = y_test.iloc[test_idx]

print(f"\nExplaining test instance #{test_idx}")
print("-----------------------------------")
print(f"True Label: {true_label}")
print("Review Snippet:", test_text[:200], "...")
#%%---------------------------------------------------------------------------------------------------------------------
explainer = LimeTextExplainer(class_names=["Negative", "Positive"])
exp_lr = explainer.explain_instance(test_text,pipeline_lr.predict_proba,num_features=10)
exp_rf = explainer.explain_instance(test_text,pipeline_rf.predict_proba,num_features=10)
exp_xgb = explainer.explain_instance(test_text,pipeline_xgb.predict_proba,num_features=10)

#%%---------------------------------------------------------------------------------------------------------------------
print("\nTop word contributions (LIME) - Logistic Regression:")
for word, weight in exp_lr.as_list():
    print(f"{word:<15} weight={weight:.3f}")
print("\nTop word contributions (LIME) - Random Forest:")
for word, weight in exp_rf.as_list():
    print(f"{word:<15} weight={weight:.3f}")
print("\nTop word contributions (LIME) - XGBoost:")
for word, weight in exp_xgb.as_list():
    print(f"{word:<15} weight={weight:.3f}")

from sklearn.datasets import fetch_20newsgroups

full_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
class_names = []
for name in full_train.target_names:
    if 'misc' not in name:
        short_name = name.split('.')[-1]
    else:
        short_name = '.'.join(name.split('.')[-2:])
    class_names.append(short_name)
class_names[3] = 'pc.hardware'
class_names[4] = 'mac.hardware'
print(f"Class Names: {', '.join(class_names)}")

#%%---------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from lime.lime_text import LimeTextExplainer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

#%%---------------------------------------------------------------------------
X_sub, _, y_sub, _ = train_test_split(
    full_train.data,
    full_train.target,
    test_size=0.8,
    random_state=42
)
vectorizer = TfidfVectorizer(lowercase=False, max_features=2000)
X_train_full = vectorizer.fit_transform(X_sub)  # Training vectors
y_train_full = y_sub
X_test_full = vectorizer.transform(newsgroups_test.data)  # Test vectors
y_test_full = newsgroups_test.target

#%%---------------------------------------------------------------------------
print("\n[1] Logistic Regression (max_iter=200)")
model_lr = LogisticRegression(max_iter=200)
model_lr.fit(X_train_full, y_train_full)
pred_lr = model_lr.predict(X_test_full)
f1_lr = f1_score(y_test_full, pred_lr, average='weighted')
print(f"  LR Weighted F1: {f1_lr:.3f}")

print("\n[2] Random Forest (n_estimators=50)")
model_rf = RandomForestClassifier(n_estimators=50, random_state=42)
model_rf.fit(X_train_full, y_train_full)
pred_rf = model_rf.predict(X_test_full)
f1_rf = f1_score(y_test_full, pred_rf, average='weighted')
print(f"  RF Weighted F1: {f1_rf:.3f}")

print("\n[3] XGBoost (n_estimators=50)")
model_xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric="mlogloss",
    n_estimators=50
)
model_xgb.fit(X_train_full, y_train_full)
pred_xgb = model_xgb.predict(X_test_full)
f1_xgb = f1_score(y_test_full, pred_xgb, average='weighted')
print(f"  XGB Weighted F1: {f1_xgb:.3f}")

#%%---------------------------------------------------------------------------
test_idx = 0
test_text = newsgroups_test.data[test_idx]
true_label = newsgroups_test.target[test_idx]
print(f"\n=== Explaining test instance #{test_idx} ===")
print("--------------------------------------------")
print("True Label:", class_names[true_label])
print("Text (truncated):", test_text[:200], "...")

explainer = LimeTextExplainer(class_names=class_names)
def predict_proba_lr(texts):
    X_vec = vectorizer.transform(texts)
    return model_lr.predict_proba(X_vec)
def predict_proba_rf(texts):
    X_vec = vectorizer.transform(texts)
    return model_rf.predict_proba(X_vec)
def predict_proba_xgb(texts):
    X_vec = vectorizer.transform(texts)
    return model_xgb.predict_proba(X_vec)
#%%---------------------------------------------------------------------------
exp_lr = explainer.explain_instance(
    test_text,
    predict_proba_lr,
    num_features=6,
    top_labels=2
)
pred_label_lr = model_lr.predict(vectorizer.transform([test_text]))[0]
print("\n=== LIME Explanation (Logistic Regression) ===")
print("Predicted class =", class_names[pred_label_lr])
for feat, weight in exp_lr.as_list(label=pred_label_lr):
    print(f"{feat:<20} weight={weight:.3f}")

exp_rf = explainer.explain_instance(
    test_text,
    predict_proba_rf,
    num_features=6,
    top_labels=2
)
pred_label_rf = model_rf.predict(vectorizer.transform([test_text]))[0]
print("\n=== LIME Explanation (Random Forest) ===")
print("Predicted class =", class_names[pred_label_rf])
for feat, weight in exp_rf.as_list(label=pred_label_rf):
    print(f"{feat:<20} weight={weight:.3f}")

exp_xgb = explainer.explain_instance(
    test_text,
    predict_proba_xgb,
    num_features=6,
    top_labels=2
)
pred_label_xgb = model_xgb.predict(vectorizer.transform([test_text]))[0]
print("\n=== LIME Explanation (XGBoost) ===")
print("Predicted class =", class_names[pred_label_xgb])
for feat, weight in exp_xgb.as_list(label=pred_label_xgb):
    print(f"{feat:<20} weight={weight:.3f}")

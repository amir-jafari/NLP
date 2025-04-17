import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

"""

A stump means a simple decision tree with one split. It always used as a weak learner.

Official Expl: A decision stump is a machine learning model consisting of a one-level decision tree.
 That is, it is a decision tree with one internal node (the root) which is immediately connected 
to the terminal nodes (its leaves). A decision stump makes a prediction based on the value of just 
a single input feature. Sometimes they are also called 1-rules

"""



###############################################################################
# Minimal XGBoost-Style Stump
###############################################################################
class XGBoostStump:

    def __init__(self, reg_lambda=1.0):
        self.reg_lambda = reg_lambda
        self.feature_index_ = None
        self.threshold_ = None
        self.weight_left_ = 0.0
        self.weight_right_ = 0.0

    def fit(self, X, grad, hess):
        if hasattr(X, "toarray"):
            X = X.toarray()

        n_samples, n_features = X.shape
        G_total = np.sum(grad)
        H_total = np.sum(hess)

        best_gain = float("-inf")

        self.feature_index_ = None
        self.threshold_ = None
        best_weight_no_split = - G_total / (H_total + self.reg_lambda)
        self.weight_left_ = best_weight_no_split
        self.weight_right_ = 0.0

        for feat_idx in range(n_features):
            feature_values = X[:, feat_idx]
            unique_vals = np.unique(feature_values)

            for thr in unique_vals:
                left_mask = (feature_values <= thr)
                right_mask = ~left_mask

                G_left = np.sum(grad[left_mask])
                H_left = np.sum(hess[left_mask])
                G_right = np.sum(grad[right_mask])
                H_right = np.sum(hess[right_mask])


                w_parent = (G_total**2) / (H_total + self.reg_lambda)
                w_left   = (G_left**2)  / (H_left  + self.reg_lambda)
                w_right  = (G_right**2) / (H_right + self.reg_lambda)

                gain = 0.5 * (w_left + w_right - w_parent)

                if gain > best_gain:
                    best_gain = gain
                    self.feature_index_ = feat_idx
                    self.threshold_ = thr
                    # Recompute actual leaf weights:
                    self.weight_left_  = - G_left  / (H_left  + self.reg_lambda)
                    self.weight_right_ = - G_right / (H_right + self.reg_lambda)

    def predict(self, X):
        if self.feature_index_ is None:
            return np.full(X.shape[0], self.weight_left_)


        if hasattr(X, "toarray"):
            X = X.toarray()

        feat_vals = X[:, self.feature_index_]
        left_mask = (feat_vals <= self.threshold_)
        return np.where(left_mask, self.weight_left_, self.weight_right_)


###############################################################################
# XGBoost-Style Binary Classifier (stump-based)
###############################################################################
class XGBoostBinaryClassifierScratch:

    def __init__(self, n_estimators=5, reg_lambda=1.0, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.reg_lambda = reg_lambda
        self.learning_rate = learning_rate
        self.stumps_ = []
        self.init_pred_ = 0.0

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, X, y):
        eps = 1e-9
        y_mean = np.clip(np.mean(y), eps, 1 - eps)
        self.init_pred_ = np.log(y_mean / (1 - y_mean))


        preds = np.full(X.shape[0], self.init_pred_)

        for _ in range(self.n_estimators):
            p = self._sigmoid(preds)
            grad = (p - y)
            hess = p * (1.0 - p)

            stump = XGBoostStump(reg_lambda=self.reg_lambda)
            stump.fit(X, grad, hess)

            update = stump.predict(X)
            preds += self.learning_rate * update

            self.stumps_.append(stump)

    def predict_proba(self, X):
        if hasattr(X, "toarray"):
            X_dense = X.toarray()
        else:
            X_dense = X

        log_odds = np.full(X_dense.shape[0], self.init_pred_)

        for stump in self.stumps_:
            log_odds += self.learning_rate * stump.predict(X_dense)
        return self._sigmoid(log_odds)

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)


###############################################################################
# Demo on 20 Newsgroups (Binary Classification)
###############################################################################
def main():
    categories = ['alt.atheism', 'soc.religion.christian']
    train_data = fetch_20newsgroups(subset='train', categories=categories,
                                    shuffle=True, random_state=42)
    test_data = fetch_20newsgroups(subset='test', categories=categories,
                                   shuffle=True, random_state=42)


    tfidf = TfidfVectorizer()
    tfidf.fit(train_data.data)
    X_train = tfidf.transform(train_data.data)
    X_test = tfidf.transform(test_data.data)

    y_train = train_data.target  # 0 or 1
    y_test  = test_data.target   # 0 or 1

    print(f"Training samples: {X_train.shape}, Testing samples: {X_test.shape}")
    print("Labels:", np.unique(y_train), categories)

    clf = XGBoostBinaryClassifierScratch(
        n_estimators=5,     # small number of boosting rounds
        reg_lambda=1.0,     # L2 regularization
        learning_rate=0.1   # shrinkage
    )
    clf.fit(X_train, y_train)


    y_pred_train = clf.predict(X_train)
    y_pred_test  = clf.predict(X_test)

    print("\n=== XGBoost-Style Scratch Model ===")
    print("Train Classification Report:")
    print(metrics.classification_report(y_train, y_pred_train, target_names=categories))

    print("Test Classification Report:")
    print(metrics.classification_report(y_test, y_pred_test, target_names=categories))

    print("Confusion Matrix (Test):")
    print(metrics.confusion_matrix(y_test, y_pred_test))


if __name__ == "__main__":
    main()

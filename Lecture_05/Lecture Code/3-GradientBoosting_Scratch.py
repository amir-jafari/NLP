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
    """
    A single-split 'tree' (stump) that uses second-order approximations
    and L2 regularization to find an optimal threshold on one feature.
    This is a simplified illustration of XGBoost logic for *binary* classification.
    """

    def __init__(self, reg_lambda=1.0):
        """
        reg_lambda: float
            L2 regularization strength (the 'lambda' in the theory).
        """
        self.reg_lambda = reg_lambda
        self.feature_index_ = None
        self.threshold_ = None
        self.weight_left_ = 0.0
        self.weight_right_ = 0.0

    def fit(self, X, grad, hess):
        """
        Fit one stump using second-order stats (grad, hess).

        Parameters
        ----------
        X    : array-like of shape (n_samples, n_features)
               TF–IDF features in dense or sparse format.
        grad : array of shape (n_samples,) -- (p_i - y_i) for logistic loss
        hess : array of shape (n_samples,) -- p_i * (1 - p_i)
        """
        # Convert to dense if sparse
        if hasattr(X, "toarray"):
            X = X.toarray()

        n_samples, n_features = X.shape
        G_total = np.sum(grad)
        H_total = np.sum(hess)

        best_gain = float("-inf")

        # If we do not split at all, we have one leaf weight:
        # w* = -G / (H + lambda).
        # We'll store that as our "no-split" baseline.
        self.feature_index_ = None
        self.threshold_ = None
        best_weight_no_split = - G_total / (H_total + self.reg_lambda)
        self.weight_left_ = best_weight_no_split
        self.weight_right_ = 0.0

        # Brute-force over features & possible thresholds
        for feat_idx in range(n_features):
            feature_values = X[:, feat_idx]
            unique_vals = np.unique(feature_values)

            for thr in unique_vals:
                # Left side: X[feat_idx] <= thr
                left_mask = (feature_values <= thr)
                # Right side: X[feat_idx] > thr
                right_mask = ~left_mask

                G_left = np.sum(grad[left_mask])
                H_left = np.sum(hess[left_mask])
                G_right = np.sum(grad[right_mask])
                H_right = np.sum(hess[right_mask])

                # Gain formula for a split in second-order approximation (no gamma for simplicity):
                # Gain = 0.5 * [G_left^2/(H_left + lambda) + G_right^2/(H_right + lambda) - G_total^2/(H_total + lambda)]
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
        """
        Predict log-odds shift for each sample.
        We'll add this shift to the existing model's log-odds.
        """
        if self.feature_index_ is None:
            # No split found, return single weight for all
            return np.full(X.shape[0], self.weight_left_)

        # Convert to dense if sparse
        if hasattr(X, "toarray"):
            X = X.toarray()

        feat_vals = X[:, self.feature_index_]
        left_mask = (feat_vals <= self.threshold_)
        return np.where(left_mask, self.weight_left_, self.weight_right_)


###############################################################################
# XGBoost-Style Binary Classifier (stump-based)
###############################################################################
class XGBoostBinaryClassifierScratch:
    """
    Minimal logistic regression boosting with second-order approximation.
    Uses multiple XGBoostStumps in sequence to update log-odds predictions.
    """

    def __init__(self, n_estimators=5, reg_lambda=1.0, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.reg_lambda = reg_lambda
        self.learning_rate = learning_rate
        self.stumps_ = []
        self.init_pred_ = 0.0  # global bias in log-odds space

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, X, y):
        """
        Fit the boosting model for binary classification.

        Parameters
        ----------
        X : TF–IDF features, shape (n_samples, n_features) (sparse or dense)
        y : binary labels (0 or 1), shape (n_samples,)
        """
        # Initialize log-odds with log(y_mean / (1 - y_mean))
        # Add small clip to avoid log(0).
        eps = 1e-9
        y_mean = np.clip(np.mean(y), eps, 1 - eps)
        self.init_pred_ = np.log(y_mean / (1 - y_mean))

        # Current predictions (log-odds)
        preds = np.full(X.shape[0], self.init_pred_)

        for _ in range(self.n_estimators):
            # Compute gradient and hessian for logistic loss
            p = self._sigmoid(preds)         # predicted probability
            grad = (p - y)                   # first derivative
            hess = p * (1.0 - p)            # second derivative

            stump = XGBoostStump(reg_lambda=self.reg_lambda)
            stump.fit(X, grad, hess)

            # Update log-odds with stump output
            update = stump.predict(X)
            preds += self.learning_rate * update

            self.stumps_.append(stump)

    def predict_proba(self, X):
        """
        Predict probability of class 1.
        """
        # Start from initial log-odds
        if hasattr(X, "toarray"):
            # We'll repeatedly convert to avoid mismatch in predictions
            X_dense = X.toarray()
        else:
            X_dense = X

        log_odds = np.full(X_dense.shape[0], self.init_pred_)

        for stump in self.stumps_:
            log_odds += self.learning_rate * stump.predict(X_dense)
        return self._sigmoid(log_odds)

    def predict(self, X):
        """
        Return binary class predictions {0,1}.
        """
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)


###############################################################################
# Demo on 20 Newsgroups (Binary Classification)
###############################################################################
def main():
    # We'll limit to two classes for this from-scratch approach:
    #    0 -> alt.atheism, 1 -> soc.religion.christian
    # Handling 4+ classes from scratch is much more complex.
    categories = ['alt.atheism', 'soc.religion.christian']
    train_data = fetch_20newsgroups(subset='train', categories=categories,
                                    shuffle=True, random_state=42)
    test_data = fetch_20newsgroups(subset='test', categories=categories,
                                   shuffle=True, random_state=42)

    # Vectorize text into TF–IDF features
    tfidf = TfidfVectorizer()
    tfidf.fit(train_data.data)
    X_train = tfidf.transform(train_data.data)
    X_test = tfidf.transform(test_data.data)

    y_train = train_data.target  # 0 or 1
    y_test  = test_data.target   # 0 or 1

    print(f"Training samples: {X_train.shape}, Testing samples: {X_test.shape}")
    print("Labels:", np.unique(y_train), categories)

    # Train our from-scratch XGBoost classifier
    clf = XGBoostBinaryClassifierScratch(
        n_estimators=5,     # small number of boosting rounds
        reg_lambda=1.0,     # L2 regularization
        learning_rate=0.1   # shrinkage
    )
    clf.fit(X_train, y_train)

    # Evaluate
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

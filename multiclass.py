import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# -------------------------
# 1) Multi-class Perceptron
# -------------------------
class MultiClass:
    """
    One-vs-Rest Perceptron for multi-class classification.
    Each class has its own weight vector w_c.
    """

    def __init__(self, n_iter=10, learning_rate=1.0):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Train separate perceptrons in a One-vs-Rest fashion.
        """
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        # We'll keep one weight vector for each class, plus bias
        # self.weights shape: (n_classes, n_features + 1)
        self.weights = np.zeros((len(self.classes_), n_features + 1))

        # Add bias term by augmenting X with a column of 1s
        X_aug = np.c_[X, np.ones((n_samples, 1))]

        for _ in range(self.n_iter):
            for i in range(n_samples):
                xi = X_aug[i]
                target = y[i]
                for idx, cls in enumerate(self.classes_):
                    # For perceptron update: label = +1 if class matches, else -1
                    label = 1 if target == cls else -1
                    # Predict sign(w_c^T xi)
                    pred = np.sign(self.weights[idx].dot(xi))
                    if pred == 0:  # sign(0) -> 0, treat as misclassification
                        pred = -1
                    if pred != label:  # misclassified
                        self.weights[idx] += self.learning_rate * label * xi

    def _linear_scores(self, X):
        """
        Compute linear decision scores for all classes.
        X is assumed to be shape (n_samples, n_features).
        Returns array of shape (n_samples, n_classes).
        """
        # Augment with bias
        X_aug = np.c_[X, np.ones((X.shape[0], 1))]
        return X_aug.dot(self.weights.T)

    def predict(self, X):
        """
        Predict class based on which linear score is largest.
        """
        scores = self._linear_scores(X)
        # Pick the class with highest score
        indices = np.argmax(scores, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X):
        """
        Convert linear scores to pseudo-probabilities via softmax.
        (Perceptron doesn't naturally output probabilities; this is just a normalization.)
        Returns shape (n_samples, n_classes).
        """
        scores = self._linear_scores(X)
        # Numerical stability trick
        scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores_shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


# --------------------------------
# 2) Multi-class Logistic Regression
# --------------------------------
class MultiClassLogisticRegression:
    """
    Softmax-based multi-class logistic regression trained via gradient descent.
    """

    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.W = None
        self.b = None
        self.classes_ = None

    def _one_hot(self, y):
        """
        Convert class labels y to one-hot vectors.
        """
        one_hot = np.zeros((len(y), len(self.classes_)))
        for i, val in enumerate(y):
            one_hot[i, np.where(self.classes_ == val)[0]] = 1
        return one_hot

    def fit(self, X, y):
        """
        Train logistic regression with softmax on multi-class data.
        """
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        # Weight matrix: (n_features, n_classes)
        # Bias vector: (n_classes,)
        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros(n_classes)

        # Convert y to one-hot
        Y_one_hot = self._one_hot(y)

        for _ in range(self.n_iter):
            # Forward pass
            logits = X.dot(self.W) + self.b
            # Numerical stability: subtract max from each row
            logits -= np.max(logits, axis=1, keepdims=True)
            exp_scores = np.exp(logits)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Compute gradients
            grad_W = (1 / n_samples) * X.T.dot(probs - Y_one_hot)
            grad_b = (1 / n_samples) * np.sum(probs - Y_one_hot, axis=0)

            # Update parameters
            self.W -= self.learning_rate * grad_W
            self.b -= self.learning_rate * grad_b

    def predict_proba(self, X):
        """
        Compute the softmax probabilities.
        Returns an array of shape (n_samples, n_classes).
        """
        logits = X.dot(self.W) + self.b
        logits -= np.max(logits, axis=1, keepdims=True)  # for stability
        exp_scores = np.exp(logits)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    def predict(self, X):
        """
        Predict the class with the highest probability.
        """
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]


if __name__ == "__main__":
    # ---------------------------------------------------------
    # Load IRIS dataset and create train/test split
    # ---------------------------------------------------------
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------------------------------------------------
    # 1) Train and test Multi-class Perceptron
    # ---------------------------------------------------------
    perceptron = MultiClass(n_iter=10, learning_rate=0.1)
    perceptron.fit(X_train, y_train)
    y_pred_perc = perceptron.predict(X_test)
    accuracy_perc = np.mean(y_pred_perc == y_test)
    print("Perceptron accuracy:", accuracy_perc)

    # Probability example with perceptron (using softmax of raw scores)
    print("\nPerceptron probabilities for the first test sample:")
    sample = X_test[0].reshape(1, -1)
    probs_perc = perceptron.predict_proba(sample)[0]
    for cls, p in zip(perceptron.classes_, probs_perc):
        print(f"  Class {cls} probability: {p:.4f}")

    # ---------------------------------------------------------
    # 2) Train and test Multi-class Logistic Regression
    # ---------------------------------------------------------
    logreg = MultiClassLogisticRegression(learning_rate=0.01, n_iter=2000)
    logreg.fit(X_train, y_train)
    y_pred_logreg = logreg.predict(X_test)
    accuracy_logreg = np.mean(y_pred_logreg == y_test)
    print("\nLogistic Regression accuracy:", accuracy_logreg)

    # Probability example with logistic regression
    print("\nLogistic Regression probabilities for the first test sample:")
    probs_logreg = logreg.predict_proba(sample)[0]
    for cls, p in zip(logreg.classes_, probs_logreg):
        print(f"  Class {cls} probability: {p:.4f}")

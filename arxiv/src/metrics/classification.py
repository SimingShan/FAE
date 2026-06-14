"""Classification probes for PDE-class discrimination."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def classification_metrics(Z_train: np.ndarray, y_train: np.ndarray,
                            Z_val: np.ndarray, y_val: np.ndarray,
                            knn_k: int = 15) -> dict:
    """Compute LogReg-3class, kNN, and Adv F1.

    y_train, y_val: int labels 0=Heat, 1=Advection, 2=Burgers
    Returns: {"logreg", "knn", "adv_f1"}
    """
    lr = LogisticRegression(max_iter=2000).fit(Z_train, y_train)
    knn = KNeighborsClassifier(n_neighbors=knn_k).fit(Z_train, y_train)
    pred = knn.predict(Z_val)
    adv_f1 = 2 * np.sum((pred == 1) & (y_val == 1)) / max(
        np.sum(pred == 1) + np.sum(y_val == 1), 1)
    return {
        "logreg": float(lr.score(Z_val, y_val)),
        "knn":    float(knn.score(Z_val, y_val)),
        "adv_f1": float(adv_f1),
    }

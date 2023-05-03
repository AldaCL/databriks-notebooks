from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


def fit_grid_search(X: Union[np.ndarray, pd.DataFrame],
                    y: Union[np.ndarray, pd.Series]) -> GridSearchCV:
    """
    Fit a GridSearchCV for a KNeighborsClassifier with specified hyperparameters.

    This function performs a grid search with cross-validation for the KNeighborsClassifier,
    using the provided hyperparameters.

    :param X: The feature matrix.
    :type X: Union[np.ndarray, pd.DataFrame]
    :param y: The target vector.
    :type y: Union[np.ndarray, pd.Series]
    :return: The fitted GridSearchCV object with the optimal KNeighborsClassifier parameters.
    :rtype: GridSearchCV
    """
    n_neighbors = [6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22]
    algorithm = ["auto"]
    weights = ["uniform", "distance"]
    leaf_size = list(range(1, 50, 5))

    hyperparams = {
        "algorithm": algorithm,
        "weights": weights,
        "leaf_size": leaf_size,
        "n_neighbors": n_neighbors,
    }

    gd = GridSearchCV(
        estimator=KNeighborsClassifier(),
        param_grid=hyperparams,
        verbose=True,
        cv=10,
        scoring="roc_auc",
    )

    return gd.fit(X, y)


def get_predictions(
    leaf_size: int, n_neighbors: int, train: pd.DataFrame, val: pd.DataFrame
) -> np.ndarray:
    """
    Train a KNeighborsClassifier with the given hyperparameters and make predictions for the validation dataset.

    :param leaf_size: The leaf size for the KNeighborsClassifier.
    :type leaf_size: int
    :param n_neighbors: The number of nearest neighbors to consider in the KNeighborsClassifier.
    :type n_neighbors: int
    :param train: The training dataset.
    :type train: pd.DataFrame
    :param val: The validation dataset.
    :type val: pd.DataFrame
    :return: The predicted labels for the validation dataset.
    :rtype: np.ndarray
    """
    knn = KNeighborsClassifier(
        algorithm="auto",
        leaf_size=leaf_size,
        metric="minkowski",
        metric_params=None,
        n_jobs=1,
        n_neighbors=n_neighbors,
        p=2,
        weights="uniform",
    )

    train_X = train[train.columns[1:]]
    train_y = train[train.columns[:1]]
    test_X = val[val.columns[1:]]

    knn.fit(train_X, train_y.values.ravel())
    y_pred = knn.predict(test_X)

    return y_pred

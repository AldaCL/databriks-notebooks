# sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


def fit_grid_search(X, y):
    n_neighbors = [6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22]
    algorithm = ['auto']
    weights = ['uniform', 'distance']
    leaf_size = list(range(1, 50, 5))

    hyperparams = {'algorithm': algorithm, 'weights': weights,
                'leaf_size': leaf_size, 'n_neighbors': n_neighbors}

    gd = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=hyperparams,
                    verbose=True, cv=10, scoring="roc_auc")

    return gd.fit(X, y)


def get_predictions(leaf_size, n_neighbors,  train, val):
    knn = KNeighborsClassifier(algorithm='auto', leaf_size=leaf_size,
                               metric='minkowski', metric_params=None,
                               n_jobs=1, n_neighbors=n_neighbors, p=2,
                               weights='uniform')

    train_X = train[train.columns[1:]]
    train_y = train[train.columns[:1]]
    test_X = val[val.columns[1:]]

    knn.fit(train_X, train_y.values.ravel())
    y_pred = knn.predict(test_X)

    return y_pred
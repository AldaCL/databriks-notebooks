import numpy as np
import pandas as pd

from titanic_nb.data_processing.processing import fit_grid_search, get_predictions


def test_fit_grid_search(mocker):
    # Create sample input data
    X = np.random.rand(10, 4)
    y = np.random.randint(0, 2, 10)

    # Mock the GridSearchCV class and its fit method
    mock_grid_search_cv = mocker.patch(
        "titanic_nb.data_processing.processing.GridSearchCV"
    )
    mock_grid_search = mocker.Mock()
    mock_grid_search_cv.return_value = mock_grid_search

    # Call the fit_grid_search function
    fit_grid_search(X, y)

    # Check if the result is the same as the mock_grid_search object
    assert mock_grid_search_cv.call_count == 1

    # Verify that the fit method is called with the correct input data
    mock_grid_search.fit.assert_called_once_with(X, y)


def test_get_predictions(mocker):
    # Set a fixed seed for NumPy's random number generator
    np.random.seed(42)

    # Create sample input data
    leaf_size = 30
    n_neighbors = 6
    train = pd.DataFrame(
        {
            "Survived": [0, 1, 1, 0, 1],
            "A": np.random.rand(5),
            "B": np.random.rand(5),
            "C": np.random.rand(5),
            "D": np.random.rand(5),
        }
    )
    val = pd.DataFrame(np.random.rand(5, 4), columns=["A", "B", "C", "D"])

    # Align the columns of the train and val dataframes
    train, val = train.align(val, axis=1, join="inner")

    # Mock the KNeighborsClassifier class and its methods
    mock_kneighbors_classifier = mocker.patch(
        "titanic_nb.data_processing.processing.KNeighborsClassifier"
    )
    mock_knn = mocker.Mock()
    mock_kneighbors_classifier.return_value = mock_knn

    # Mock the fit method
    mock_knn.fit.return_value = mock_knn

    # Create a mock y_pred object
    mock_y_pred = np.random.randint(0, 2, 5)
    mock_knn.predict.return_value = mock_y_pred

    # Call the get_predictions function
    result = get_predictions(leaf_size, n_neighbors, train, val)

    # Check if the result is the same as the mock_y_pred object
    assert np.array_equal(result, mock_y_pred)

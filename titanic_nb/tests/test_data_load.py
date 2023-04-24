import pytest
from data_processing.load_data import get_dataframe


@pytest.mark.parametrize(
    "databricks_env, expected_path_prefix",
    [
        (True, "/dbfs/FileStore/shared_uploads/aldair.alda27@gmail.com/test_file.csv"),
        (False, "datasource/test_file.csv"),
    ],
)
def test_get_dataframe(mocker, databricks_env, expected_path_prefix, monkeypatch):
    if databricks_env:
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "dummy_value")
    else:
        monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    # Mock the pd.read_csv function and set its return value to the sample DataFrame
    pandas_mock = mocker.patch("data_processing.load_data.pd.read_csv")

    # Call the get_dataframe function
    filename = "test_file"
    get_dataframe(filename)
    pandas_mock.assert_called_once_with(expected_path_prefix)

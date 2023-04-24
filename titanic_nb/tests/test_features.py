import pandas as pd
from data_processing import features as titanic_features


def test_clean_names(raw_data):
    cleaned_data = titanic_features.clean_names(raw_data)
    assert isinstance(cleaned_data, pd.DataFrame)
    assert "Lastname" in cleaned_data.columns
    assert "Title" not in cleaned_data.columns

    expected_lastnames = [
        "Braund",
        "Cumings",
        "Heikkinen",
        "Futrelle",
        "Allen",
        "Green",
        "Brown",
        "Smith",
    ]
    assert list(cleaned_data["Lastname"]) == expected_lastnames

    # Test if Age is imputed correctly for missing values
    raw_data_with_missing_age = raw_data.copy()
    raw_data_with_missing_age.loc[0, "Age"] = None
    cleaned_data_with_imputed_age = titanic_features.clean_names(
        raw_data_with_missing_age
    )
    assert not cleaned_data_with_imputed_age["Age"].isnull().any()


def test_fill_missing_fares(family_survival_data):
    family_survival_data["Fare"][3] = None
    titanic_features.fill_missing_fares(family_survival_data)
    assert family_survival_data["Fare"].isnull().sum() == 0


def test_update_family_survival_by_lastname_and_fare(family_survival_data):
    titanic_features.update_family_survival_by_lastname_and_fare(family_survival_data)

    # Expected values based on the input data
    expected_family_survival_values = [0.5, 0.5, 0.5, 0.5, 1.0, 0.0]
    assert (
        family_survival_data["Family_Survival"].tolist()
        == expected_family_survival_values
    )


def test_update_family_survival_by_ticket(family_survival_data):
    titanic_features.update_family_survival_by_ticket(family_survival_data)

    # Expected values based on the input data
    expected_family_survival_values = [0.5, 0.5, 0.5, 0.5, 1.0, 0.0]
    assert (
        family_survival_data["Family_Survival"].tolist()
        == expected_family_survival_values
    )


def test_family_size(family_survival_data):
    expected_family_survival = [
        0.5,
        0.5,
        0.5,
        0.5,
        1.0,
        0.0,
    ]

    df_with_family_size = titanic_features.family_size(family_survival_data)

    assert "Family_Survival" in df_with_family_size.columns
    assert df_with_family_size["Family_Survival"].tolist() == expected_family_survival


def test_feature_bins():
    # Given
    data = {
        "Age": [22, 38, 26, 35, 35, 32],
        "Fare": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.05],
    }
    df = pd.DataFrame(data)
    column_name = "Age"
    quantiles = 3

    # When
    df_with_bins = titanic_features.feature_bins(df, column_name, quantiles)

    # Then
    bin_column = column_name + "Bin"
    bin_code = bin_column + "_Code"

    assert bin_column in df_with_bins.columns
    assert bin_code in df_with_bins.columns
    assert column_name not in df_with_bins.columns

    # Check if the correct number of bins is created
    unique_bins = df_with_bins[bin_code].nunique()
    assert unique_bins == quantiles

    # Check if the bins have approximately equal number of elements
    bin_counts = df_with_bins[bin_code].value_counts().tolist()
    assert all(
        abs(bin_counts[i] - bin_counts[i + 1]) <= 1 for i in range(len(bin_counts) - 1)
    )


def test_get_train_df(sample_data_df):
    nrow = 3

    # Prepare the expected output data
    expected_data = {
        "Survived": [0, 1, 0],
        "Pclass": [1, 2, 1],
        "Sex": ["male", "female", "male"],
    }
    expected_output_df = pd.DataFrame(expected_data)

    # Call the function and get the result
    result_df = titanic_features.get_train_df(sample_data_df, nrow)

    # Compare the result with the expected output
    pd.testing.assert_frame_equal(result_df, expected_output_df)


def test_get_test_df(sample_data_df):
    nrow = 3

    # Prepare the expected output data
    expected_data = {"Pclass": [2], "Sex": ["female"]}
    expected_output_df = pd.DataFrame(expected_data)

    # Call the function and get the result
    result_df = titanic_features.get_test_df(sample_data_df, nrow)

    # Reset the index for both DataFrames
    result_df.reset_index(drop=True, inplace=True)
    expected_output_df.reset_index(drop=True, inplace=True)

    # Compare the result with the expected output
    pd.testing.assert_frame_equal(result_df, expected_output_df)

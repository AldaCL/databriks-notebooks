import pandas as pd
from lib.constants import DEFAULT_SURVIVAL_VALUE
from sklearn.preprocessing import LabelEncoder


def clean_names(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean name column, extract Title and Lastname, and impute missing Age values.
    """
    # Extract Title from Name
    data_df["Title"] = data_df["Name"].str.extract(r"([A-Za-z]+)\.", expand=True)

    # Replace rare titles with more common ones
    title_mapping = {
        "Mlle": "Miss",
        "Major": "Mr",
        "Col": "Mr",
        "Sir": "Mr",
        "Don": "Mr",
        "Mme": "Miss",
        "Jonkheer": "Mr",
        "Lady": "Mrs",
        "Capt": "Mr",
        "Countess": "Mrs",
        "Ms": "Miss",
        "Dona": "Mrs",
    }
    data_df.replace({"Title": title_mapping}, inplace=True)

    # List of common titles
    common_titles = ["Dr", "Master", "Miss", "Mr", "Mrs", "Rev"]

    # Impute missing Age values based on median Age for each Title
    for title in common_titles:
        median_age = data_df.groupby("Title")["Age"].median().loc[title]
        data_df.loc[
            (data_df["Age"].isnull()) & (data_df["Title"] == title), "Age"
        ] = median_age

    # Drop Title column as it's no longer needed
    data_df.drop("Title", axis=1, inplace=True)

    # Extract Lastname from Name
    data_df["Lastname"] = data_df["Name"].apply(lambda x: str.split(x, ",")[0])

    return data_df


def family_size(data_df: pd.DataFrame) -> pd.DataFrame:
    fill_missing_fares(data_df)
    data_df["Family_Survival"] = DEFAULT_SURVIVAL_VALUE
    update_family_survival_by_lastname_and_fare(data_df)
    update_family_survival_by_ticket(data_df)
    return data_df


def fill_missing_fares(data_df: pd.DataFrame):
    """Fills the missing fares with the mean fare."""
    data_df["Fare"].fillna(data_df["Fare"].mean(), inplace=True)


def update_family_survival_by_lastname_and_fare(data_df: pd.DataFrame) -> None:
    """Updates the 'Family_Survival' column based on the groupings of last name and fare"""
    for _, grp_df in data_df[
        [
            "Survived",
            "Name",
            "Lastname",
            "Fare",
            "Ticket",
            "PassengerId",
            "SibSp",
            "Parch",
            "Age",
            "Cabin",
        ]
    ].groupby(["Lastname", "Fare"]):
        if len(grp_df) > 1:
            for index, row in grp_df.iterrows():
                max_survived = grp_df.drop(index)["Survived"].max()
                min_survived = grp_df.drop(index)["Survived"].min()
                passenger_id = row["PassengerId"]
                if max_survived == 1.0:
                    data_df.loc[
                        data_df["PassengerId"] == passenger_id, "Family_Survival"
                    ] = 1
                elif min_survived == 0.0:
                    data_df.loc[
                        data_df["PassengerId"] == passenger_id, "Family_Survival"
                    ] = 0


def update_family_survival_by_ticket(data_df: pd.DataFrame) -> None:
    """Updates the 'Family_Survival' column based on the groupings of tickets"""
    for _, grp_df in data_df.groupby("Ticket"):
        if len(grp_df) > 1:
            for index, row in grp_df.iterrows():
                if (row["Family_Survival"] == 0) or (row["Family_Survival"] == 0.5):
                    max_survived = grp_df.drop(index)["Survived"].max()
                    min_survived = grp_df.drop(index)["Survived"].min()
                    passenger_id = row["PassengerId"]
                    if max_survived == 1.0:
                        data_df.loc[
                            data_df["PassengerId"] == passenger_id, "Family_Survival"
                        ] = 1
                    elif min_survived == 0.0:
                        data_df.loc[
                            data_df["PassengerId"] == passenger_id, "Family_Survival"
                        ] = 0


def feature_bins(
    data_df: pd.DataFrame, column_name: str, quantiles: int
) -> pd.DataFrame:
    # Create names for the new columns
    bin_range_column = column_name + "Bin"
    bin_code_column = bin_range_column + "_Code"

    # Create the bin ranges using the specified quantiles
    data_df[bin_range_column] = pd.qcut(data_df[column_name], quantiles)

    # Encode the bin ranges as integer codes using LabelEncoder
    label_encoder = LabelEncoder()
    data_df[bin_code_column] = label_encoder.fit_transform(data_df[bin_range_column])

    # Drop the original column from the DataFrame
    data_df.drop(columns=column_name, inplace=True)

    return data_df


def get_train_df(data_df: pd.DataFrame, nrow: int) -> pd.DataFrame:
    """
    Get training dataframe
    """
    train_df = data_df.copy().iloc[:nrow, :]
    train_df.drop(
        columns=[
            "Name",
            "PassengerId",
            "SibSp",
            "Parch",
            "Ticket",
            "Cabin",
            "Embarked",
            "Lastname",
            "FareBin",
            "AgeBin",
        ],
        inplace=True,
    )
    return train_df


def get_test_df(data_df: pd.DataFrame, nrow: int) -> pd.DataFrame:
    """
    Get test dataframe
    """
    test_df = data_df.copy().iloc[nrow:, :]
    test_df.drop(
        columns=[
            "Name",
            "PassengerId",
            "SibSp",
            "Parch",
            "Ticket",
            "Cabin",
            "Embarked",
            "Lastname",
            "FareBin",
            "Survived",
            "AgeBin",
        ],
        inplace=True,
    )

    return test_df

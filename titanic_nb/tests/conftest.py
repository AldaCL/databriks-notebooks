import pandas as pd
import pytest
from lib.constants import DEFAULT_SURVIVAL_VALUE


@pytest.fixture
def raw_data():
    data = {
        "Name": [
            "Braund, Mr. Owen Harris",
            "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
            "Heikkinen, Miss. Laina",
            "Futrelle, Mrs. Jacques Heath (Lily May Peel)",
            "Allen, Mr. William Henry",
            "Green, Dr. John",
            "Brown, Master. James",
            "Smith, Rev. Adam",
        ],
        "Age": [22, 38, 26, 35, 35, 45, 12, 40],
    }
    return pd.DataFrame(data)


@pytest.fixture
def family_survival_data():
    data = {
        "Survived": [0, 1, 1, 0, 0, 1],
        "Name": [
            "Braund, Mr. Owen Harris",
            "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
            "Heikkinen, Miss. Laina",
            "Futrelle, Mrs. Jacques Heath (Lily May Peel)",
            "Allen, Mr. William Henry",
            "Allen, Mrs. Jane",
        ],
        "Lastname": ["Braund", "Cumings", "Heikkinen", "Futrelle", "Allen", "Allen"],
        "Fare": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.05],
        "Ticket": [
            "A/5 21171",
            "PC 17599",
            "STON/O2. 3101282",
            "113803",
            "373450",
            "373450",
        ],
        "PassengerId": [1, 2, 3, 4, 5, 6],
        "SibSp": [1, 1, 0, 1, 0, 1],
        "Parch": [0, 0, 0, 0, 0, 0],
        "Age": [22, 38, 26, 35, 35, 32],
        "Cabin": [None, "C85", None, "C123", None, None],
        "Family_Survival": DEFAULT_SURVIVAL_VALUE,
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_data_df():
    data = {
        "Name": ["A", "B", "C", "D"],
        "PassengerId": [1, 2, 3, 4],
        "SibSp": [1, 0, 1, 0],
        "Parch": [0, 1, 0, 1],
        "Ticket": ["A1", "B2", "C3", "D4"],
        "Cabin": [None, "C1", None, "C2"],
        "Embarked": ["S", "C", "S", "C"],
        "Lastname": ["LN1", "LN2", "LN3", "LN4"],
        "FareBin": [1, 2, 3, 4],
        "AgeBin": [1, 1, 2, 2],
        "Survived": [0, 1, 0, 1],
        "Pclass": [1, 2, 1, 2],
        "Sex": ["male", "female", "male", "female"],
    }
    return pd.DataFrame(data)

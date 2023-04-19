import pandas as pd
from sklearn.preprocessing import LabelEncoder


def clean_names(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning name and extracting Title
    """
    data_df['Title'] = ''
    # Replacing rare titles with more common ones
    mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr',
               'Don': 'Mr', 'Mme': 'Miss', 'Jonkheer': 'Mr', 'Lady': 'Mrs',
               'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}

    # Cleaning name and extracting Title
    for name_string in data_df['Name']:
        data_df['Title'] = data_df['Name'].str.extract('([A-Za-z]+)\.', expand=True)

    data_df.replace({'Title': mapping}, inplace=True)

    titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']
    
    for title in titles:
        age_to_impute = data_df.groupby('Title')['Age'].median()[titles.index(title)]
        data_df.loc[(data_df['Age'].isnull()) & (data_df['Title'] == title), 'Age'] = age_to_impute

    data_df.drop('Title', axis=1, inplace=True)
    data_df['Lastname'] = data_df['Name'].apply(lambda x: str.split(x, ",")[0])

    return data_df


def family_size(data_df: pd.DataFrame) -> pd.DataFrame:
    """ TODO: Este metodo esta bien chato, no me gusta pero no se me ocurre como refactorizarlo ni testearlo
    family size"""

    DEFAULT_SURVIVAL_VALUE = 0.5
    data_df['Fare'].fillna(data_df['Fare'].mean(), inplace=True)

    data_df['Family_Survival'] = DEFAULT_SURVIVAL_VALUE

    for grp, grp_df in data_df[['Survived', 'Name', 'Lastname', 'Fare', 'Ticket', 'PassengerId',
                            'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Lastname', 'Fare']):

        if (len(grp_df) != 1):
            # A Family group is found.
            for ind, row in grp_df.iterrows():
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin==0.0):
                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0

    print("Number of passengers with family survival information:",
        data_df.loc[data_df['Family_Survival'] != 0.5].shape[0])

    for _, grp_df in data_df.groupby('Ticket'):
        if len(grp_df) != 1:
            for ind, row in grp_df.iterrows():
                if (row['Family_Survival'] == 0) | (row['Family_Survival'] == 0.5):
                    smax = grp_df.drop(ind)['Survived'].max()
                    smin = grp_df.drop(ind)['Survived'].min()
                    passID = row['PassengerId']
                    if smax == 1.0:
                        data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
                    elif smin == 0.0:
                        data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0
                            
    print("Number of passenger with family/group survival information: " +
          str(data_df[data_df['Family_Survival']!=0.5].shape[0]))

    return data_df


def feature_bins(data_df: pd.DataFrame,
                 column_name: str,
                 quantiles: int) -> pd.DataFrame:

    bin_column = column_name + 'Bin'
    bin_code = bin_column+'_Code'

    data_df[bin_column] = pd.qcut(data_df[column_name], quantiles)

    label = LabelEncoder()
    data_df[bin_code] = label.fit_transform(data_df[bin_column])

    data_df.drop(columns=column_name, inplace=True)

    return data_df


def get_train_df(df_data: pd.DataFrame,
                 nrow: int) -> pd.DataFrame:
    """
    Get training dataframe
    """
    train_df = df_data.copy().iloc[:nrow, :]
    train_df.drop(columns=['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket',
                           'Cabin', 'Embarked', 'Lastname', 'FareBin', 'AgeBin'], inplace=True)
    return train_df


def get_test_df(df_data: pd.DataFrame, nrow: int) -> pd.DataFrame:
    """
    Get test dataframe
    """
    test_df = df_data.copy().iloc[nrow:, :]
    test_df.drop(columns=['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket',
                          'Cabin', 'Embarked', 'Lastname', 'FareBin', 'Survived', 'AgeBin'], inplace=True)

    return test_df

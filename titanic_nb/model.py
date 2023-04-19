# Databricks notebook source
import pandas as pd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


gender_df = pd.read_csv("datasource/gender_submission.csv")
test_df = pd.read_csv("datasource/test.csv")
train_df = pd.read_csv("datasource/train.csv")
data_df = pd.concat([train_df, test_df], sort=False)

data_df['Title'] = ''

# Cleaning name and extracting Title
for name_string in data_df['Name']:
    data_df['Title'] = data_df['Name'].str.extract('([A-Za-z]+)\.', expand=True)

# Replacing rare titles with more common ones
mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}

data_df.replace({'Title': mapping}, inplace=True)
titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']

for title in titles:
    age_to_impute = data_df.groupby('Title')['Age'].median()[titles.index(title)]
    data_df.loc[(data_df['Age'].isnull()) & (data_df['Title'] == title), 'Age'] = age_to_impute
    
# Substituting Age values in TRAIN_DF and TEST_DF:
train_df['Age'] = data_df['Age'][:891]
test_df['Age'] = data_df['Age'][891:]

# Dropping Title feature
data_df.drop('Title', axis = 1, inplace = True)
data_df.Age.isna().sum()

data_df['Family_Size'] = data_df['Parch'] + data_df['SibSp']

# Substituting Age values in TRAIN_DF and TEST_DF:
train_df['Family_Size'] = data_df['Family_Size'][:891]
test_df['Family_Size'] = data_df['Family_Size'][891:]


data_df['Lastname'] = data_df['Name'].apply(lambda x: str.split(x, ",")[0])
data_df['Fare'].fillna(data_df['Fare'].mean(), inplace=True)

DEFAULT_SURVIVAL_VALUE = 0.5
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
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin==0.0):
                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0
                        
print("Number of passenger with family/group survival information: " 
      +str(data_df[data_df['Family_Survival']!=0.5].shape[0]))

# # Family_Survival in TRAIN_DF and TEST_DF:
train_df['Family_Survival'] = data_df['Family_Survival'][:891]
test_df['Family_Survival'] = data_df['Family_Survival'][891:]

data_df['Fare'].fillna(data_df['Fare'].median(), inplace = True)

# Making Bins
data_df['FareBin'] = pd.qcut(data_df['Fare'], 5)

label = LabelEncoder()
data_df['FareBin_Code'] = label.fit_transform(data_df['FareBin'])

train_df['FareBin_Code'] = data_df['FareBin_Code'][:891]
test_df['FareBin_Code'] = data_df['FareBin_Code'][891:]

train_df.drop(columns='Fare', inplace=True)
test_df.drop(columns='Fare', inplace=True)

data_df['AgeBin'] = pd.qcut(data_df['Age'], 4)

label = LabelEncoder()
data_df['AgeBin_Code'] = label.fit_transform(data_df['AgeBin'])

train_df['AgeBin_Code'] = data_df['AgeBin_Code'][:891]
test_df['AgeBin_Code'] = data_df['AgeBin_Code'][891:]

train_df.drop(columns='Age', inplace=True)
test_df.drop(columns='Age', inplace=True)

data_df.head()

train_df['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
test_df['Sex'].replace(['male', 'female'], [0, 1], inplace=True)

train_df.drop(columns=['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket',
                       'Cabin', 'Embarked'], inplace=True)
test_df.drop(columns=['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket',
                      'Cabin', 'Embarked'], inplace=True)

X = train_df.drop(columns='Survived')
y = train_df['Survived']
X_test = test_df.copy()

std_scaler = StandardScaler()
X = std_scaler.fit_transform(X)
X_test = std_scaler.transform(X_test)

pd.DataFrame(X).head()

n_neighbors = [6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22]
algorithm = ['auto']
weights = ['uniform', 'distance']
leaf_size = list(range(1, 50, 5))

hyperparams = {'algorithm': algorithm, 'weights': weights,
               'leaf_size': leaf_size, 'n_neighbors': n_neighbors}

gd = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=hyperparams,
                  verbose=True, cv=10, scoring="roc_auc")

gd.fit(X, y)
print(gd.best_score_)
print(gd.best_params_)

gd.best_estimator_.fit(X, y)
y_pred = gd.best_estimator_.predict(X_test)

pd.DataFrame(y_pred).head()

knn_1 = KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski',
                             metric_params=None, n_jobs=1, n_neighbors=22, p=2,
                             weights='uniform')

knn_2 = KNeighborsClassifier(algorithm='auto', leaf_size=16, metric='minkowski',
                             metric_params=None, n_jobs=1, n_neighbors=6, p=2, 
                             weights='uniform')

train, val = train_test_split(train_df, test_size=0.3, random_state=0,
                              stratify=train_df['Survived'])

train_X = train[train.columns[1:]]
train_y = train[train.columns[:1]]
test_X = val[val.columns[1:]]
test_y = val[val.columns[:1]]

knn_1.fit(train_X, train_y.values.ravel())
knn_2.fit(train_X, train_y.values.ravel())

y_pred_1 = knn_1.predict(test_X)
y_pred_2 = knn_2.predict(test_X)

# Perform PCA on data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(test_X)

# Create a dataframe with PCA components and predicted labels
pca_df_1 = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df_1['y_pred'] = y_pred_1
pca_df_2 = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df_2['y_pred'] = y_pred_2

# Plot clusters using seaborn
f, ax = plt.subplots(1, 2, figsize=(18,8))

sns.scatterplot(data=pca_df_1, x='PC1', y='PC2', hue='y_pred', ax=ax[0])
sns.scatterplot(data=pca_df_2, x='PC1', y='PC2', hue='y_pred', ax=ax[1])
plt.show()
plt.savefig('knn.png')
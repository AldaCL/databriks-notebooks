# Databricks notebook source
import pandas as pd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# data processing
from data_processing import load_data
from data_processing import features
from data_processing import ploting


# Loads datasources
test_df = load_data.get_dataframe("test")
train_df = load_data.get_dataframe("train")

# Generate a unique dataframe to clean and process data
data_df = pd.concat([train_df, test_df], sort=False)

# Generates feature engineering on names
data_df = features.clean_names(data_df)

# Prepare and generates feature engineering on Family_Size
data_df['Family_Size'] = data_df['Parch'] + data_df['SibSp']
data_df = features.family_size(data_df)

# Prepare and generates feature engineering on Fare
data_df['Fare'].fillna(data_df['Fare'].median(), inplace=True)
data_df = features.feature_bins(data_df, 'Fare', 5)

# Generate feature engineering on Age
data_df = features.feature_bins(data_df, 'Age', 4)

# Change age to numeric values
data_df = data_df.replace(['male', 'female'], [0, 1], inplace=True)

# Generate training and testing dataFrames
train_df = data_df.copy().iloc[:891, :]
test_df = data_df.copy().iloc[891:, :]

# Drop columns to 
train_df.drop(columns=['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket',
                       'Cabin', 'Embarked', 'Lastname', 'FareBin'], inplace=True)
test_df.drop(columns=['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket',
                      'Cabin', 'Embarked', 'Lastname', 'FareBin', 'Survived'], inplace=True)


X = train_df.drop(columns='Survived')
y = train_df['Survived']
X_test = test_df.copy()

std_scaler = StandardScaler()
X = std_scaler.fit_transform(X)
X_test = std_scaler.transform(X_test)

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

# pd.DataFrame(y_pred).head()

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
f, ax = plt.subplots(1, 2, figsize=(18, 8))

sns.scatterplot(data=pca_df_1, x='PC1', y='PC2', hue='y_pred', ax=ax[0])
sns.scatterplot(data=pca_df_2, x='PC1', y='PC2', hue='y_pred', ax=ax[1])
plt.show()
plt.savefig('knn.png')
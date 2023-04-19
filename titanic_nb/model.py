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
from data_processing import processing
from data_processing import ploting


# Loads datasources
test_df = load_data.get_dataframe("test")
train_df = load_data.get_dataframe("train")

# Generate a unique dataframe to clean and process data
data_df = pd.concat([train_df, test_df], sort=False)

# Generates feature engineering on names
data_df = features.clean_names(data_df=data_df)
# data_df = features.normalize_titles(data_df)

# Prepare and generates feature engineering on Family_Size
data_df['Family_Size'] = data_df['Parch'] + data_df['SibSp']

data_df = features.family_size(data_df=data_df)

# Prepare and generates feature engineering on Fare
data_df['Fare'].fillna(data_df['Fare'].median(), inplace=True)

data_df = features.feature_bins(data_df=data_df,
                                column_name='Fare',
                                quantiles=5)

# Generate feature engineering on Age
data_df = features.feature_bins(data_df=data_df,
                                column_name='Age',
                                quantiles=4)

# Change age to numeric values
data_df.replace(['male', 'female'], [0, 1], inplace=True)

# Generate training and testing dataFrames
train_df = features.get_train_df(data_df=data_df, nrow=891)
test_df = features.get_test_df(data_df=data_df, nrow=891)

X = train_df.drop(columns='Survived')
y = train_df['Survived']
X_test = test_df.copy()

# Generate scalers in data
std_scaler = StandardScaler()
X = std_scaler.fit_transform(X)
X_test = std_scaler.transform(X_test)

# Fit grid search model
gd = processing.fit_grid_search(X, y)

print(gd.best_score_)
print(gd.best_params_)

gd.best_estimator_.fit(X, y)
y_pred = gd.best_estimator_.predict(X_test)

# Training and validation data for predictions
train, val = train_test_split(train_df,
                              test_size=0.3,
                              random_state=0,
                              stratify=train_df['Survived'])

y_pred_1 = processing.get_predictions(leaf_size=26,
                                      n_neighbors=22,
                                      train=train,
                                      val=val)

test_X = val[val.columns[1:]]
test_y = val[val.columns[:1]]

# Perform PCA on data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(test_X)

# Create a dataframe with PCA components and predicted labels to plot
pca_df_1 = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df_1['y_pred'] = y_pred_1

# Plot clusters using seaborn
f, ax = plt.plot(figsize=(18, 8))
sns.scatterplot(data=pca_df_1, x='PC1', y='PC2', hue='y_pred')
plt.show()
plt.savefig('knn_classified.png')

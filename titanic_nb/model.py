# Databricks notebook source
import numpy as np
import pandas as pd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from collections import Counter


# COMMAND ----------

# MAGIC %md
# MAGIC ## 💾 Loading Data
# MAGIC 
# MAGIC This code block reads in the training and test data for the Titanic sinking competition from CSV files located in the input directory. It uses Pandas' read_csv() function to read in the data and store it as a DataFrame.
# MAGIC 
# MAGIC The train_df DataFrame contains the training data, which includes features such as passenger class, age, and sex, as well as the target variable of survival status. The test_df DataFrame contains the test data, which includes the same features as the training data, but the target variable of survival status is not included.
# MAGIC 
# MAGIC The last line of code appends the test data to the training data to create a single DataFrame called data_df. This is often done in data preprocessing to allow for consistent feature engineering and transformation across the entire dataset

# COMMAND ----------

gender_df = pd.read_csv("/dbfs/FileStore/shared_uploads/aldair.alda27@gmail.com/gender_submission.csv")
test_df = pd.read_csv("/dbfs/FileStore/shared_uploads/aldair.alda27@gmail.com/test.csv")
train_df = pd.read_csv("/dbfs/FileStore/shared_uploads/aldair.alda27@gmail.com/train.csv")

data_df = train_df.append(test_df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## ⚙️ Feature Engineering
# MAGIC We are directly diving into feature engineering in this notebook as my Titanic's EDA can be found here.
# MAGIC 
# MAGIC ### Imputing Age
# MAGIC This code block performs data preprocessing steps to extract and clean the Title feature from the Name feature of the data_df DataFrame.
# MAGIC 
# MAGIC First, a new empty column called Title is added to the DataFrame using data_df['Title'] = ''.
# MAGIC 
# MAGIC Next, a for loop is used to iterate through each name string in the Name feature of the DataFrame. The str.extract() method is then used to extract the title from the name string using a regular expression. The resulting title is stored in the Title column of the corresponding row in the DataFrame.
# MAGIC 
# MAGIC The code then replaces rare titles with more common ones using a predefined mapping dictionary. The titles included in the mapping are 'Mlle', 'Major', 'Col', 'Sir', 'Don', 'Mme', 'Jonkheer', 'Lady', 'Capt', 'Countess', 'Ms', and 'Dona'.
# MAGIC 
# MAGIC After mapping the titles, the code calculates the median age for each title group and imputes missing age values in the DataFrame based on the title of the corresponding passenger.
# MAGIC 
# MAGIC Finally, the code substitutes the imputed age values back into the original training and test DataFrames using indexing. The Title feature is then dropped from the data_df DataFrame using the drop() method.
# MAGIC 
# MAGIC Overall, this code block performs an important step in feature engineering by extracting meaningful information from the Name feature and using it to impute missing age values in the dataset.

# COMMAND ----------

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

# COMMAND ----------

data_df.Age.isna().sum()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Family Size
# MAGIC This code block calculates the Family_Size feature by adding the Parch (number of parents or children aboard) and SibSp (number of siblings or spouses aboard) features together for each passenger in the data_df DataFrame. The resulting values are stored in a new column called Family_Size.
# MAGIC 
# MAGIC The code then substitutes the Family_Size values back into the original training and test DataFrames using indexing. This feature could be used to infer information about the passenger's social status or support network aboard the Titanic, which could be useful in predicting survival status.

# COMMAND ----------

data_df['Family_Size'] = data_df['Parch'] + data_df['SibSp']

# Substituting Age values in TRAIN_DF and TEST_DF:
train_df['Family_Size'] = data_df['Family_Size'][:891]
test_df['Family_Size'] = data_df['Family_Size'][891:]

# COMMAND ----------

# MAGIC %md
# MAGIC Family Survival Status
# MAGIC This block of code creates a new feature called Family_Survival that attempts to infer the survival status of a passenger's family based on their shared ticket number, last name, and fare. The code first creates a new column called Lastname by extracting the last name from the Name column using the lambda function and the split() method.
# MAGIC 
# MAGIC The missing Fare values are then replaced with the mean fare value using the fillna() method. The default survival value is set to 0.5, which is used to initialize the Family_Survival column.
# MAGIC 
# MAGIC Next, the code iterates over each family group (identified by shared Last_Name and Fare values) in the data_df DataFrame using the groupby() method. For each family group, the code checks whether any family member has survived or perished. If at least one family member has survived, then the Family_Survival value for all family members in the group is set to 1. If all family members have perished, then the Family_Survival value is set to 0 for all members. If the group only has one member, the Family_Survival value remains at its default value of 0.5.
# MAGIC 
# MAGIC The last line of code prints the number of passengers for which the Family_Survival value is not 0.5, indicating that family survival information is available for those passengers.

# COMMAND ----------

data_df['Lastname'] = data_df['Name'].apply(lambda x: str.split(x, ",")[0])
data_df['Fare'].fillna(data_df['Fare'].mean(), inplace=True)

DEFAULT_SURVIVAL_VALUE = 0.5
data_df['Family_Survival'] = DEFAULT_SURVIVAL_VALUE

for grp, grp_df in data_df[['Survived','Name', 'Lastname', 'Fare', 'Ticket', 'PassengerId',
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
      data_df.loc[data_df['Family_Survival']!=0.5].shape[0])

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fare
# MAGIC The code above prepares the Fare feature by filling missing values with the median, binning the feature using pd.qcut, and encoding the bin labels with a LabelEncoder. It then assigns the bin codes to the corresponding rows in the train_df and test_df dataframes, and drops the original Fare column from both dataframes.
# MAGIC 
# MAGIC The pd.qcut function divides the range of values in the Fare column into 5 bins with equal number of observations. This is done to reduce the noise in the data and to capture the patterns in the data that may not be apparent when looking at the raw values.
# MAGIC 
# MAGIC The LabelEncoder is used to transform the categorical bin labels into numerical values that can be used as input for the machine learning models. The encoded values range from 0 to 4, corresponding to the 5 bins generated by pd.qcut.

# COMMAND ----------

data_df['Fare'].fillna(data_df['Fare'].median(), inplace = True)

# Making Bins
data_df['FareBin'] = pd.qcut(data_df['Fare'], 5)

label = LabelEncoder()
data_df['FareBin_Code'] = label.fit_transform(data_df['FareBin'])

train_df['FareBin_Code'] = data_df['FareBin_Code'][:891]
test_df['FareBin_Code'] = data_df['FareBin_Code'][891:]

train_df.drop(columns='Fare', inplace=True)
test_df.drop(columns='Fare', inplace=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Age
# MAGIC Here, we are performing a binning of the Age feature into 4 bins using the qcut method of Pandas, and then labeling the bins using LabelEncoder. The resulting AgeBin_Code column is then added to the data_df DataFrame. Finally, the AgeBin_Code column is extracted for the training and testing datasets, and the original Age column is dropped from both.

# COMMAND ----------

data_df['AgeBin'] = pd.qcut(data_df['Age'], 4)

label = LabelEncoder()
data_df['AgeBin_Code'] = label.fit_transform(data_df['AgeBin'])

train_df['AgeBin_Code'] = data_df['AgeBin_Code'][:891]
test_df['AgeBin_Code'] = data_df['AgeBin_Code'][891:]

train_df.drop(columns='Age', inplace=True)
test_df.drop(columns='Age', inplace=True)

# COMMAND ----------

data_df.head()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Sex
# MAGIC These lines of code replace the male and female values in the Sex column of the train_df and test_df dataframes with 0 and 1, respectively. This is done to convert the categorical Sex feature into a numerical one so that machine learning models can be trained on the data.

# COMMAND ----------

train_df['Sex'].replace(['male','female'],[0,1],inplace=True)
test_df['Sex'].replace(['male','female'],[0,1],inplace=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean unused data
# MAGIC This code drops several columns from the train_df and test_df dataframes that are not needed for training or testing the model. Specifically, the columns being dropped are:
# MAGIC 
# MAGIC Name : Names of passengers
# MAGIC PassengerId : Unique IDs of passengers
# MAGIC SibSp : Number of siblings/spouses aboard the Titanic
# MAGIC Parch : Number of parents/children aboard the Titanic
# MAGIC Ticket : Ticket number of passengers
# MAGIC Cabin : Cabin number of passengers
# MAGIC Embarked : Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
# MAGIC These columns have been dropped as they are not expected to contribute much to the prediction of the survival of passengers, and including them in the model could potentially add unnecessary noise to the training process.

# COMMAND ----------

train_df.drop(columns=['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin',
               'Embarked'], inplace = True)
test_df.drop(columns=['Name','PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin',
              'Embarked'], inplace = True)

# COMMAND ----------

# MAGIC %md
# MAGIC ##🤖 Predictive Modeling
# MAGIC As I already tried other predictive modeling algorithms here, I will only focus on the K-Nearest Neighbors classifier algorithm in this notebook.
# MAGIC 
# MAGIC Training
# MAGIC Let's start by the training phase of our predictive model.
# MAGIC 
# MAGIC Creating X and y
# MAGIC The code below is splitting the training dataset train_df into two sets - X and y.
# MAGIC 
# MAGIC X contains all the features (columns) of the training dataset except for the target variable, Survived.
# MAGIC 
# MAGIC y is the target variable, Survived, and is a one-dimensional array or series that contains the corresponding class labels for each row in X.
# MAGIC 
# MAGIC The test_df dataset is also copied to X_test so that it can be used for predictions later on.

# COMMAND ----------

X = train_df.drop(columns='Survived')
y = train_df['Survived']
X_test = test_df.copy()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scaling features
# MAGIC The code standardizes the numerical features in the training and test datasets using StandardScaler() from the scikit-learn library. Standardization involves transforming the data so that it has a mean of 0 and a standard deviation of 1.
# MAGIC 
# MAGIC The fit_transform() method of the scaler is called on the training set X to compute the mean and standard deviation for each feature and then transform the data. The transform() method is then called on the test set X_test to apply the same transformation, but using the mean and standard deviation computed on the training set.
# MAGIC 
# MAGIC Standardizing the data can help improve the performance of some machine learning models, particularly those that are sensitive to the scale of the input features, such as logistic regression or support vector machines.

# COMMAND ----------

std_scaler = StandardScaler()
X = std_scaler.fit_transform(X)
X_test = std_scaler.transform(X_test)

# COMMAND ----------

pd.DataFrame(X).head()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Hyperparameters tuning
# MAGIC This code is performing a grid search using cross-validation to find the best hyperparameters for a K-Nearest Neighbors classifier. The hyperparameters being searched over are:
# MAGIC 
# MAGIC algorithm : the algorithm used to compute the nearest neighbors
# MAGIC weights : the weighting function used in prediction
# MAGIC leaf_size : the size of the leaf in the KD tree
# MAGIC n_neighbors : the number of neighbors to use for classification
# MAGIC The grid search is performed using 10-fold cross-validation, with the roc_auc scoring metric. Once the grid search is complete, the best score and best hyperparameters are printed out.

# COMMAND ----------

n_neighbors = [6,7,8,9,10,11,12,14,16,18,20,22]
algorithm = ['auto']
weights = ['uniform', 'distance']
leaf_size = list(range(1,50,5))

hyperparams = {'algorithm': algorithm, 'weights': weights, 'leaf_size': leaf_size, 
               'n_neighbors': n_neighbors}

gd=GridSearchCV(estimator = KNeighborsClassifier(), param_grid = hyperparams, verbose=True, 
                cv=10, scoring = "roc_auc")

gd.fit(X, y)
print(gd.best_score_)
print(gd.best_params_)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predicting
# MAGIC Let's see how well our KNN model performs on the public test dataset.
# MAGIC 
# MAGIC ### Model found by Grid Searching
# MAGIC This code fits the best estimator found by the grid search cross-validation on the training data (X and y). Then, it uses the fitted model to make predictions on the test data X_test and stores the predictions in y_pred. The y_pred variable contains the predicted binary labels for the survival of each passenger in the test set.

# COMMAND ----------

gd.best_estimator_.fit(X, y)
y_pred = gd.best_estimator_.predict(X_test)

# COMMAND ----------

pd.DataFrame(y_pred).head()


# COMMAND ----------

knn_1 = KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski',
                            metric_params=None, n_jobs=1, n_neighbors=22, p=2,
                            weights='uniform')

knn_2 = KNeighborsClassifier(algorithm='auto', leaf_size=16, metric='minkowski', 
                           metric_params=None, n_jobs=1, n_neighbors=6, p=2, 
                           weights='uniform')

# COMMAND ----------

train, val = train_test_split(train_df, test_size=0.3, random_state=0, stratify=train_df['Survived'])
train_X = train[train.columns[1:]]
train_y = train[train.columns[:1]]
test_X = val[val.columns[1:]]
test_y = val[val.columns[:1]]

# COMMAND ----------

knn_1.fit(train_X, train_y.values.ravel())
knn_2.fit(train_X, train_y.values.ravel())

# COMMAND ----------

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

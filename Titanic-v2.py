"""
Using Minsuk Heo's approach
https://www.youtube.com/watch?v=COUWKVf6zKY&t=12s
"""

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Importing Classifier Modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

sns.set()

# Collecting Data
train_data = pd.read_csv("train.csv")
train_data.head()
test_data = pd.read_csv("test.csv")
test_data.head()
y = train_data["Survived"]

total_dataset = [ train_data, test_data ]


for row in total_dataset:
    row["Title"] = row["Name"].str.extract(' ([a-zA-Z]+)\.', expand=False)

title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

for dataset in total_dataset:
    dataset['Title'] = dataset['Title'].map(title_mapping)

# delete unnecessary feature from dataset
train_data.drop('Name', axis=1, inplace=True)
test_data.drop('Name', axis=1, inplace=True)

sex_mapping = {"male": 0, "female": 1}
for dataset in total_dataset:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

# fill missing age with median age for each title (Mr, Mrs, Miss, Others)
train_data["Age"].fillna(train_data.groupby("Title")["Age"].transform("median"), inplace=True)
test_data["Age"].fillna(test_data.groupby("Title")["Age"].transform("median"), inplace=True)
train_data.groupby("Title")["Age"].transform("median")


#Binning values decided based on data analysis
for dataset in total_dataset:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4

#Embarked
for dataset in total_dataset:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in total_dataset:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

# fill missing Fare with median fare for each Pclass
train_data["Fare"].fillna(train_data.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test_data["Fare"].fillna(test_data.groupby("Pclass")["Fare"].transform("median"), inplace=True)

for dataset in total_dataset:
    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3

for dataset in total_dataset:
    dataset['Cabin'] = dataset['Cabin'].str[:1]


cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in total_dataset:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)

# fill missing Fare with median fare for each Pclass
train_data["Cabin"].fillna(train_data.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test_data["Cabin"].fillna(test_data.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

#Family Size
train_data["FamilySize"] = train_data["SibSp"] + train_data["Parch"] + 1
test_data["FamilySize"] = test_data["SibSp"] + test_data["Parch"] + 1

family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in total_dataset:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)
"""

for dataset in total_dataset:
    dataset.loc[ dataset['FamilySize'] == 1, 'FamilySize'] = 0,
    dataset.loc[ (2 <= dataset['FamilySize']) & (dataset['FamilySize'] <= 4) , 'FamilySize'] = 1,
    dataset.loc[ (5 <= dataset['FamilySize']) & (dataset['FamilySize'] <= 8) , 'FamilySize'] = 2,
    dataset.loc[ (9 <= dataset['FamilySize']) , 'FamilySize'] = 3,
"""
features_drop = ['Ticket', 'SibSp', 'Parch']
train_data = train_data.drop(features_drop, axis=1)
test_data = test_data.drop(features_drop, axis=1)
train_data = train_data.drop(['PassengerId'], axis=1)

target = train_data['Survived']
train_data = train_data.drop('Survived', axis=1)

#kNN
clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(round(np.mean(score)*100, 2))

#Decision Tree
clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(round(np.mean(score)*100, 2))


#Random Forest
clf = RandomForestClassifier(n_estimators=13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(round(np.mean(score)*100, 2))


#Naive Bayes
clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(round(np.mean(score)*100, 2))


#SVM
clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(round(np.mean(score)*100,2))

clf = SVC()
clf.fit(train_data, target)

test = test_data.drop("PassengerId", axis=1).copy()
prediction = clf.predict(test)

submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": prediction
    })

submission.to_csv('submission.csv', index=False)
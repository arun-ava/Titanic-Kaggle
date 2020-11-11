import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

# Collecting Data
train_data = pd.read_csv("train.csv")
train_data.head()
test_data = pd.read_csv("test.csv")
test_data.head()
y = train_data["Survived"]



#Analyzing Traing Data
"""
print('head', train_data.head)
print('shape', train_data.shape)
print('info', train_data.info)
print('null count', train_data.isnull().sum())
"""


def bar_chart(feature, dataset):
    survived = dataset[dataset["Survived"] == 1][feature].value_counts();
    dead = dataset[dataset["Survived"] == 0][feature].value_counts();
    df = pd.DataFrame([survived, dead])
    df.index = ["Survived", "Dead"]
    df.plot(kind = "bar", stacked = True, figsize = (10, 5))
    plt.show()
"""
bar_chart('Sex', train_data);
bar_chart('Pclass', train_data);
bar_chart('SibSp', train_data);
bar_chart('Parch', train_data);
bar_chart('Embarked', train_data);
"""


total_dataset = [ train_data, test_data ]


for row in total_dataset:
    row["Title"] = row["Name"].str.extract(' ([a-zA-Z]+)\.', expand=False)

title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

for dataset in total_dataset:
    dataset['Title'] = dataset['Title'].map(title_mapping)

#bar_chart('Title', train_data);

# delete unnecessary feature from dataset
train_data.drop('Name', axis=1, inplace=True)
test_data.drop('Name', axis=1, inplace=True)

sex_mapping = {"male": 0, "female": 1}
for dataset in total_dataset:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
#bar_chart('Sex', train_data)

# fill missing age with median age for each title (Mr, Mrs, Miss, Others)
train_data["Age"].fillna(train_data.groupby("Title")["Age"].transform("median"), inplace=True)
test_data["Age"].fillna(test_data.groupby("Title")["Age"].transform("median"), inplace=True)
train_data.groupby("Title")["Age"].transform("median")

facet = sns.FacetGrid(train_data, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train_data['Age'].max()))
facet.add_legend()
plt.show()


facet = sns.FacetGrid(train_data, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train_data['Age'].max()))
facet.add_legend()
plt.xlim(0, 20)
plt.show()

facet = sns.FacetGrid(train_data, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train_data['Age'].max()))
facet.add_legend()
plt.xlim(20, 30)
plt.show()

facet = sns.FacetGrid(train_data, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train_data['Age'].max()))
facet.add_legend()
plt.xlim(30, 40)
plt.show()

facet = sns.FacetGrid(train_data, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train_data['Age'].max()))
facet.add_legend()
plt.xlim(40, 60)
plt.show()

facet = sns.FacetGrid(train_data, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train_data['Age'].max()))
facet.add_legend()
plt.xlim(40, 60)
plt.show()

facet = sns.FacetGrid(train_data, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train_data['Age'].max()))
facet.add_legend()
plt.xlim(60)
plt.show()

#Binning values decided based on data analysis
for dataset in total_dataset:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4
bar_chart('Age', train_data)

#Embarked
Pclass1 = train_data[train_data['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train_data[train_data['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train_data[train_data['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))

for dataset in total_dataset:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in total_dataset:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

# fill missing Fare with median fare for each Pclass
train_data["Fare"].fillna(train_data.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test_data["Fare"].fillna(test_data.groupby("Pclass")["Fare"].transform("median"), inplace=True)
facet = sns.FacetGrid(train_data, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train_data['Fare'].max()))
facet.add_legend()
plt.show()
facet = sns.FacetGrid(train_data, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train_data['Fare'].max()))
facet.add_legend()
plt.xlim(0, 20)
plt.show()
facet = sns.FacetGrid(train_data, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train_data['Fare'].max()))
facet.add_legend()
plt.xlim(0, 30)
plt.show()
facet = sns.FacetGrid(train_data, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train_data['Fare'].max()))
facet.add_legend()
plt.xlim(0)
plt.show()

for dataset in total_dataset:
    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3

print(train_data.Cabin.value_counts())
for dataset in total_dataset:
    dataset['Cabin'] = dataset['Cabin'].str[:1]


Pclass1 = train_data[train_data['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train_data[train_data['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train_data[train_data['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))
plt.show()

cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in total_dataset:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)

# fill missing Fare with median fare for each Pclass
train_data["Cabin"].fillna(train_data.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test_data["Cabin"].fillna(test_data.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

#Family Size
train_data["FamilySize"] = train_data["SibSp"] + train_data["Parch"] + 1
test_data["FamilySize"] = test_data["SibSp"] + test_data["Parch"] + 1

facet = sns.FacetGrid(train_data, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'FamilySize',shade= True)
facet.set(xlim=(0, train_data['FamilySize'].max()))
facet.add_legend()
plt.xlim(0)
plt.show()

family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in total_dataset:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)

features_drop = ['Ticket', 'SibSp', 'Parch']
train_data = train_data.drop(features_drop, axis=1)
test_data = test_data.drop(features_drop, axis=1)
train_data = train_data.drop(['PassengerId'], axis=1)

train = train_data.drop('Survived', axis=1)
target = train_data['Survived']

print(train.shape, target.shape)
print(train_data.head(10))

infant = [];

for row in train_data["Age"]:
    if row <= 1:
        infant.append(1)
    elif row <= 2:
        infant.append(2)
    elif row <= 5:
        infant.append(3)
    elif row <= 10:
        infant.append(4)
    else:
        infant.append(5)

train_data.insert(3, "Infant", infant, False)


#bar_chart('Infant', train_data);


infant2 = [];

for row in test_data["Age"]:
    if row <= 1:
        infant2.append(1)
    elif row <= 2:
        infant2.append(2)
    elif row <= 5:
        infant2.append(3)
    elif row <= 10:
        infant2.append(4)
    else:
        infant2.append(5)

test_data.insert(3, "Infant", infant2, False)


features = ["Pclass", "Sex", "SibSp", "Parch", "Infant"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

#X = imputer.fit_transform(train_data[features])
#X_test = imputer.fit_transform(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
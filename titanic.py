#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


data_train = pd.read_csv("/kaggle/input/titanic/train.csv")
data_train


# #### Survived : 0 = perished / 1 = survived

# ### Count of unique values of data_train

# In[ ]:


data_train.nunique()


# In[ ]:


data_train.info()


# In[ ]:


data_test = pd.read_csv("/kaggle/input/titanic/test.csv")
data_test


# ### Count of unique values of data_test

# In[ ]:


data_test.nunique()


# In[ ]:


data_test.info()


# ### object into int64 in Sex

# In[ ]:


map_dict = {'female' : 0, 'male' : 1}
data_train['Sex'] = data_train['Sex'].map(map_dict).astype(int)
data_test['Sex'] = data_test['Sex'].map(map_dict).astype(int)


# In[ ]:


data_train.head()


# In[ ]:


data_test.head()


# ### The percentage of Survived and Perished

# In[ ]:


pos = data_train.Survived.value_counts() / data_train.Survived.value_counts().sum()
pos


# In[ ]:


(pos * 100).plot.bar()


# ### The percentage of each Pclass

# In[ ]:


pop = data_train.Pclass.value_counts() / data_train.Pclass.value_counts().sum()
pop


# In[ ]:


(pop * 100).plot.bar()


# ### The percentage of Sex

# In[ ]:


pox = data_train.Sex.value_counts() / data_train.Sex.value_counts().sum()
pox


# In[ ]:


(pox * 100).plot.bar()


# ### The correlation between Pclass and Survived

# In[ ]:


PS_train = data_train[['Pclass', 'Survived']]
PS_train.groupby('Pclass').mean().sort_values(by = 'Survived', ascending = False)


# ### The correlation between Sex and Survived

# In[ ]:


SS_train = data_train[['Sex', 'Survived']]
SS_train.groupby('Sex').mean().sort_values(by = 'Survived', ascending = False)


# ### The correlation between SibSp and Survived

# In[ ]:


SibS_train = data_train[['SibSp', 'Survived']]
SibS_train.groupby('SibSp').mean().sort_values(by = 'Survived', ascending = False)


# It doesn't seem to be highly correlated.

# ### The correlation between Parch and Survived

# In[ ]:


ParS_train = data_train[['Parch', 'Survived']]
ParS_train.groupby('Parch').mean().sort_values(by = 'Survived', ascending = False)


# It doesn't seem to be highly correlated.

# ### The correlation between Age and Survived

# In[ ]:


a = sns.FacetGrid(data_train, col = 'Survived')
a.map(sns.histplot, 'Age', bins = 20)


# ### The correlation between Sex and Survived in terms of Pclass

# In[ ]:


b = sns.FacetGrid(data_train, col = 'Survived', row = 'Pclass', hue = 'Sex')
b.map(sns.histplot, 'Sex')


# ### The correlation between Age and Survived in terms of Pclass

# In[ ]:


c = sns.FacetGrid(data_train, col = 'Survived', row = 'Pclass')
c.map(sns.histplot, 'Age', bins = 20)


# ### The correlation between Embarked and Survived

# In[ ]:


ParS_train = data_train[['Embarked', 'Survived']]
ParS_train.groupby('Embarked').mean().sort_values(by = 'Survived', ascending = False)


# ### Remove less correlated columns in data_train and date_test

# In[ ]:


data_train.columns


# In[ ]:


df_train = data_train.drop(columns = ['Name', 'Ticket', 'Cabin'], axis = 1)


# In[ ]:


df_train


# In[ ]:


df_test = data_test.drop(columns = ['Name', 'Ticket', 'Cabin'], axis = 1)


# In[ ]:


df_test


# In[ ]:


df_train.isnull().sum()


# In[ ]:


df_test.isnull().sum()


# ### Fill NaNs with mean of Age

# In[ ]:


df_train.fillna(df_train.mean()[['Age']], inplace = True)
df_test.fillna(df_test.mean()[['Age']], inplace = True)


# In[ ]:


df_train.isnull().sum()


# In[ ]:


df_test.isnull().sum()


# ### Categorise Age

# In[ ]:


pd.DataFrame(df_train.Age.describe())


# In[ ]:


def function(x) :
    if x < 10 :
        return 0
    if 10 <= x < 20 :
        return 1
    if 20 <= x < 30 :   
        return 2
    if 30 <= x < 40 :
        return 3
    if 40 <= x < 50 :
        return 4
    if 50 <= x < 60 :
        return 5
    if 60 <= x < 70 :
        return 6
    if 70 <= x < 80 :
        return 7
    if 80 <= x < 90 :
        return 8
    else:
        return 9


# In[ ]:


df_train['Age'] = df_train['Age'].apply(function)
df_test['Age'] = df_test['Age'].apply(function)


# In[ ]:


df_train


# In[ ]:


df_test


# ### Fill NaNs with mode of Embarked

# In[ ]:


df_test.Embarked.value_counts()


# In[ ]:


df_train[df_train['Embarked'].isnull()]


# In[ ]:


df_train['Embarked'] = df_train['Embarked'].fillna('S')


# In[ ]:


df_train.isnull().sum()


# In[ ]:


df_train.loc[[62, 829]]


# ### object into int64 in Embarked
# S = 1
# C = 2
# Q = 3

# In[ ]:


map_dict = {'S' : 1, 'C' : 2, 'Q' : 3}
df_train['Embarked'] = df_train['Embarked'].map(map_dict).astype(int)
df_test['Embarked'] = df_test['Embarked'].map(map_dict).astype(int)


# In[ ]:


df_train


# In[ ]:


df_test


# ### Fill NaNs with mean of Fare in df_test

# In[ ]:


df_test.isnull().sum()


# In[ ]:


df_test.fillna(df_test.mean()[['Fare']], inplace = True)


# In[ ]:


df_test.isnull().sum()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


Y = df_train["Survived"]
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = pd.get_dummies(df_train[features])
X_test = pd.get_dummies(df_test[features])


# In[ ]:


model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, Y)
predictions = model.predict(X_test)
output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': predictions})


# In[ ]:


output.head()


# In[ ]:


output.to_csv('submission.csv', index=False)



# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


train = pd.read_csv("cs-training.csv",index_col = 0)
#train.rename(columns={'Unnamed: 0': 'id'})


# In[3]:


print(train.head())


# In[4]:


correlation_matrix = train.corr()
fig = plt.figure(figsize = (10,10))
sns.heatmap(correlation_matrix, square = True)
plt.show()


# According to this the average probability of severe financial distress is 6.7%

# In[63]:


train = train.dropna(how='any')
train.isnull().sum()


# In[124]:


Y = train['SeriousDlqin2yrs']
train.pop('SeriousDlqin2yrs')
print(train.columns)


# In[125]:


print(Y.shape,train.shape)


# Some data wrangling needs to be done so that the number of samples are equal

# In[126]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(train,Y,test_size=.1)


# In[132]:


from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier()
foo = classifier.fit(X_train,Y_train)
y_pred = classifier.predict(X_test)


# In[130]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,Y_test))


# In[166]:


test = pd.read_csv("cs-test.csv",index_col = 0)
print(test.head())
print(test.isnull().sum())


# In[168]:


Y = test['SeriousDlqin2yrs']
test.pop('SeriousDlqin2yrs')


# In[199]:


test.fillna(0,inplace=True)
test.isnull().sum()


# In[222]:


Y = classifier.predict(test)
print(str(Y[0])+","+str(0))


# In[234]:


import csv
import os

csvFile = "submission.csv"
if (os.path.isfile(csvFile)):
    os.remove(csvFile)
filename = csvFile

f = open(filename,"w+")
headers = "id,Probability\n"
f.write(headers)

for x in range(Y.shape[0]):
    f.write(str(x+1)+","+str(Y[x])+"\n")
f.close()


# In[235]:


print(pd.read_csv("submission.csv").head())


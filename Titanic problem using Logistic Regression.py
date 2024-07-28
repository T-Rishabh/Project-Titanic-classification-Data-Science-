#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression with Python
#  [Titanic Data Set from Kaggle](https://www.kaggle.com/c/titanic). 
# 
# We'll be trying to predict a classification- survival or deceased.
# Let's begin our understanding of implementing Logistic Regression in Python for classification.
# 
# We'll use a "semi-cleaned" version of the titanic data set, if you use the data set hosted directly on Kaggle, you may need to do some additional cleaning not shown in this lecture notebook.
# 

# In[26]:


import pandas as pd
import numpy as np


# In[27]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'in line')


# In[28]:


train = pd.read_csv('titanic_train.csv')


# In[29]:


train.head()


# In[30]:


sns.heatmap(train.isnull(),yticklabels=False,cmap='viridis')


# Roughly 20 percent of the Age data is missing. The proportion of Age missing is likely small enough for reasonable replacement with some form of imputation. Looking at the Cabin column, it looks like we are just missing too much of that data to do something useful with at a basic level. We'll probably drop this later, or change it to another feature like "Cabin Known: 1 or 0"
# 

# In[31]:


sns.set_style('whitegrid')


# In[32]:


sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')


# In[33]:


sns.countplot(x='Survived',hue='Pclass',data=train)


# In[34]:


train['Age'].plot.hist(bins=35)


# In[35]:


train.info()


# In[36]:


sns.countplot(x='SibSp',data=train)


# In[37]:


train['Fare'].hist(bins=40,figsize=(10,4))


# In[38]:


import cufflinks as cf


# In[39]:


cf.go_offline()


# In[40]:


train['Fare'].iplot(kind='hist',bins=50)


# ## Data Cleaning
# We want to fill in missing age data instead of just dropping the missing age data rows. One way to do this is by filling in the mean age of all the passengers (imputation).
# However we can be smarter about this and check the average age by passenger class. For example:

# In[41]:


plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass',y='Age',data=train)


# We can see the wealthier passengers in the higher classes tend to be older, which makes sense. We'll use these average age values to impute based on Pclass for Age.

# In[42]:


def impute_age(cols):
    Age=cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass  == 1:
            return 37
        elif Pclass ==2:
            return 29
        else:
            return 24
    else:
        return Age


# In[43]:


train['Age']=train[['Age','Pclass']].apply(impute_age,axis=1)


# In[44]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[45]:


train.drop('Cabin',axis=1,inplace=True)


# In[46]:


train.head()


# In[47]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# ## Converting Categorical Features 
# 
# We'll need to convert categorical features to dummy variables using pandas! Otherwise our machine learning algorithm won't be able to directly take in those features as inputs.

# In[48]:


train.info()


# In[49]:


pd.get_dummies(train['Sex'],drop_first=True)


# In[50]:


sex=pd.get_dummies(train['Sex'],drop_first=True)


# In[51]:


embark=pd.get_dummies(train['Embarked'],drop_first=True)


# In[52]:


train=pd.concat([train,sex,embark],axis=1)


# In[53]:


train.head(2)


# In[54]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[55]:


train.head()


# In[56]:


train.drop('PassengerId',axis=1,inplace=True)


# In[57]:


train.head()


# 
# # Building a Logistic Regression model
# 
# Let's start by splitting our data into a training set and test set (there is another test.csv file that you can play around with in case you want to use all this data for training).
# 
# ## Train Test Split

# In[58]:


X=train.drop('Survived',axis=1)
y=train['Survived']


# In[59]:


from sklearn.model_selection import train_test_split


# In[60]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)


# In[61]:


from sklearn.linear_model import LogisticRegression


# In[62]:


logmodel=LogisticRegression()


# In[63]:


logmodel.fit(X_train,y_train)


# In[64]:


predictions = logmodel.predict(X_test)


# ## Evaluation

# In[65]:


from sklearn.metrics import classification_report


# In[66]:


print(classification_report(y_test,predictions))


# In[ ]:





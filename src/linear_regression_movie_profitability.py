#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing usefull libraries for data pre-processing and in order to run Machine Learning model


# In[2]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[3]:


# Reading dataframe


# In[4]:


df = pd.read_csv('../data/movies_data.csv')


# In[5]:


# Displaying columns


# In[6]:


df.columns


# In[7]:


######################################################################################################################


# In[8]:


# Getting unique Id and the feature we wanna target


# In[9]:


Id = 'Titre'
target = 'Rentabilité (%)'


# In[10]:


# Getting categorical features in order to transorm it later into "dummies"


# In[11]:


features_categ = [u'Réalisateur', u'Scénariste', u'Compositeur', u'Directeur photo', u'Directeur montage',
             u'Acteur 1', u'Acteur 2', u'Acteur 3', u'Genre']


# In[12]:


# Getting numerical features


# In[13]:


features_numerical = ['Popularité genre', 'Popularité thème', 'Rareté émotion', 'Référence',
       'Budget (M$)']


# In[14]:


# Defining which column we want to keep


# In[15]:


to_keep = [Id] + features_numerical + features_categ + [target]


# In[16]:


# Getting in the our datafame only the column we want to keep


# In[17]:


df = df[to_keep]


# In[18]:


# Getting dummies on categorials features


# In[19]:


df = pd.get_dummies(df, columns = features_categ)


# In[20]:


# Creating train and test dataframes by dividing df into two different dataframes 


# In[21]:


df_train = df[:258]
df_test = df[258:]


# In[22]:


# Dropping Id and Target inside intputs: X_train and X_test


# In[23]:


X_train = df_train.drop([Id,target],axis=1)
X_test = df_test.drop([Id,target],axis=1)


# In[24]:


# Getting outputs as y_train and y_test


# In[25]:


y_test = df_test[target]
y_train = df_train[target]


# In[26]:


# Creating Linear Regression


# In[27]:


clf = LinearRegression()


# In[28]:


# Training the model using the training dataset


# In[29]:


clf.fit(X_train, y_train)


# In[30]:


# Making the predictions with Linear Regression


# In[31]:


clf.predict(X_test)


# In[32]:


# Creating y_pred variable representing the predictions made by the model on "Rentabilité" feature
# for each row we wanted to predict


# In[33]:


y_pred = clf.predict(X_test)


# In[34]:


################################
#### performance indicators ####


# In[35]:


mse = mean_squared_error(y_test, clf.predict(X_test))


# In[36]:


mse


# In[37]:


np.sqrt(mse)


# In[38]:


clf.score(X_test,y_test)


# In[39]:


#### performance indicators ####
################################


# In[40]:


# Getting values we need for the analyze inside X_test dataframe


# In[41]:


X_test['Titre'] = df_test.Titre


# In[42]:


X_test['y_pred'] = y_pred


# In[43]:


X_test['y_test'] = y_test


# In[44]:


X_test


#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing usefull libraries for data pre-processing and in order to run Machine Learning model


# In[2]:


import numpy as np
import pandas as pd
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier


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


ID = "Titre"
target = "Succès"


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


to_keep = [ID] + features_numerical + features_categ + [target]


# In[16]:


# Getting in the our datafame only the column we want to keep


# In[17]:


df = df[to_keep]


# In[18]:


# Getting dummies on categorials features


# In[19]:


df = pd.get_dummies(df, columns=features_categ)


# In[20]:


# Creating train and test dataframes by dividing df into two different dataframes 


# In[21]:


df_train = df[:258]
df_test = df[258:]


# In[22]:


# Dropping Id and Target inside intputs: X_train and X_test


# In[23]:


X_train = df_train.drop([ID,target],axis=1)
X_test = df_test.drop([ID,target],axis=1)


# In[24]:


# Getting outputs as y_train and y_test


# In[25]:


y_train = df_train[target]
y_test = df_test[target]


# In[26]:


# Creating Random Forest Classifier


# In[27]:


clf = RandomForestClassifier(class_weight='balanced', 
                             verbose=1,
                             n_estimators=500,
                             random_state=46,
                             max_depth=6)


# In[28]:


# Training the model using the training dataset


# In[29]:


clf.fit(X_train, y_train)


# In[30]:


# Creating y_pred variable representing the predictions made by the model on "Succès" feature
# for each row we wanted to predict


# In[31]:


y_pred = clf.predict(X_test)


# In[32]:


# Getting Scores


# In[33]:


y_pred_proba = clf.predict_proba(X_test)
y_pred_proba


# In[34]:


# Getting values we need for the analyze inside X_test dataframe


# In[35]:


X_test['Titre'] = df_test.Titre


# In[36]:


X_test['y_pred'] = y_pred


# In[37]:


X_test['y_test'] = y_test


# In[38]:


X_test


# In[39]:


################################
#### performance indicators ####


# In[40]:


# Getting the accuracy of the model


# In[41]:


print("Precision Score : ",precision_score(y_test, y_pred, 
                                           pos_label='positive',
                                           average='micro'))


# In[42]:


# Confusion matrix


# In[43]:


confusion_matrix(y_test, y_pred)


# In[44]:


#### performance indicators ####
################################


# In[45]:


# Getting the feature importances


# In[46]:


importance = pd.Series(clf.feature_importances_)


# In[47]:


# Getting the most important features by sorting it


# In[48]:


importance.sort_values(ascending = False)


# In[49]:


# Displaying names of the most important columns of the feature importance


# In[50]:


sorted(zip(clf.feature_importances_, X_test.columns), reverse=True)


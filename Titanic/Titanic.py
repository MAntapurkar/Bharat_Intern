#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score


# In[2]:


t_d = pd.read_csv(r"E:\Bharat Intern\2nd\Titanic\train.csv")


# In[3]:


t_d.head()


# In[4]:


t_d.describe()


# In[5]:


sns.countplot(x='Survived',data=t_d)


# In[6]:


sns.countplot(x='Survived',data=t_d,hue='Sex')


# In[7]:


t_d.isna().sum()


# # Data PreProcessing

# In[8]:


t_d['Age'].fillna(t_d['Age'].mean(),inplace=True)


# In[9]:


t_d.drop('Cabin',axis=1,inplace=True)


# In[10]:


t_d.info()


# In[11]:


t_d.dtypes


# In[12]:


gender=pd.get_dummies(t_d['Sex'],drop_first=True)


# In[13]:


t_d['Gender']=gender


# In[14]:


t_d.drop(['Name','Sex','Ticket','Embarked'],axis=1,inplace=True)


# In[15]:


t_d.head()


# In[16]:


x=t_d[['PassengerId','Pclass','Age','SibSp','Parch','Fare','Gender']]
y=t_d['Survived']


# In[17]:


x,y


# # Data Modelling

# In[18]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[19]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)


# In[20]:


predict=lr.predict(x_test)


# In[21]:


x_train_prediction = lr.predict(x_train)
print(x_train_prediction)


# In[22]:


training_data_accuracy = accuracy_score(y_train, x_train_prediction)
print("Accuracy Score of Training Data is:" , training_data_accuracy)


# In[23]:


x_test_prediction = lr.predict(x_test)
print(x_test_prediction)


# In[24]:


training_data_accuracy = accuracy_score(y_test, x_test_prediction)
print("Accuracy Score of Testing Data is:" , training_data_accuracy)


# In[25]:


from sklearn.ensemble import RandomForestClassifier
dt=RandomForestClassifier()
dt.fit(x_train,y_train)


# In[38]:


from sklearn import metrics
y_pred= dt.predict(x_test)
AC = metrics.accuracy_score(y_pred,y_test)
print("Accuracy of Random Forest Classifier is:", AC)

y_pred


# In[27]:


from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier()
kn.fit(x_train,y_train)


# In[40]:


y_pred=kn.predict(x_test)
AB = metrics.accuracy_score(y_pred,y_test)
print("Accuracy of kN is", AB)


# # All Accuracy Scores

# In[42]:


print("Accuracy Score of Testing Data of Logistic Regression is:" , training_data_accuracy)
print("Accuracy of Random Forest Classifier is:",AC)
print("Accuracy of kN is", AB)


# # Through the above Accuracy Scores We can say that Random Forest Classifier has the highest accuracy

# # Let's See how our model is performing

# In[30]:


#print confusion matrix 


# In[31]:


from sklearn.metrics import confusion_matrix
pd.DataFrame(confusion_matrix(y_test,predict),columns=['Predicted No','Predicted Yes'],index=['Actual No','Actual Yes'])


# In[32]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predict))


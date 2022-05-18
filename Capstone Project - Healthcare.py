#!/usr/bin/env python
# coding: utf-8

# Date - 19 May 2022
# 
# Project by - Anirudh Agarwal
# 
# Cohort - August 2021

# # Capstone Project
# 
# # Healthcare

# **Problem Statement:**
# * NIDDK (National Institute of Diabetes and Digestive and Kidney Diseases) research creates knowledge about and treatments for the most chronic, costly, and consequential diseases.
# * The dataset used in this project is originally from NIDDK. The objective is to predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset.
# * Build a model to accurately predict whether the patients in the dataset have diabetes or not.
# 
# **Dataset Description:**
# The datasets consists of several medical predictor variables and one target variable (Outcome). Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and more.
#  
# **Variables	- Description**
# * Pregnancies -	Number of times pregnant
# * Glucose - Plasma glucose concentration in an oral glucose tolerance test
# * BloodPressure	- Diastolic blood pressure (mm Hg)
# * SkinThickness	- Triceps skinfold thickness (mm)
# * Insulin - Two hour serum insulin
# * BMI - Body Mass Index
# * DiabetesPedigreeFunction - Diabetes pedigree function
# * Age - Age in years
# * Outcome - Class variable (either 0 or 1). 268 of 768 values are 1, and the others are 0

# In[1]:


import pandas as pd
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)


# In[2]:


dataframe = pd.read_csv("/Users/anirudhagarwal/Library/CloudStorage/OneDrive-Personal/Purdue DS Course/Course 8/My Project 2 - Healthcare/health care diabetes.csv")


# In[3]:


# Importing the data

df = dataframe.copy()


# ## Data Exploration

# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.info()               # no null values


# In[7]:


# Checking the number of null values

df.isna().sum()


# In[8]:


df.describe()


# In[9]:


# Although there are no missing values in the Glucose,BP, Skin thickness, Insulin, BMI columns
# of the dataset, value of 0 indicates missing value

print("Percentage of missing values :")
((df.iloc[:,1:6][df.iloc[:,1:6]==0].count())/df.shape[0])*100


# In[10]:


null_col = df.iloc[:,1:6].columns.to_list()
null_col


# In[11]:


# Replacing the null values in the above columns with their respective median

for i in null_col:
    df[i][df[i]==0] = df[df[i]!=0][i].median()    


# In[12]:


# no more 0 values

print("Percentage of missing values :")
((df.iloc[:,1:6][df.iloc[:,1:6]==0].count())/df.shape[0])*100


# In[13]:


df.head()


# In[14]:


df.dtypes,df.dtypes.value_counts()


# In[15]:


# Created a count (frequency) plot describing the data types and the count of variables.
df.dtypes.value_counts().plot(kind='bar');


# In[16]:


# Plotting the count of outcomes by their value
df.Outcome.value_counts(normalize=True)


# In[17]:


df.Outcome.value_counts(normalize=True).plot(kind='bar');


# In[18]:


# Scatter plot amongst features

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
plt.figure(figsize=(10,15))
sns.pairplot(df, height=2)
plt.show()


# In[19]:


# No strong relationship between different features based on scatter plot. Checking correlation for much detailed analysis


# In[20]:


# Correlation matrix
df.corr()


# In[21]:


# Plotting the heatmap using seaborn library
sns.heatmap(df.corr());


# ## Data Modeling

# **I will be using different classifier algorithms and logistic regression algorithm
# to predict the outcome , since the output is a categorical variable.**

# In[22]:


x = df.drop("Outcome", axis=1)
y = df['Outcome']


# In[23]:


x.head()


# In[24]:


y.head()


# In[25]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[26]:


x_scaled = scaler.fit_transform(x)
x_scaled


# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,test_size=0.3,random_state=77,stratify=y)


# ### Random Forest Classifier

# In[29]:


from sklearn.ensemble import RandomForestClassifier
rfc1 = RandomForestClassifier()


# In[30]:


rfc1.fit(x_train,y_train)


# In[31]:


rfc1.score(x_train,y_train)


# In[32]:


rfc1.score(x_test,y_test)


# In[33]:


y_pred = rfc1.predict(x_test)


# In[34]:


from sklearn.metrics import classification_report as cr


# In[35]:


print(cr(y_test,y_pred))


# In[36]:


from sklearn.metrics import confusion_matrix as cm


# In[37]:


pd.DataFrame(cm(y_test,y_pred))


# In[38]:


import numpy as np


# In[39]:


# Hyperparameter tuning

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 120, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [2,4,6,8]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 7]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 3, 5]
# Method of selecting samples for training each tree
bootstrap = [True, False]


# In[40]:


# Create the parameter grid
param_grid1 = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(param_grid1)


# In[41]:


from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, auc, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_val_score


# In[42]:


rfc_gs = GridSearchCV(estimator=rfc1, param_grid=param_grid1, cv=5, verbose=2)
rfc_gs.fit(x_train, y_train)


# In[43]:


rfc_gs.best_params_


# In[44]:


rfc2 = RandomForestClassifier(max_depth=8,max_features='sqrt', min_samples_leaf=3, n_estimators=22,min_samples_split=5,bootstrap=True)


# In[45]:


rfc2.fit(x_train,y_train)


# In[46]:


rfc2.score(x_train,y_train)


# In[47]:


rfc2.score(x_test,y_test)


# In[48]:


y_pred = rfc2.predict(x_test)


# In[49]:


pd.DataFrame(cm(y_test,y_pred))


# In[50]:


# Plotting the ROC curve
prob = rfc2.predict_proba(x_test)                
prob = prob[:, 1]                             
auc_rfc = roc_auc_score(y_test, prob)           
print('AUC: %.3f' %auc_rfc)
fpr, tpr, thresholds = roc_curve(y_test, prob)  
plt.plot([0, 1], [0, 1], linestyle='--')         
plt.plot(fpr, tpr, marker='.')                   
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve");


# ### Decision Tree Classifier

# In[51]:


from sklearn.tree import DecisionTreeClassifier
dtc1 = DecisionTreeClassifier(random_state=77)  


# In[52]:


dtc1.fit(x_train,y_train)


# In[53]:


y_pred = dtc1.predict(x_test)


# In[54]:


dtc1.score(x_test,y_test)


# In[55]:


DecisionTreeClassifier()


# In[56]:


param_grid2 = {
    'criterion':['gini','entropy'],
    'splitter':['best'],
    'max_depth':[list(range(1,9,1)),None],
    'min_samples_split':[1,3,5,7],
    'min_samples_leaf':[1,3,5,7],
    'max_features':['auto'],
    'min_impurity_split':[1,3,5,7]}


# In[57]:


dtc_gs = GridSearchCV(estimator=dtc1, param_grid=param_grid2, cv=5, verbose=2)
dtc_gs.fit(x_train, y_train)


# In[58]:


dtc_gs.best_params_


# In[59]:


dtc2 = DecisionTreeClassifier(criterion='gini',max_features='auto',min_impurity_split=1,min_samples_leaf=1,min_samples_split=3,splitter='best')


# In[60]:


dtc2.fit(x_train,y_train)


# In[61]:


y_pred = dtc2.predict(x_test)


# In[62]:


dtc2.score(x_train,y_train)


# In[63]:


dtc2.score(x_test,y_test)


# In[64]:


pd.DataFrame(cm(y_test,y_pred))


# In[65]:


print(cr(y_test,y_pred))


# In[66]:


prob = dtc2.predict_proba(x_test)                
prob = prob[:, 1]                             
auc_dtc = roc_auc_score(y_test, prob)           
print('AUC: %.3f' %auc_dtc)
fpr, tpr, thresholds = roc_curve(y_test, prob)  
plt.plot([0, 1], [0, 1], linestyle='--')         
plt.plot(fpr, tpr, marker='.')                   
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve");


# ### Support Vector Machines Classifier

# In[67]:


from sklearn.svm import SVC
svm1 = SVC(kernel='rbf')


# In[68]:


svm1.fit(x_train, y_train)


# In[69]:


svm1.score(x_train, y_train)


# In[70]:


svm1.score(x_test, y_test)


# In[71]:


param_grid3 = {
    'C':[1, 5, 10, 15, 20, 25,30,35,40],
    'gamma':[0.001, 0.005, 0.0001, 0.00001]
}


# In[72]:


svm_gs = GridSearchCV(estimator=svm1, param_grid=param_grid3, cv=5, verbose=0)
svm_gs.fit(x_train, y_train)


# In[73]:


svm_gs.best_params_


# In[74]:


svm2 = SVC(kernel='rbf', C=40, gamma=0.005, probability=True)


# In[75]:


svm2.fit(x_train,y_train)


# In[76]:


svm2.score(x_train,y_train)


# In[77]:


y_pred = svm2.predict(x_test)


# In[78]:


svm2.score(x_test,y_test)


# In[79]:


print(cr(y_test,y_pred))


# In[80]:


pd.DataFrame(cm(y_test,y_pred))


# In[81]:


prob = svm2.predict_proba(x_test)                
prob = prob[:, 1]                             
auc_svm = roc_auc_score(y_test, prob)           
print('AUC: %.3f' %auc_svm)
fpr, tpr, thresholds = roc_curve(y_test, prob)  
plt.plot([0, 1], [0, 1], linestyle='--')         
plt.plot(fpr, tpr, marker='.')                   
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve");


# ### Naive Bayes Classifier

# In[82]:


from sklearn.naive_bayes import GaussianNB


# In[83]:


nbc = GaussianNB()


# In[84]:


nbc.fit(x_train,y_train)


# In[85]:


y_pred = nbc.predict(x_test)


# In[86]:


nbc.score(x_train,y_train)


# In[87]:


nbc.score(x_test,y_test)


# No hyper-parameter tuning for Naive Bayes, as there are no critical parameters to optimise

# In[88]:


print(cr(y_test,y_pred))


# In[89]:


pd.DataFrame(cm(y_test,y_pred))


# In[90]:


prob = nbc.predict_proba(x_test)                
prob = prob[:, 1]                             
auc_nbc = roc_auc_score(y_test, prob)           
print('AUC: %.3f' %auc_nbc)
fpr, tpr, thresholds = roc_curve(y_test, prob)  
plt.plot([0, 1], [0, 1], linestyle='--')         
plt.plot(fpr, tpr, marker='.')                   
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve");


# ### Logistic Regression

# In[91]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# In[92]:


lr.fit(x_train,y_train)


# In[93]:


lr.score(x_train,y_train)


# In[94]:


lr.score(x_test, y_test)


# In[95]:


y_pred = lr.predict(x_test)


# No hyper-parameter tuning for Logistic Regression, as there are no critical parameters to optimise

# In[96]:


pd.DataFrame(cm(y_test,y_pred))


# In[97]:


print(cr(y_test,y_pred))


# In[98]:


prob = lr.predict_proba(x_test)                
prob = prob[:, 1]                             
auc_lr = roc_auc_score(y_test, prob)           
print('AUC: %.3f' %auc_lr)
fpr, tpr, thresholds = roc_curve(y_test, prob)  
plt.plot([0, 1], [0, 1], linestyle='--')         
plt.plot(fpr, tpr, marker='.')                   
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve");


# ### K-Nearest Neighbour (KNN) Classifier

# In[99]:


from sklearn.neighbors import KNeighborsClassifier
knn1 = KNeighborsClassifier()


# In[100]:


knn1.fit(x_train, y_train)


# In[101]:


knn1.score(x_train,y_train)


# In[102]:


knn1.score(x_test,y_test)


# In[103]:


param_grid4 = {
    'n_neighbors': list(range(1,20,1))
}


# In[104]:


knn_gs = GridSearchCV(estimator=knn1, param_grid=param_grid4, cv=5, verbose=0)
knn_gs.fit(x_train, y_train)


# In[105]:


knn_gs.best_params_


# In[106]:


knn2 = KNeighborsClassifier(n_neighbors=12)


# In[107]:


knn2.fit(x_train,y_train)


# In[108]:


knn2.score(x_train,y_train)


# In[109]:


y_pred = knn2.predict(x_test)


# In[110]:


pd.DataFrame(cm(y_test,y_pred))


# In[111]:


print(cr(y_test,y_pred))


# In[112]:


prob = knn2.predict_proba(x_test)                
prob = prob[:, 1]                             
auc_knn = roc_auc_score(y_test, prob)           
print('AUC: %.3f' %auc_knn)
fpr, tpr, thresholds = roc_curve(y_test, prob)  
plt.plot([0, 1], [0, 1], linestyle='--')         
plt.plot(fpr, tpr, marker='.')                   
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve");


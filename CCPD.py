import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier
#import xgboost as xgb
import scipy.stats as stats
import seaborn as sns
#conda install -c anaconda xgboost


train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

train.isnull().sum()
test.isnull().sum()
# no missing data

sns.countplot(train['LIMIT_BAL'])
sns.countplot(train['SEX'])
sns.countplot(train['EDUCATION'])
sns.countplot(train['MARRIAGE'])
sns.countplot(train['AGE'])
sns.countplot(train['MARRIAGE'],hue=train['SEX'])
sns.countplot(train['EDUCATION'],hue=train['SEX'])
sns.countplot(train['AGE'],hue=train['SEX'])
sns.countplot(train['default_payment_next_month'])# THERE IS HIGH PROB. THAT FEMALE CUSTOMER MAY DEFAULT


sns.countplot(test['LIMIT_BAL'])
sns.countplot(test['SEX'])
sns.countplot(test['EDUCATION'])
sns.countplot(test['MARRIAGE'])
sns.countplot(test['AGE'])
sns.countplot(test['MARRIAGE'],hue=test['SEX'])
sns.countplot(test['EDUCATION'],hue=test['SEX'])
sns.countplot(test['AGE'],hue=test['SEX'])


#joining train and test dataset
train['source']='train'
test['source']='test'
data=pd.concat([train,test], ignore_index = True, sort = False)
print(train.shape, test.shape, data.shape)

#to see correlation
corr=data.drop('ID',axis=1).corr()
sns.heatmap(corr,vmin=0,vmax=1,center=0,square=True, linewidths=.5)

#to see null values
data.isnull().sum()
data['default_payment_next_month']=data['default_payment_next_month'].fillna(0)
data.isnull().sum()

#splitting the data
data=data.drop('source',axis=1)
X=data.drop('default_payment_next_month',axis=1)
Y=data['default_payment_next_month']

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=3)

#applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=1)
X_train = lda.fit_transform(x_train,y_train)
X_test = lda.transform(x_test)

#Uing random forest classifier
rf=RandomForestClassifier(max_depth=5,random_state=8)

rf.fit(X_train, y_train)
y_pred=rf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy' + str(accuracy_score(y_test, y_pred)))#84.74%

#using decision tree
dt=DecisionTreeClassifier(max_leaf_nodes=2,random_state=4)
dt.fit(X_train, y_train)
y_pred1=dt.predict(X_test)

cm1 = confusion_matrix(y_test, y_pred1)
print(cm1)
print('Accuracy' + str(accuracy_score(y_test, y_pred1)))#84.66%

submission1=pd.DataFrame({'ID':test['ID'],'default_payment_next_month':y_pred})
submission1.to_csv('SUBMISSION_RF.csv',index=False)

submission2=pd.DataFrame({'ID':test['ID'],'default_payment_next_month':y_pred1})
submission2.to_csv('SUBMISSION_DT.csv',index=False)
























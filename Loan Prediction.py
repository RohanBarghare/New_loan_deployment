import pandas as pd 
import numpy as np
import pickle

ss=pd.read_csv('sample_submission.csv')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#categorical
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)

train[(train.isna().sum(axis=1)>0)==True]

train['LoanAmount'].fillna(train['LoanAmount'].mean(), inplace=True)

train[(train.isna().sum(axis=1)>0)==True]

train.to_csv('Cleaned_data.csv', index=False)

modified_train=train
modified_train['Loan_Status']=train['Loan_Status'].apply(lambda x: 0 if x=="N" else 1 )

data = pd.get_dummies(modified_train,columns=['Gender',
 'Married',
 'Dependents',
 'Education',
 'Self_Employed',
 'Property_Area'],drop_first=True)

data

data.drop(['Loan_ID'], axis=1, inplace=True)

data_test = pd.get_dummies(test,columns=['Gender',
 'Married',
 'Dependents',
 'Education',
 'Self_Employed',
 'Property_Area'],drop_first=True)

data_test

X = data.drop(['Loan_Status'],axis=1)
y = data['Loan_Status']

X.dtypes

for x in X.columns:
  X[x]=X[x].astype(int)
  df_types = X.dtypes.value_counts()
print(df_types)

X.dtypes

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import precision_score, recall_score,confusion_matrix,accuracy_score,f1_score,roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state =0)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X_train, y_train)

ypred = model.predict(X_test)

print(classification_report(y_test,ypred))
cm=confusion_matrix(y_test,ypred)
print(cm)
print ("Accuracy of prediction:",round((cm[0,0]+cm[1,1])/cm.sum(),3))
print('F1 score of model is              ',f1_score(y_test,ypred,average='weighted'))

fpr, tpr, _ = metrics.roc_curve(y_test, ypred)
auc = metrics.roc_auc_score(y_test, ypred)

test_dataset=data_test.copy()

data_test.drop(['Loan_ID'], axis=1, inplace=True)

#categorical
data_test['Loan_Amount_Term'].fillna(data_test['Loan_Amount_Term'].mode()[0], inplace=True)
data_test['Credit_History'].fillna(data_test['Credit_History'].mode()[0], inplace=True)

#numerical
data_test['LoanAmount'].fillna(data_test['LoanAmount'].mean(), inplace=True)

ran_test_pred = model.predict(data_test)

ran_test_pred

submission=pd.DataFrame({'Loan_ID' : test_dataset['Loan_ID'],'Loan_Status':ran_test_pred,})

submission

submission['Loan_Status']=submission['Loan_Status'].apply(lambda x: "Y" if x==1 else "N" )

submission.to_csv('Loan Eligibility.csv', index=False)

pickle.dump(model, open('house.pkl', 'wb'))
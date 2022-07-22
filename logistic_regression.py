import pandas as pd
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# load dataset
pima = pd.read_csv("diabetes.csv", skiprows=1, names=col_names) 
print(pima.head(5))
#split dataset in features and target variable
feature_cols = ['pregnant', 'skin', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pima[feature_cols] # Features
y = pima.label # Target variable

# print the first 5 rows of the features 
print(X.head(5))

# print the first 5 rows of the target
print(y.head(5))

# split into train and test 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


print(X_train.shape, y_train.shape, X_test.shape,y_test.shape )


# create an instance of the model 
logreg = LogisticRegression(solver='lbfgs', max_iter=1000)

# fit the model with data
logreg.fit(X_train,y_train)

# predict the model 
y_pred=logreg.predict(X_test)

#evaluate the model using confusion matrix  
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

cnf_matrix = pd.crosstab(y_test, y_pred, rownames = ['Actual'], colnames=['Predicted'])
#sns.heatmap(cnf_matrix, annot =True)
#get the accuracy scores
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

# ROC curve 
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

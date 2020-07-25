# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 04:23:31 2020

@author: aman kumar
"""


"""Wine Quality:
Dataset Description:
The dataset consists of certain features of wine in order to predict its quality as good / not good.
Input variables (based on physicochemical tests):
1 - fixed acidity
2 - volatile acidity
3 - citric acid
4 - residual sugar
5 - chlorides
6 - free sulfur dioxide
7 - total sulfur dioxide
8 - density
9 - pH
10 - sulphates
11 - alcohol
Output variable (based on sensory data):
12 - quality (score between 0 and 10)
Tips
What might be an interesting thing to do, is aside from using regression modelling, is to set an
arbitrary cutoff for your dependent variable (wine quality) at e.g. 7 or higher getting classified as
'good/1' and the remainder as 'not good/0'.
This allows you to practice with hyper parameter tuning on e.g. decision tree algorithms looking at
the ROC curve and the AUC value.
TASK : Build classifier models and ensemble models(ONLY for MEDIAN) to train on the given
dataset"""

#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing dataset
dataset=pd.read_csv('winequality-red.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,[-1]].values

#training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#fitting decision tree classification to the dataset
from sklearn.tree import DecisionTreeClassifier 
classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)

#predicting the test results
y_pred = classifier.predict(X_test)

# Create Classification version of target variable
dataset['goodquality'] = [1 if x >= 7 else 0 for x in dataset['quality']]
dataset_quality = ['good' if x==1 else 'bad' for x in dataset['goodquality']]

y_train_goodquality= [1 if t >= 7 else 0 for t in y_train ] 
y_test_goodquality= [1 if m >= 7 else 0 for m in y_test ] 
y_pred_goodquality= [1 if p >= 7 else 0 for p in y_pred ] 

#creating the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_goodquality, y_pred_goodquality)



















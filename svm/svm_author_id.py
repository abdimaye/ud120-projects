#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn import svm
from sklearn.metrics import accuracy_score

#clf = svm.SVC(kernel="linear")
#clf = svm.SVC(kernel="rbf")
clf = svm.SVC(kernel="rbf", C=10000)


# reduce dataset to 1% to train faster
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)//100]

clf.fit(features_train, labels_train)

predict = clf.predict(features_test)

acc = accuracy_score(predict, labels_test)

print(acc)

# print predictions for elements 10, 26, 50
# 0 corresponds to Sara and 1 corresoponds to Chris
print('predictions for elements: 10, 26, 50', predict[10], predict[26], predict[50])

# print all chris 
# first convert numpy array to a list
list = list(predict)
print(list.count(1))

#########################################################



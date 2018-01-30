#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 10:05:01 2018

@author: llw
"""
#logistic regression
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
density=np.array([0.697,0.774,0.634,0.608,0.556,0.430,0.481,0.437,0.666,0.243,0.245,0.343,0.639,0.657,0.360,0.593,0.719]).reshape(-1,1)
sugar_rate=np.array([0.460,0.376,0.264,0.318,0.215,0.237,0.149,0.211,0.091,0.267,0.057,0.099,0.161,0.198,0.370,0.042,0.103]).reshape(-1,1)
xtrain=np.hstack((density,sugar_rate))
xtrain=np.hstack((np.ones([density.shape[0],1]),xtrain))
ytrain=np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0]).reshape(-1,1)
xtrain,xtest,ytrain,ytest=train_test_split(xtrain,ytrain,test_size=0.25,random_state=33)
def sigmoid(z):
    return 1/(1+np.exp(-z))
#print(sigmoid(density))
def logit_regression(theta,x,y,iteration=100,learning_rate=0.1,lbd=0.01):
    for i in range(iteration):
        theta=theta-learning_rate/y.shape[0]*(np.dot(x.transpose(),(sigmoid(np.dot(x,theta))-y))+lbd*theta)
        cost=-1/y.shape[0]*(np.dot(y.transpose(),np.log(sigmoid(np.dot(x,theta))))+np.dot((1-y).transpose(),np.log(1-sigmoid(np.dot(x,theta)))))+lbd/(2*y.shape[0])*np.dot(theta.transpose(),theta)
        print('---------Iteration %d,cost is %f-------------'%(i,cost))
    return theta
def predict(theta,x):
    pre=np.zeros([x.shape[0],1])
    for idx,valu in enumerate(np.dot(x,theta)):
        if sigmoid(valu)>=0.5:
            pre[idx]=1
        else:
            pre[idx]=0
    return pre
                
'''
theta_init=np.random.rand(3,1)
pre=predict(theta,xtest)
theta=logit_regression(theta_init,xtrain,ytrain,learning_rate=1)
print('predictions are',pre)
print('ground truth is',ytest)
print('theta is ',theta)
print('the accuracy is',np.mean(pre==ytest))
print(classification_report(ytest,pre,target_names=['Bad','Good']))
    
'''
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(C=10)
lr.fit(xtrain,ytrain)
pre=lr.predict(xtest)
print('the accuracy is',lr.score(xtest,ytest))
print(classification_report(ytest,pre,target_names=['Bad','Good']))
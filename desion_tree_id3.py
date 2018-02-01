#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 16:46:11 2018

@author: llw
"""
import numpy as np
class DecisionTree:
    def __init__(self,alpha=0.1,algorithm='id3',threshold=0):
        self.alpha=alpha
        self.algorithm=algorithm
        self.threshold=threshold
        self.m=0      #number of training examples
        self.n=0      #number of features
        self.num_class=0
        self.y_non_repeat=[]
        self.attributes=[]
        self.tree={}
        self.layers=0
        self.leaves=0
    def fit(self,x,y):
        self.m=x.shape[0]
        self.n=x.shape[1]
        self.y_non_repeat=list(set(y[:,0]))
        self.num_class=len(self.y_non_repeat)
        self.attributes=list(range(x.shape[1]))
        if self.algorithm=='id3':
            self.id3(x,y,self.layers,self.attributes)
        elif self.algorithm=='c45':
            self.c45(x,y,self.n)
        elif self.algorithm=='cart_r':
            self.cart_r(x,y,self.n)
        elif self.algorithm=='cart_c':
            self.cart_c(x,y,self.n)
        else:
            print('you need to choose a valid algorithm:id3,c45,cart for classification,cart for regression')
    def id3(self,x,y,layers,attributes):
        myTree={}
        '''
        [x_res,y_res,best_attr,max]=self.sub_tree_generate(x,y,attributes)
        attributes.remove(best_attr)
        if max<self.threshold:
            return {}
        else:
            for key in x_res:
                for attr in attributes:
                    self.tree['layer%d,attribute%d:'%(layers,best_attr)+key]=self.id3(x_res[key],y_res[key],layers,attributes)
            layers+=1
       '''        
            
    def c45(self,x,y):
        pass
    def cart_r(self,x,y):
        pass
    def cart_c(self,x,y):
        pass
    def get_probability(self,x,attribute):
        prob={}
        atb_column=x[:,attribute]    #this is a numpy row !!
        for item in atb_column:
            prob[item.astype(str)]=prob.get(item.astype(str),0)+1
        for key in prob:
            prob[key]=prob[key]/len(atb_column)
        return prob
    def get_class_for_attribute(self,x,y,attribute):
        y=y[:,0]
        attr_column=x[:,attribute]
        num_type_attribute=len(set(attr_column))
        class_prob={}
        prob_array_for_each=np.zeros([num_type_attribute,self.num_class,1])
        for idx,value in enumerate(attr_column):
            class_vec=np.array(self.y_non_repeat).reshape(-1,1)                  # the vector of classes  eg.[0;1;2]
            #print(class_vec)
            
            prob_array_for_each[value]=prob_array_for_each[value]+(class_vec==y[idx]).astype(int)
            #print(prob_array_for_each)
            class_prob[value.astype(str)]=prob_array_for_each[value]
        for key in class_prob:
            class_prob[key]=class_prob[key]/sum(class_prob[key])
        return class_prob
        
    def inform_gain_for_attribute(self,x,y,attribute):
        before_entropy=0;after_entropy=0
        class_prob=self.get_probability(y,0)
        class_attribute_prob=self.get_class_for_attribute(x,y,attribute)
        attribute_prob=self.get_probability(x,attribute)
        for key in class_prob:
            before_entropy+=-class_prob[key]*np.log2(class_prob[key])
        for key in attribute_prob:
            for value in class_attribute_prob[key]:
                    if value!=0:
                        after_entropy+=-attribute_prob[key]*value*np.log2(value)
                    else:
                        continue                 
        return before_entropy-after_entropy
    def choose_best_attr(self,x,y,attributes):
        #num_attr=len(attributes)
        maximum=0;idx=0
        for attr in attributes: 
            if maximum<self.inform_gain_for_attribute(x,y,attr):
                #print(self.inform_gain_for_attribute(x,y,i))
                maximum=self.inform_gain_for_attribute(x,y,attr)
                idx=attr
        return [idx,maximum]
    def sub_tree_generate(self,x,y,attributes):
        y_non_repeat=set(y[:,0])
        if len(y_non_repeat)==1:
            return y_non_repeat[0]
        if not attributes:                                                                   
            cls_list=[value for value in self.get_probability(y,0).values()]               
            return cls_list.index(max(cls_list))
        [best_attr,maximum]=self.choose_best_attr(x,y,attributes)    
        attr_column=x[:,best_attr]
        attr_column_non_repeat=set(attr_column)
        clf_res={}
        for value in attr_column_non_repeat:
            clf_res[value.astype(str)]=[]
        for idx,value in enumerate(attr_column):
            clf_res[value.astype(str)].append(idx)
        x_res={}
        y_res={}
        for key in clf_res:
            x_res[key]=x[clf_res[key]]
            y_res[key]=y[clf_res[key]]
        for key in x_res:
            
        
        

if __name__=='__main__':
    
    x_train=np.array([[0,0,0,0],[0,0,0,1],[0,1,0,1],[0,1,1,0],[0,0,0,0],[1,0,0,0],[1,0,0,1],[1,1,1,1],[1,0,1,2],[1,0,1,2],[2,0,1,2],[2,0,1,1],[2,1,0,1],[2,1,0,2],[2,0,0,0]])
    y_train=np.array([[0],[0],[1],[1],[0],[0],[0],[1],[1],[1],[1],[1],[1],[1],[0]])
    dt=DecisionTree(alpha=1)
    dt.fit(x_train,y_train)
    test=dt.get_probability(y_train,0)
    #dt.get_class_for_attribute(x_train,y_train,3)
    #dt.inform_gain_for_attribute(x_train,y_train,0)
    test=dt.sub_tree_generate(x_train,y_train,[])
    print(test)
    
        
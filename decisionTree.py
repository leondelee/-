#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 16:46:11 2018

@author: llw
"""
import numpy as np

class DecisionTree:
    '''
---作用：生成一个决策数类
---参数说明：
  --labels：这是训练样本的不同feature的标签，是必须输入的，比如一个人的年龄，工作，信贷情况，要输入一个字符串的列表
  --alpha：这是决策数的正则化系数，默认为0.1
  --threshold：这是信息增益的阈值，当一个属性的信息增益小于这个阈值时，结束迭代，生成叶节点

'''
    def __init__(self,labels,alpha=0.1,threshold=0):
        self.alpha=alpha
        self.threshold=threshold
        self.m=0      #number of training examples
        self.n=0      #number of features
        self.num_class=0
        self.y_non_repeat=[]
        self.attributes=[]
        self.tree={}
        self.labels=labels
    def fit(self,x,y):
        self.m=x.shape[0]
        self.n=x.shape[1]
        self.y_non_repeat=list(set(y[:,0]))
        self.num_class=len(self.y_non_repeat)
        self.attributes=list(range(x.shape[1]))
        self.tree=self.sub_tree_generate(x,y,self.attributes,{})
    
    def get_probability(self,x,attribute):
        '''
    ---作用：依据选择的属性将样本分割
    ---参数说明：
      --x：一个m×n的矩阵
      --attribute：分割x所依据的属性，是一个0——（n-1）的int
    ---返回值：
      --prob 返回一个字典，字典的键是这组数据所包含的选择的属性的取值，对应的值是该属性取值所占的比例
    '''
        prob={}
        atb_column=x[:,attribute]    
        for item in atb_column:
            prob[item.astype(str)]=prob.get(item.astype(str),0)+1
        for key in prob:
            prob[key]=prob[key]/len(atb_column)
        return prob
    def get_class_for_attribute(self,x,y,attribute):
        '''
        ---作用：对于每一个属性，将他们按照类别分割
	    ---参数说明：
	     --x：训练样本x
    	     --y：训练样本y
             --attribute：所依据的属性，取值为一个0——（n-1）的int
	---返回值：
             --返回一个字典，字典的键是属性的取值，对应的值是在这个属性取值的情况下样本的种类分布
	'''
        y=y[:,0]
        attr_column=x[:,attribute]
        num_type_attribute=len(set(attr_column))
        class_prob={}
        prob_array_for_each=np.zeros([num_type_attribute,self.num_class,1])
        for idx,value in enumerate(attr_column):
            
            class_vec=np.array(self.y_non_repeat).reshape(-1,1)                  
            
            prob_array_for_each[value.astype(int)]=prob_array_for_each[value.astype(int)]+(class_vec==y[idx]).astype(int)
        
            class_prob[value.astype(int)]=prob_array_for_each[value.astype(int)]
        for key in class_prob:
            class_prob[key]=class_prob[key]/sum(class_prob[key])
        return class_prob
            
    
    def inform_gain_for_attribute(self,x,y,attribute):
        '''
    ---作用：获取某一属性的信息增益
    ---参数说明：
      --x：训练样本x
      --y：训练样本y
      --attribute：所依据的属性，取值为一个0——（n-1）的int
    返回值：
      --before_entropy-after_entropy：选择该属性的信息增益
    '''
        before_entropy=0;after_entropy=0
        class_prob=self.get_probability(y,0)
        class_attribute_prob=self.get_class_for_attribute(x,y,attribute)
        attribute_prob=self.get_probability(x,attribute)
        for key in class_prob:
            before_entropy+=-class_prob[key]*np.log2(class_prob[key])
        for key in attribute_prob:
            for value in class_attribute_prob[key.astype(int)]:
                    if value!=0:
                        after_entropy+=-attribute_prob[key]*value*np.log2(value)
                    else:
                        continue                 
        return before_entropy-after_entropy
    
    def choose_best_attr(self,x,y,attributes):
        '''
    ---作用：选择最优的属性（信息增益最大）
    ---参数说明：
      --x：训练样本x
      --y：训练样本y
      --attributes：该训练样本所有可以选择的属性，是一个列表
    返回值：
     --[idx,maximum]：
      -->idx：信息增益最大的属性所在attributes列表中的下标
      -->maximum：最大的信息增益值
    '''
        #num_attr=len(attributes)
        maximum=0;idx=0'''
    ---作用：预测结果
    ---输入变量：
    --x：需要预测的子变量
    ---返回值：
    --pre_res:预测结果
    '''
        for attr in attributes: 
            if maximum<self.inform_gain_for_attribute(x,y,attr):
                #print(self.inform_gain_for_attribute(x,y,i))
                maximum=self.inform_gain_for_attribute(x,y,attr)
                idx=attr
        return [idx,maximum]
        
    def sub_tree_generate(self,x,y,attributes,myTree):
        '''
            ---作用：生成决策树
            ---参数说明：
              --x：训练样本x
              --y：训练样本y
             --attributes：该训练样本所有可以选择的属性，是一个列表
            myTree：是一个字典，表示该节点的父节点
            ---：
            返回在该节点生成的子节点
            '''

        y_non_repeat=set(y[:,0])
        if len(y_non_repeat)==1:
            return list(y_non_repeat)[0]
        if not attributes:                                                                   
            cls_list=[value for value in self.get_probability(y,0).values()]               
            return cls_list.index(max(cls_list))
        [best_attr,maximum]=self.choose_best_attr(x,y,attributes)    
        attributes.remove(best_attr)
        attr_column=x[:,best_attr]
        attr_column_non_repeat=list(set(attr_column))
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
            #myTree[key]=self.sub_tree_generate(x_res[key],y_res[key],attributes,{})
            myTree[self.labels[best_attr]+':'+key]=self.sub_tree_generate(x_res[key],y_res[key],attributes,{})
        return myTree

    def makeDecision(self,thisFeature,tree):
        '''
       ---作用：进行预测
       ---输入值：
           --thisFeature：是一个列表，列表的每一个元素都对应了输入变量的每一个属性映射到决策树上的字典形式
           --tree：对输入变量的筛选子节点
       ---返回值：
        --res：该输入变量的最终预测结果
        '''
        res=0
        for key in tree:
            if key in thisFeature:
                if type(tree[key])!=dict:
                    res=tree[key]
                else:
                    self.makeDecision(thisFeature,tree[key])
        return res
    def predict(self,x):
        '''
        ---作用：预测结果
        ---输入变量：
        --x：需要预测的子变量
        ---返回值：
        --pre_res:预测结果
        '''
        pre_res=np.zeros([x.shape[0],1])
        for idx,sample in enumerate(x):
            thisFeature=[]
            tree=self.tree
            for attr_idx,attribute in enumerate(sample):
                thisFeature.append(self.labels[attr_idx]+':'+str(attribute))
            pre_res[idx]=self.makeDecision(thisFeature,tree)
        return pre_res

if __name__=='__main__':
    x_train=np.array([[0,0,0,0],[0,0,0,1],[0,1,0,1],[0,1,1,0],[0,0,0,0],[1,0,0,0],[1,0,0,1],[1,1,1,1],[1,0,1,2],[1,0,1,2],[2,0,1,2],[2,0,1,1],[2,1,0,1],[2,1,0,2],[2,0,0,0]])
    y_train=np.array([[0],[0],[1],[1],[0],[0],[0],[1],[1],[1],[1],[1],[1],[1],[0]])
    dt=DecisionTree(labels=['age','job','house','credit'])
    #dt=DecisionTree(labels=digits.target_names)
    dt.fit(x_train,y_train.reshape(-1,1))
    print(dt.tree)                      
        

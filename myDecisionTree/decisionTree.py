# -*- coding: UTF-8 -*-

from math import  log


"""
计算给定数据集的香农熵
parameter:
    dataset:数据集
returns:
    香农熵
"""
def calShang(dataset):
    datasize = len(dataset)
    lables={}
    for vec in dataset:
        currentlable = vec[-1]
        if currentlable not in lables.keys():
            lables[currentlable]=0
        lables[currentlable]=lables[currentlable]+1
    shang=0
    for key in lables.values():
        prob=float(key/datasize)
        shang-=prob*log(prob,2)
    return  shang

""""
创建数据集
paramet:
无
returns:
dataset:数据集
lables:标签
"""
def createData():
    dataset=[[0, 0, 0, 0, 'no'],
			[0, 0, 0, 1, 'no'],
			[0, 1, 0, 1, 'yes'],
			[0, 1, 1, 0, 'yes'],
			[0, 0, 0, 0, 'no'],
			[1, 0, 0, 0, 'no'],
			[1, 0, 0, 1, 'no'],
			[1, 1, 1, 1, 'yes'],
			[1, 0, 1, 2, 'yes'],
			[1, 0, 1, 2, 'yes'],
			[2, 0, 1, 2, 'yes'],
			[2, 0, 1, 1, 'yes'],
			[2, 1, 0, 1, 'yes'],
			[2, 1, 0, 2, 'yes'],
			[2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    return dataset,labels

"""
选择最优特征
parater:
数据集
returns：
bestfeature
"""
def chooseBestFeature(dataset):
    featureNum = len(dataset[0])-1
    baseshang=calShang(dataset)
    bestinfoGain=0
    bestfeature=-1
    for i in range(featureNum):
        featurelist=[feature[i] for feature in dataset]
        unifeature = set(featurelist)
        newshang=0
        for feature in unifeature:
            subdata = split(dataset,i,feature)
            prob = len(subdata)/float(len(dataset))
            newshang+=prob*calShang(subdata)
        infoGain = baseshang-newshang
        if infoGain>bestinfoGain:
            bestfeature=i
            bestinfoGain=infoGain
    return  bestfeature

if  __name__=="__main__":
    dataset,lables = createData()
    print(calShang(dataset))
# -*- coding: UTF-8 -*-

import  numpy as np
from matplotlib.font_manager import FontProperties
import  matplotlib.pyplot as plt
import matplotlib.lines as mlines


"""
打开文件，解析数据
parameters:
    filename:文件名
return:
    returnMat:特征矩阵
    lableVector:类别标签

"""
def file2matrix(filename):
    fr = open(filename,"r",encoding="utf-8")
    arrayoflines = fr.readlines()
    arrayoflines[0] = arrayoflines[0].lstrip('\ufeff')
    datanumber = arrayoflines.__len__()
    returnMat=np.zeros((datanumber,3))
    lableVector=[]
    index=0
    for line in arrayoflines:
        line=line.strip()
        listfromLine = line.split('\t')
        templist=[]
        for item in listfromLine[0:3]:
            templist.append(float(item))
        returnMat[index]= templist
        if(listfromLine[-1]=="didntLike"):
            lableVector.append(1)
        elif listfromLine[-1]=="smallDoses":
            lableVector.append(2)
        elif listfromLine[-1]=="largeDoses":
            lableVector.append(3)
        index=index+1
    return returnMat,lableVector

"""
可视化数据
parameter:
    dataMat:数据矩阵
    dataLable:分类标签
return:
    无
"""
def showdata(dataMat,dataLable):
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13, 8))
    LabelsColors=[]
    for i in dataLable:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
        if i == 3:
            LabelsColors.append('red')
    axs[0][0].scatter(x=dataMat[:,1],y=dataMat[:,2],color=LabelsColors,s=15, alpha=.5)
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比')
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数')
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占比')
    plt.setp(axs0_title_text, size=9, weight='bold', color='red',FontProperties=font)
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black',FontProperties=font)
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black',FontProperties=font)

    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=dataMat[:, 0], y=dataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数', FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数', FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=dataMat[:, 1], y=dataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数', FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比', FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数', FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
    # 设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                              markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                               markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                               markersize=6, label='largeDoses')

    # 添加图例
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])

"""
数据归一化
parameter:
    dataMat:数据矩阵
return:
    normMat:归一化后的数据矩阵
"""
def normData(dataMat):
    minValue = dataMat.min(0)
    maxValue = dataMat.max(0)
    range = maxValue-minValue
    normdata=np.zeros(np.shape(dataMat))
    normdata = dataMat-np.tile(minValue,(dataMat.shape[0],1))
    normdata = normdata / np.tile(range,(dataMat.shape[0],1))
    return  normdata

"""
:parameter
datatest：测试集
datatrain：训练集
lablevector：类别标签
k:近邻个数
:return
类别标签
"""
def classify(datatest,datatrain,lablevector,k):
    datanumber = datatrain.shape[0]
    diffMat = np.tile(datatest,(datanumber,1))-datatrain
    sqdiffMat = diffMat**2
    sqDistances=sqdiffMat.sum(axis=1)
    distance=sqDistances**0.5
    sortedDistance =  distance.argsort()
    classCount={}
    for i in range(k):
        lable = lablevector[sortedDistance[i]]
        classCount[lable] = classCount.get(lable,0)+1
    sortedclassCount=sorted(classCount,key=operator.itemgetter(1),reverse=True)
    return sortedclassCount[0][0]

"""
:parameter
    无
:return
    无

"""
def classifyTest():
    filename = "datingTestSet.txt"
    dataMat,lable = file2matrix(filename)

    ration=0.1
    normdata = normData(dataMat)
    testnumber = int(normData.shape[0]*0.1)

    errorCount=0

    for i in range(testnumber):
        classresult = classify(normdata[i,:],normdata[testnumber:m,:],lable[testnumber:m,1],4)
        print("分类结果%s\t真实类别%s" %(classresult,lable[i]))





if __name__=="__main__":
    filename="datingTestSet.txt"
    data,lable = file2matrix(filename)
    classifyTest()
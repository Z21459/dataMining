from math import log
import pandas as pd
import numpy as np
import treePlotter
import copy
import operator
from itertools import combinations

def huafenMethod(listt):
    ans=[]
    if(len(listt)==2):
        pre=[]
        listtt=list(listt)
        pre.append(list(listtt[0]))
        pre.append(list(listtt[1]))
        ans.append(pre)
        print("2个元素时返回了："+str(ans))
        return ans
    t=combinations(listt,2)

    for i in t:
        u = []
        #print("i:  "+str(i))
        for ii in listt:
            if ii not in i:
                #print("ii"+ii)
                u.append(list(ii))
        u.append(list(i))
        ans.append(u)
    print("3个元素时返回了"+str(ans))
    return ans

#读取数据，step1：计算信息熵
def readData(fileName):
    f=open(fileName,'r',encoding="utf-8")
    lines=f.readlines()
    labels=['计数','年龄','收入','学生','信誉']
    labelCount={}
    dataSet=[]
    lenn=0
    print("lines:  "+str(lines))
    for line in lines[0:]:
        print("line: "+line)
        line=line.strip().split(',')
        dataSet.append(line)
        print("linnne: "+str(line))
    return dataSet,labels


def calEnt(dataSet):
    print("进入calENT的dataset："+str(dataSet))
    labelCount={}
    lenn=0
    for line in dataSet:
        currentLabel=line[-1]
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel]=0
        #for i in range(0,line[0]):
        labelCount[currentLabel]+=int(line[0])
        lenn+=int(line[0])
        #print("line[0]: "+line[0])
    print("labelCount:  "+str(labelCount))
    ent=0.0

    for i in labelCount:
        print("---------"+str(labelCount[i])+" "+str(lenn))
        r=float(labelCount[i])/lenn
        ent=ent-r*log(r,2)
    return labelCount,lenn,ent,dataSet

def calGini(dataSet):
    labelCount = {}
    lenn = 0
    for line in dataSet:
        currentLabel = line[-1]
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel] = 0
        # for i in range(0,line[0]):
        labelCount[currentLabel] += int(line[0])
        lenn += int(line[0])
        # print("line[0]: "+line[0])
    print("labelCount:  " + str(labelCount))
    ent = 0.0
    preGini = 1.0
    for i in labelCount:
        print("---------" + str(labelCount[i]) + " " + str(lenn))
        r = float(labelCount[i]) / lenn
        preGini=preGini-r*r
    return preGini

#划分数据集，把原有的数据集去掉属性为value的
def splitdataset(dataset,feather,value):#数据集，属性，属性值
    retdataset=[]#创建返回的数据集列表
    lenn=0
    for line in dataset:#抽取符合划分特征的值
        if line[feather]==value:
            reducedfeatVec=line[:feather] #去掉feather特征
            reducedfeatVec.extend(line[feather+1:])#将符合条件的特征添加到返回的数据集列表
            retdataset.append(reducedfeatVec)
        #else:
            #print(line[0])
            lenn+=int(line[0])
    print("split后返回的"+str(lenn))
    return retdataset,lenn

#划分数据集，把原有的数据集去掉属性为value的
def splitdataset2(dataset,feather,value):#数据集，属性，属性值
    retdataset=[]#创建返回的数据集列表
    lenn=0
    labelCount = {}
    if isinstance(value,list):
        value = list(value)
    for line in dataset:#抽取符合划分特征的值
        #print("line feather"+str(type(line)))
        if line[feather] in value:
            reducedfeatVec=line[:feather] #去掉feather特征
            reducedfeatVec.extend(line[feather+1:])#将符合条件的特征添加到返回的数据集列表
            retdataset.append(reducedfeatVec)

            currentLabel = line[-1]
            if currentLabel not in labelCount.keys():
                labelCount[currentLabel] = 0
                # for i in range(0,line[0]):
            labelCount[currentLabel] += int(line[0])
            lenn += int(line[0])

    print("split222222后返回的"+str(value)+" 中 "+str(labelCount))
    return retdataset,lenn,labelCount

'''
选择最好的数据集划分方式
ID3算法:以信息增益为准则选择划分属性
C4.5算法：使用“增益率”来选择划分属性
'''

def ID3_chooseBestFeatureToSplit(dataset):#ID3算法
    print("调用选择属性算法")
    numFeatures=len(dataset[0])-1
    labelCount,lenn,ent,dataSet=calEnt(dataset)
    baseEnt=ent
    print("baseEnt"+str(baseEnt))
    bestInfoGain=0.0
    bestFeature=-1
    for i in range(1,numFeatures): #遍历所有特征
        #for example in dataset:
            #featList=example[i]
        featList=[example[i]for example in dataset]
        uniqueVals=set(featList) #将特征列表创建成为set集合，元素不可重复。创建唯一的分类标签列表
        newEnt=0.0
        for value in uniqueVals:     #计算每种划分方式的信息熵
            print("ID3划分："+str(value))
            subdataset,lenn=splitdataset(dataset,i,value)
            p=lenn/float(lenns)#float(len(dataset))(len(subdataset))
            labelCount, lenn, ent, dataSet=calEnt(subdataset)
            newEnt+=p*ent
            print("ID3:ent:"+str(ent))
            print("ID3:p"+str(p))
        print("ID3:allnewent"+str(newEnt))
        infoGain=baseEnt-newEnt
        print(u"ID3中第%d个特征的信息增益为：%.4f"%(i,infoGain))
        if (infoGain>bestInfoGain):
            bestInfoGain=infoGain    #计算最好的信息增益
            bestFeature=i
    return bestFeature


def C45_chooseBestFeatureToSplit(dataset):#C4.5算法
    numFeatures=len(dataset[0])-1
    labelCount, lenn, baseEnt, dataSet=calEnt(dataset)
    bestInfoGain_ratio=0.0
    bestFeature=-1
    for i in range(1,numFeatures): #遍历所有特征
        featList=[example[i]for example in dataset]
        uniqueVals=set(featList) #将特征列表创建成为set集合，元素不可重复。创建唯一的分类标签列表
        newEnt=0.0
        IV=0.0
        for value in uniqueVals:     #计算每种划分方式的信息熵
            print("C4.5划分：" + str(value))
            subdataset, lenn=splitdataset(dataset,i,value)
            p=lenn/float(lenns)#len(subdataset)/float(len(dataset))
            labelCount, lenn, ent, dataSet=calEnt(subdataset)
            newEnt+=p*ent
            IV=IV-p*log(p,2)
        infoGain=baseEnt-newEnt
        if (IV == 0): # fix the overflow bug
            continue
        infoGain_ratio = infoGain / IV #这个feature的infoGain_ratio
        print(u"C4.5中第%d个特征的信息增益率为：%.3f"%(i,infoGain_ratio))
        if (infoGain_ratio >bestInfoGain_ratio):  #选择最大的gain ratio
            bestInfoGain_ratio = infoGain_ratio
            bestFeature = i #选择最大的gain ratio对应的feature
    return bestFeature


def CART_chooseBestFeatureToSplit(dataset):#CART算法

    numFeatures = len(dataset[0]) - 1
    bestGini = float('inf') #初始化为最大值
    bestFeature = -1
    bestHuaFen = []
    for i in range(1,numFeatures):
        featList = [example[i] for example in dataset]
        uniqueVals = set(featList)
        lei = huafenMethod(uniqueVals)
        gini = 1.0
        print("lei:  "+str(lei))
        #eachhuafen
        for eachhuafen in lei:
            print("CART划分：" + str(eachhuafen))
            gini=0.0
            for eachLei in eachhuafen:
                preGini=1.0
                retdataset,lenn,labelCount=splitdataset2(dataset,i,eachLei)
                p=lenn/float(lenns)
                for labels in labelCount:
                    preGini=preGini-pow(labelCount[labels]/lenns,2)
                gini=gini+p*preGini
                print("Gini: "+str(gini))
                if (gini < bestGini):
                    bestGini = gini
                    bestFeature = i
                    bestHuaFen = eachhuafen
        print(u"CART中第%d个特征的基尼值为：%.3f"%(i,gini))
    return bestFeature,bestHuaFen


def ID3_createTree(dataset,labels):#利用ID3算法创建决策树
    print("进入ID3 create_tree"+str(dataset))
    classList=[]
    classList=[str(example[-1]) for example in dataset]
    print("classList:"+str(classList))
    if classList.count(classList[0]) == len(classList):# 类别完全相同，停止划分
        return classList[0]
    if len(dataset[0]) == 2:# 遍历完所有特征时返回出现次数最多的
        return majorityCnt(dataset)
    bestFeat = ID3_chooseBestFeatureToSplit(dataset)

    print("best:"+str(bestFeat))
    print("labels:"+str(labels))
    bestFeatLabel = labels[bestFeat]
    print(u"此时最优索引为："+(bestFeatLabel))
    ID3Tree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataset]# 得到列表包括节点所有的属性值
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        dataSet, lenn =splitdataset(dataset, bestFeat, value)
        #if bestFeat != -1:
        ID3Tree[bestFeatLabel][value] = ID3_createTree(dataSet, subLabels)
    return ID3Tree

def C45_createTree(dataset,labels):#C4.5创建决策树
    print("进入C45"+str(dataset))
    classList=[]
    for example in dataset:
        print(example)
        classList.append(example[-1])
    if classList.count(classList[0]) == len(classList):
        # 类别完全相同，停止划分
        return classList[0]
    if len(dataset[0]) == 2:
        # 遍历完所有特征时返回出现次数最多的
        return majorityCnt(dataset)
    bestFeat = C45_chooseBestFeatureToSplit(dataset)
    bestFeatLabel = labels[bestFeat]
    print(u"此时最优索引为："+(bestFeatLabel))
    C45Tree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    # 得到列表包括节点所有的属性值
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        dataSet, lenn=splitdataset(dataset, bestFeat, value)
        C45Tree[bestFeatLabel][value] = C45_createTree(dataSet, subLabels)
    return C45Tree

def CART_createTree(dataset,labels):#CART创建决策树
    print("进入CART建树： "+str(dataset))
    classList = []
    for example in dataset:
        print(example)
        classList.append(example[-1])
    if classList.count(classList[0]) == len(classList):# 类别完全相同，停止划分
        return classList[0]
    if len(dataset[0]) == 2:# 遍历完所有特征时返回出现次数最多的
        return majorityCnt(dataset)
    bestFeat,bestHuaFen = CART_chooseBestFeatureToSplit(dataset)
    bestFeatLabel = labels[bestFeat]
    print(u"此时最优索引为："+(bestFeatLabel))
    CARTTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = []# 得到列表包括节点所有的属性值
    for classs in bestHuaFen:
        print(str(classs))
        featValues.append(str(classs))
    print((featValues))
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        retdataset, lenn, labelCount=splitdataset2(dataset, bestFeat, value)
        CARTTree[bestFeatLabel][value] = CART_createTree(retdataset, subLabels)
    return CARTTree

def majorityCnt(classList):#多数表决法
    #print("进入major"+str(classList))
    classCont={}
    for vote in classList:
        if vote[-1] not in classCont.keys():
            classCont[vote[-1]]=0
        classCont[vote[-1]]+=int(vote[0])
    sortedClassCont=sorted(classCont.items(),key=operator.itemgetter(1),reverse=True)
    #print("major返回了什么"+str(sortedClassCont[0][0]))
    return sortedClassCont[0][0]


def classify(inputTree, featLabels, testVec):
    """
    输入：决策树，分类标签，测试数据
    输出：决策结果
    描述：跑决策树
    """
    print("inputTree, featLabels, testVec"+str(inputTree)+str(featLabels)+str(testVec))
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    print("secondDict"+str(secondDict))
    featIndex = featLabels.index(firstStr)
    classLabel = '0'
    for key in secondDict.keys():
        print("testVec[featIndex] == key"+str(testVec[featIndex])+str(key))
        if testVec[featIndex] in key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    print("分类结果"+str(classLabel))
    return classLabel

def calLeaf(inputTree):
    leafCnt = 0
    keys = list(inputTree.keys())

    for key in keys:
        #if testVec[featIndex] == key:
        print("key" +str(key)+"  "+str(inputTree[key]))
        print("type(inputTree[key]).__name__ == 'dict'"+str(type(inputTree[key]).__name__ == 'dict'))
        if type(inputTree[key]).__name__ == 'dict':
            leafCnt +=   calLeaf(inputTree[key])
        else:
            leafCnt +=  1

    print("return "+str(leafCnt))
    return leafCnt

# 计算预测误差
def calcTestErr(myTree, testData, labels):
    print(str(testData))
    errorCount = 0.0
    RR = 0.0
    for i in range(len(testData)):
        RR += int(testData[i][0])
        if classify(myTree, labels, testData[i]) != testData[i][-1]:
            errorCount += int(testData[i][0])
    TT = calLeaf(myTree)
    errGain = (float(RR/float(lenns))-float(errorCount/lenns))/float(TT-1.0)
    return errGain


#使用模型进行预测
def isTree(obj):#判断当前节点是否是叶节点
    return type(obj).__name__ == 'dict'

def pruningTree(inputTree, dataset,  labels):
    print("inputTree"+str(inputTree))
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]  # 获取子树
    #classList = [example[-1] for example in dataSet]
    classList = []
    print("dataset"+str(dataset))
    #dataset = dataset[0]
    for example in dataset:
        print("example: "+str(example))
        classList.append(example[-1])

    if classList.count(classList[0]) == len(classList):
        # 类别完全相同，停止划分
        return classList[0]
    if len(dataset[0]) == 2:
        # 遍历完所有特征时返回出现次数最多的
        return majorityCnt(dataset)

    featKey = copy.deepcopy(firstStr)
    labelIndex = labels.index(featKey)
    subLabels = copy.deepcopy(labels)
    del (labels[labelIndex])
    print("jinfor"+str(classList))
    retdataset=[]
    rettestset=[]
    print("list(secondDict.keys())"+str(list(secondDict.keys())))
    print("labelIndex"+str(labelIndex))
    for key in list(secondDict.keys()):
        if isTree(secondDict[key]):
            # 深度优先搜索,递归剪枝
            retdataset,lenn,labelCount = splitdataset2(dataset, labelIndex, key)
            #rettestset, lenn, labelCount = splitdataset2(testData, labelIndex, key)
            print("ret: "+str(retdataset))
            if len(retdataset[0]) > 2 :
                inputTree[firstStr][key] = pruningTree(secondDict[key], retdataset, copy.deepcopy(labels))
    print("if判断之前"+str(classList))
    err = calcTestErr(inputTree, dataset, subLabels)
    print("err"+str(err))
    if  err>0.2 : #testMajor(majorityCnt(dataset), testData):
        # 剪枝后的误差反而变大，不作处理，直接返回
        return inputTree
    else:
        # 剪枝，原父结点变成子结点，其类别由多数表决法决定
        print("classList"+str(classList))
        return majorityCnt(dataset)

dataSet,labels = readData("./dataOfID3andC45.csv")
labelCount,lenns,ent,dataSet=calEnt(dataSet)
print("step1:总共"+str(lenns)+"个样本"+"\n\t"+str(labelCount)+"\n\t"+"决策属性的熵："+str(ent))
print(str(dataSet))
print(u"以下为首次寻找最优索引:\n")
print(u"ID3算法的最优特征索引为:"+str(ID3_chooseBestFeatureToSplit(dataSet)))
print ("--------------------------------------------------")
print(u"C4.5算法的最优特征索引为:"+str(C45_chooseBestFeatureToSplit(dataSet)))
print ("--------------------------------------------------")
#dataSet,labels=readData("./dataOfCART.csv")
print(u"CART算法的最优特征索引为:"+str(CART_chooseBestFeatureToSplit(dataSet)))

print("000000000000"+str(dataSet))

"""
孔令鑫版权所有，禁止抄袭
"""
while(True):
        print("孔令鑫版权所有，禁止抄袭")
        dec_tree=str(input("请选择决策树类型:(1:ID3; 2:C4.5; 3:CART)："))
        #ID3决策树
        if dec_tree=='1':
            labels_tmp = labels[:] # 拷贝，createTree会改变labels
            ID3desicionTree = ID3_createTree(dataSet,labels_tmp)
            print('ID3desicionTree:\n', ID3desicionTree)
            #treePlotter.createPlot(ID3desicionTree)
            treePlotter.ID3_Tree(ID3desicionTree)
            # C4.5决策树
            # C4.5决策树
        if dec_tree == '2':
            labels_tmp = labels[:]  # 拷贝，createTree会改变labels
            C45desicionTree = C45_createTree(dataSet, labels_tmp)
            print('C45desicionTree:\n', C45desicionTree)
            treePlotter.C45_Tree(C45desicionTree)
            # CART决策树
        if dec_tree == '3':
            labels_tmp = labels[:]  # 拷贝，createTree会改变labels
            dataSet_tmp = copy.deepcopy(dataSet)
            labels_tmp2 = labels[:]

            CARTdesicionTree = CART_createTree(dataSet, labels_tmp)
            #print("kkkkkkkkkkk" + str(dataSet))
            print('CARTdesicionTree:\n', CARTdesicionTree)
            print("LeafCnt: "+str(calLeaf(CARTdesicionTree)))
            treePlotter.CART_Tree(CARTdesicionTree)

            tt = pruningTree(CARTdesicionTree,dataSet_tmp,labels_tmp2)
            print("afterPrunCARTdesicionTree"+str(tt))
            treePlotter.CART_Tree(tt)
            print("具体的结果解释见实验报告")
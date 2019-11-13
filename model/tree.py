# coding=utf-8
from math import log
import copy

equalNums = lambda x,y: 0 if x is None else x[x==y].size

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for feaVec in dataSet:
        currentLabel = feaVec[-1]
        if currentLabel not in labelCounts:
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# def chooseBestFeatureToSplit(dataSet, labels):
#     numFeatures = len(dataSet[0]) - 1
#     baseEntropy = calcShannonEnt(dataSet)
#     bestInfoGain = 0.0
#     bestFeature = -1
#     bestSplitDict = {}
#     for i in range(numFeatures):
#         featList = [example[i] for example in dataSet]
#         uniqueVals = set(featList)
#         newEntropy = 0.0
#         # 计算该特征下每种划分的信息熵
#         for value in uniqueVals:
#             subDataSet = splitDataSet(dataSet, i, value)
#             prob = len(subDataSet) / float(len(dataSet))
#             newEntropy += prob * calcShannonEnt(subDataSet)
#         infoGain = baseEntropy - newEntropy
#     if infoGain > bestInfoGain:
#         bestInfoGain = infoGain
#         bestFeature = i
#     return bestFeature

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    return max(classCount)


def testing(myTree,data_test,labels):
    error=0.0
    for i in range(len(data_test)):
        if (classify(myTree,labels,data_test[i])!=data_test[i][-1]):
            error+=1
    return float(error)

def testing_feat(feat, train_data, test_data, labels):
    class_list = [example[-1] for example in train_data]
    bestFeatIndex = labels.index(feat)
    train_data = [example[bestFeatIndex] for example in train_data]
    test_data = [(example[bestFeatIndex], example[-1]) for example in test_data]
    all_feat = set(train_data)
    error = 0.0
    for value in all_feat:
        class_feat = [class_list[i] for i in range(len(class_list)) if train_data[i] == value]
        major = majorityCnt(class_feat)
        for data in test_data:
            if data[0] == value and data[1] != major:
                error += 1.0
    return error

def testingMajor(major, data_test):
    error = 0.0
    for i in range(len(data_test)):
        if major != data_test[i][-1]:
            error += 1
    return float(error)


def createTree(dataSet,labels,data_full,labels_full,test_data,mode):
    classList=[example[-1] for example in dataSet]
    if classList.count(classList[0])==len(classList):
       return classList[0]
    if len(dataSet[0])==1:
       return majorityCnt(classList)
    labels_copy = copy.deepcopy(labels)
    bestFeat=chooseBestFeatureToSplit(dataSet)
    bestFeatLabel=labels[bestFeat]

    if mode == "unpro" or mode == "post":
        myTree = {bestFeatLabel: {}}
    elif mode == "prev":
        if testing_feat(bestFeatLabel, dataSet, test_data, labels_copy) < testingMajor(majorityCnt(classList),test_data):
            myTree = {bestFeatLabel: {}}
        else:
            return majorityCnt(classList)
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)

    if type(dataSet[0][bestFeat]).__name__ == 'unicode':
        currentlabel = labels_full.index(labels[bestFeat])
        featValuesFull = [example[currentlabel] for example in data_full]
        uniqueValsFull = set(featValuesFull)

    del (labels[bestFeat])

    for value in uniqueVals:
        subLabels = labels[:]
        if type(dataSet[0][bestFeat]).__name__ == 'unicode':
            uniqueValsFull.remove(value)

        myTree[bestFeatLabel][value] = createTree(splitDataSet \
                                                      (dataSet, bestFeat, value), subLabels, data_full, labels_full,
                                                  splitDataSet \
                                                      (test_data, bestFeat, value), mode=mode)
    if type(dataSet[0][bestFeat]).__name__ == 'unicode':
        for value in uniqueValsFull:
            myTree[bestFeatLabel][value] = majorityCnt(classList)

    if mode == "post":
        if testing(myTree, test_data, labels_copy) > testingMajor(majorityCnt(classList), test_data):
            return majorityCnt(classList)
    return myTree

# def createTree(dataSet, features):
#     subfeatures=features[:]
#     classList = [example[-1] for example in dataSet]
#     if classList.count(classList[0]) == len(classList):
#         return classList[0]
#     if len(dataSet[0]) == 1:
#         return majorityCnt(classList)
#     bestFeat = chooseBestFeatureToSplit(dataSet)
#     bestFeatLabel = subfeatures[bestFeat]
#     myTree = {bestFeatLabel: {}}
#     del (subfeatures[bestFeat])
#     featValues = [example[bestFeat] for example in dataSet]
#     uniqueVals = set(featValues)
#     for value in uniqueVals:
#         myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat, value), subfeatures)
#     return myTree

def classify(inputTree,featLabels,test):
    if(isinstance(inputTree,dict) is False):
        return inputTree
    firstStr=list(inputTree.keys())[0]
    secondDict=inputTree[firstStr]
    featIndex=featLabels.index(firstStr)
    classLabel=""
    for key in secondDict.keys():
        if(test[featIndex]== key):
            if(isinstance(secondDict[key],dict)):
                classLabel=classify(secondDict[key],featLabels,test)
            else:
                classLabel=secondDict[key]

    return  classLabel




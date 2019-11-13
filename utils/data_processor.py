import numpy as np
from utils.constants import labels_dict
from utils.constants import TEST
from utils.constants import TRAIN
from utils.random_num_generator import rand_int

cla_num=[1210,384,69,65]
cur_cla_num=[0,0,0,0]
data_num=1728

def test_or_train(type):
    if(cur_cla_num[type]<cla_num[type]*0.7):
        return TRAIN
    else:
        return TEST

def attribute_and_class(dataSet):
    x_test = np.array(dataSet)[:, 0:6]
    y_test=np.array(dataSet)[:,6:7]
    return x_test.tolist(),y_test.tolist()

def generate_random_data(dataSet):
    iterate_num=500
    for i in range(iterate_num):
        pos1=rand_int(1,1727)
        pos2=rand_int(1,1727)
        if(pos1==pos2): continue
        dataSet[pos1],dataSet[pos2]=dataSet[pos2],dataSet[pos1]
    return dataSet

def split_data(dataSet):
    x_test=[]
    x_train=[]
    dataSet=generate_random_data(dataSet)
    for data in dataSet:
       label=labels_dict[data[-1]]
       if(test_or_train(label)==TRAIN):
           x_train.append(data)
       else:
           x_test.append(data)
       cur_cla_num[label]=cur_cla_num[label]+ 1

    x_test,y_test=attribute_and_class(x_test)
    #x_train,y_train=attribute_and_class(x_train)

    return x_train,x_test,y_test

def split_data_intentionally(dataSet):
    x_test = []
    x_train = []
    train_num=0
    data_distribution=[0,0,0,0]
    dataSet = generate_random_data(dataSet)
    for data in dataSet:
        if(train_num<1728*0.7):
            label = labels_dict[data[-1]]
            data_distribution[label]+=1
            x_train.append(data)
        else:
            x_test.append(data)
    x_test, y_test = attribute_and_class(x_test)
    # x_train,y_train=attribute_and_class(x_train)
    return data_distribution,x_train, x_test, y_test

def load_data(filename):
    file=open(filename)
    data_list=file.readlines()
    dataSet=[]

    for data in data_list:
        data=data.strip('\n').split(',')
        dataSet.append(data)

    return dataSet
from model.tree import  createTree
from model.tree import classify
from model import bpnn
from utils.data_processor import load_data
from utils.data_processor import split_data
from utils import feature_mapper
import numpy as np
import time


def get_mapped_data(x_train,x_test,y_test):
    y_train = np.array(x_train)[:, 6:7]
    x_train = np.array(x_train)[:, 0:6]
    mapped_x_train = []
    mapped_y_train = []
    mapped_x_test = []
    mapped_y_test = []

    for x in x_train:
        mapped_x_train.append(feature_mapper.map_value(x))
    for y in y_train:
        mapped_y_train.append(feature_mapper.map_value(y))
    for x in x_test:
        mapped_x_test.append(feature_mapper.map_value(x))
    for y in y_test:
        mapped_y_test.append(feature_mapper.map_value(y))

    return mapped_x_train,mapped_y_train,mapped_x_test,mapped_y_test

def test_normal_data_tree(dataSet,featlabels):
    x_train, x_test, y_test = split_data(dataSet)
    data_full=x_train[:]
    labels_full=featlabels[:]
    labels=featlabels[:]
    classify_res=0
    error_classify_res={'unacc':0,'acc':0,'good':0,'vgood':0,'':0}
    error_original_res = {'unacc': 0, 'acc': 0, 'good': 0, 'vgood': 0,'':0}
    sum_count = len(x_test)
    my_tree = createTree(x_train, labels,data_full,labels_full,x_test,mode='post')
    print("generated decision tree is ", my_tree)
    tree_correct_count = 0
    print("decision tree classify results are: ")
    for index,x in enumerate(x_test):
        res=classify(my_tree,featlabels,x)
        # print(res,y_test[index])
        if(res==y_test[index][0]):
            tree_correct_count+=1
        else:
            print("error classify result:"+res+" correct result is ",y_test[index])
            error_original_res[y_test[index][0]]+=1
            error_classify_res[res]+=1
    classify_res=tree_correct_count/sum_count

    return classify_res,error_classify_res,error_original_res

def test_acc_data_tree(dataSet,featlabels):
    x_train, x_test, y_test = split_data(dataSet)
    data_full=x_train[:]
    labels_full=featlabels[:]
    labels=featlabels[:]
    classify_res=0
    error_classify_res={'unacc':0,'acc':0,'good':0,'vgood':0,'':0}
    error_original_res = {'unacc': 0, 'acc': 0, 'good': 0, 'vgood': 0,'':0}
    sum_count = len(x_test)
    my_tree = createTree(x_train, labels,data_full,labels_full,x_test,mode='post')
    print("generated decision tree is ", my_tree)
    tree_correct_count = 0
    print("decision tree classify results are: ")
    for index,x in enumerate(x_test):
        res=classify(my_tree,featlabels,x)
        # print(res,y_test[index])
        if(res==y_test[index][0]):
            tree_correct_count+=1
        else:
            print("error classify result:"+res+" correct result is ",y_test[index])
            error_original_res[y_test[index][0]]+=1
            error_classify_res[res]+=1
    classify_res=tree_correct_count/sum_count

    return classify_res,error_classify_res,error_original_res

def test_normal_bpnn(dataSet,rate):
    x_train, x_test, y_test = split_data(dataSet)
    bpnn_correct_count = 0
    sum_count = len(x_test)
    my_network = bpnn.BPNNet(12, 10, 2)
    mapped_x_train, mapped_y_train, mapped_x_test, mapped_y_test = get_mapped_data(x_train, x_test, y_test)
    my_network.train(mapped_x_train, mapped_y_train, N=rate,show_error=True)
    my_network.save_weights("res/network_weight")
    # my_network.load_weights("res/network_weight")
    print("bp network classify results are: ")
    for j, x in enumerate(mapped_x_test):
        res = feature_mapper.formatting(my_network.test(x))
        if ((res == mapped_y_test[j])): bpnn_correct_count += 1
        else:
            print(mapped_y_test[j], res)
    correct_rate=bpnn_correct_count/sum_count
    return correct_rate

def main():
    dataSet=load_data('res/car.txt')
    featlabels = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

    # tree_classify_res,tree_error_res,tree_orignal_res=test_normal_data_tree(dataSet,featlabels)
    # print(tree_classify_res)
    # print(tree_error_res)
    # print(tree_orignal_res)
    bpnn_classify_res=test_normal_bpnn(dataSet,0.3)
    print(bpnn_classify_res)

if __name__ == '__main__':
    main()
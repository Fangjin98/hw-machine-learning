from model.tree import  createTree
from model.tree import classify
from model import bpnn
from utils.data_processor import load_data
from utils.data_processor import split_data
from utils import feature_mapper
import numpy as np


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


def main():
    dataSet=load_data('res/car.txt')
    x_train,x_test,y_test=split_data(dataSet)
    featlabels=['buying','maint','doors','persons','lug_boot','safety']

    tree_correct_count=0
    bpnn_correct_count = 0
    sum_count=len(x_test)

    my_tree = createTree(x_train,featlabels)
    print("generated decision tree is ",my_tree)
    print("decision tree classify results are: ")
    for index,x in enumerate(x_test):
        res=classify(my_tree,featlabels,x)
        print(res,y_test[index])
        if(res==y_test[index]):
            tree_correct_count+=1

    my_network = bpnn.BPNNet(12, 20, 10, 2)
    mapped_x_train, mapped_y_train, mapped_x_test, mapped_y_test=get_mapped_data(x_train,x_test,y_test)
    my_network.train(mapped_x_train, mapped_y_train, N=0.1, M=0.1)
    my_network.save_weights("res/network_weight")
    #my_network.load_weights("res/network_weight")

    print("bp network classify results are: ")
    for j, x in enumerate(mapped_x_test):
        res = feature_mapper.formatting(my_network.classify(x))
        print(mapped_y_test[j], res)
        if ((res == mapped_y_test[j])): bpnn_correct_count += 1


    print("decision tree correct rate=", '%.4f' % (tree_correct_count / sum_count))
    print("bp network correct rate=", "%.4f" % (bpnn_correct_count / sum_count))

if __name__ == '__main__':
    main()
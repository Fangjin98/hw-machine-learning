import math

def sigmoid(x):
    return math.tanh(x)

def dsigmoid(y):
    return 1.0 - y**2
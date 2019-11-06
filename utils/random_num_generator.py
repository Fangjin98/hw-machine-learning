import random

random.seed(0)

def rand(a, b):
    return (b-a)*random.random() + a

def rand_int(a,b):
    return random.randint(a,b)
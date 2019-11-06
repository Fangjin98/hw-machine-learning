from utils.activations import sigmoid
from utils.activations import dsigmoid
from utils.random_num_generator import rand
try:
    import cPickle as pickle
except:
    import pickle

class Unit:
    def __init__(self, length):
        self.weight = [rand(0, 1) for i in range(length)]
        self.change = [0.0] * length
        self.threshold = rand(0, 1)

    def calc(self, sample):
        self.sample = sample[:]
        tmp = sum([i * j for i, j in zip(self.sample, self.weight)]) - self.threshold
        self.output = sigmoid(tmp)
        return self.output

    def update(self, diff, rate=0.5, factor=0.1):
        change = [rate * x * diff + factor * c for x, c in zip(self.sample, self.change)]
        self.weight = [w + c for w, c in zip(self.weight, change)]
        self.change = [x * diff for x in self.sample]
        #self.threshold = rateN * factor + rateM * self.change_threshold + self.threshold
        #self.change_threshold = factor

    def get_weight(self):
        return self.weight[:]

    def set_weight(self, weight):
        self.weight = weight[:]

class Layer:
    def __init__(self, input_length, output_length):
        self.units = [Unit(input_length) for i in range(output_length)]
        self.output = [0.0] * output_length
        self.len = input_length

    def calc(self, sample):
        self.output = [unit.calc(sample) for unit in self.units]
        return self.output[:]

    def update(self, diffs, rate=0.5, factor=0.1):
        for diff, unit in zip(diffs, self.units):
            unit.update(diff, rate, factor)

    def get_error(self, deltas):
        def _error(deltas, j):
            return sum([delta * unit.weight[j] for delta, unit in zip(deltas, self.units)])
        return [_error(deltas, j) for j  in range(self.len)]

    def get_weights(self):
        weights = {}
        for key, unit in enumerate(self.units):
            weights[key] = unit.get_weight()
        return weights

    def set_weights(self, weights):
        for key, unit in enumerate(self.units):
            unit.set_weight(weights[key])

class BPNNet:
    def __init__(self, input_num, hidden_num1,hidden_num2, output_num):
        self.input_num = input_num + 1 # +1 for bias node
        self.hidden_num1 = hidden_num1
        self.hidden_num2=hidden_num2
        self.output_num = output_num
        self.hidden_layer1 = Layer(self.input_num, self.hidden_num1)
        self.hidden_layer2=Layer(self.hidden_num1,self.hidden_num2)
        self.output_layer = Layer(self.hidden_num2, self.output_num)

    def calc(self, inputs):
        if len(inputs) != self.input_num-1:
            raise ValueError('wrong number of inputs')

        self.input_activations = inputs[:] + [1.0]
        self.hidden_activations1 = self.hidden_layer1.calc(self.input_activations)
        self.hidden_activations2 = self.hidden_layer2.calc(self.hidden_activations1)
        self.output_activations = self.output_layer.calc(self.hidden_activations2)

        return self.output_activations[:]


    def update(self, targets, rate, factor):
        if len(targets) != self.output_num:
            raise ValueError('wrong number of target values')

        output_deltas = [dsigmoid(ao) * (target - ao) for target, ao in zip(targets, self.output_activations)]
        hidden_deltas2 = [dsigmoid(ah) * error for ah, error in zip(self.hidden_activations2, self.output_layer.get_error(output_deltas))]
        hidden_deltas1 = [dsigmoid(ah) * error for ah, error in zip(self.hidden_activations1, self.hidden_layer2.get_error(hidden_deltas2))]
        self.output_layer.update(output_deltas, rate, factor)
        self.hidden_layer2.update(hidden_deltas2, rate, factor)
        self.hidden_layer1.update(hidden_deltas1, rate, factor)

        return sum([0.5 * (t-o)**2 for t, o in zip(targets, self.output_activations)])


    def classify(self, pattern):
        return self.calc(pattern)

    def train(self, features,types, iterations=1000, N=0.5, M=0.1,show_error=False):
        for i in range(iterations):
            error = 0.0
            for j,feature in enumerate(features):
                inputs = feature
                targets = types[j]
                self.calc(inputs)
                error = error + self.update(targets, N, M)
            if(show_error):
                if (i % 100 == 0): print(error)

    def save_weights(self, filename):
        weights = {
            "output_layer": self.output_layer.get_weights(),
            "hidden_layer1": self.hidden_layer1.get_weights(),
            "hidden_layer2": self.hidden_layer2.get_weights()
        }

        with open(filename, "wb") as f:
            pickle.dump(weights, f)

    def load_weights(self, fn):
        with open(fn, "rb") as f:
            weights = pickle.load(f)
            self.output_layer.set_weights(weights["output_layer"])
            self.hidden_layer1.set_weights(weights["hidden_layer1"])
            self.hidden_layer2.set_weights(weights["hidden_layer2"])
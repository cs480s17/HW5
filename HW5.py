import struct
import random
import numpy as np
Data = []
import sys
import copy
#import mnist
Epochs = 100
TrainingPortion = 0.9


class Perceptron:
    def __init__(self, num_inputs, eta):
        self.weights = [0.0]*(num_inputs + 1)  # +1 is for W0, the threshold weight
        for i in range(num_inputs + 1):  # randomize all weights
            self.weights[i] = random.uniform(-0.1, 0.1)
        self.eta = eta

    def copy(self):
        p = Perceptron(len(self.weights)-1)
        for i in range(len(self.weights)):
            p.weights[i] = self.weights[i]
        return p

    def learn(self, inputs, correct_output):  # useful for training single perceptron, not so much for neural net
        prediction = self.predict(inputs)
        if prediction * correct_output <= 0:  # if prediction and the correct output have a different sign:
            for i in range(len(self.weights)):
                new = self.weights[i] - (self.eta * (prediction - correct_output) * inputs[i])
                self.weights[i] = new
        else:
            return

    def train(self, inputs, error):  # used by neural net
        for i in range(len(self.weights)):
            new = self.weights[i] + (self.eta * error * inputs[i])
            self.weights[i] = new
        return

    def test(self, inputs, correct_output):
        prediction = self.predict(inputs)
        return not(prediction * correct_output <= 0)
        # if prediction and the correct output have a different sign, return false

    def predict(self, inputs): #remember to always have threshold input as -1
        total = 0.0
        for i in range(len(self.weights)):
            total += self.weights[i] * inputs[i]
        return total

    def listWeights(self):
        for i in range(len(self.weights)):
            sys.stdout.write("w" + str(i) + " = " + str(self.weights[i]) + " ")
        sys.stdout.flush()
        return


class Numimg:
    def __init__(self, data = ""):
        self.label = data[0]
        temp = data[1:].split("\n")  # like firewood
        self.Array = []
        for i in temp:
            self.Array.append(i.split(" ")[1:])  # because
        self.rows = len(self.Array)
        self.cols = len(self.Array[0])
        num_ones = self.num_ones()
        self.density = num_ones / (self.rows * self.cols)
        self.h_symmetry = self.__h_symmetry__() / num_ones
        self.v_symmetry = self.__v_symmetry__() / num_ones
        self.min_h_intercepts, self.max_h_intercepts = self.__h_intercepts__()
        self.min_v_intercepts, self.max_v_intercepts = self.__v_intercepts__()

    def print(self):
        print("Label:", self.label)
        print("density:", self.density)
        print("h_symmetry:", self.h_symmetry)
        print("v_symmetry:", self.v_symmetry)
        print("h intercepts(min,max):", self.min_h_intercepts, self.max_h_intercepts)
        print("v intercepts(min,max):", self.min_v_intercepts, self.max_v_intercepts)
        for i in self.Array:
            print(i)

    def num_ones(self):
        counter = 0
        for i in self.Array:
            for j in i:
                if(j == '1'):
                    counter += 1

        return counter

    def __h_symmetry__(self):
        temp = 0
        for i in range(self.rows//2):
            for j in range(self.cols//2):
                if(self.Array[i][j] != self.Array[i][self.cols-(j+1)]):
                    temp += 1
        return temp

    def __v_symmetry__(self):
        temp = 0
        for i in range(self.rows//2):
            for j in range(self.cols//2):
                if(self.Array[i][j] != self.Array[self.rows-(i+1)][j]):
                    temp += 1
        return temp

    def __h_intercepts__(self):
        min = sys.maxsize
        max = 0
        for i in range(self.rows):
            count = 0
            prev = '0'
            for j in range(self.cols):  # following will count the number of 1 to 0 borders in current row
                curr = self.Array[i][j]
                if prev != curr:
                    if prev == '1':
                        count += 1
                    prev = curr
            if count < min:
                min = count
            if count > max:
                max = count
        return min, max

    def __v_intercepts__(self):
        min = sys.maxsize
        max = 0
        for i in range(self.cols):
            count = 0
            prev = '0'
            for j in range(self.rows):  # following will count the number of 1 to 0 borders in current col
                curr = self.Array[j][i]
                if prev != curr:
                    if prev == '1':
                        count += 1
                    prev = curr
            if count < min:
                min = count
            if count > max:
                max = count
        return min, max

    def inputs(self):
        return [-1,
                self.density,
                self.h_symmetry,
                self.v_symmetry,
                self.min_h_intercepts,
                self.max_h_intercepts,
                self.min_v_intercepts,
                self.max_v_intercepts]


def parseData():
    global Data
    global cols, rows

    print("Opening Test Data File")
    f = open('testdata', 'r')
    print("Reading File to String")
    tempdata = f.read()
    print("Splitting into Samples")
    dataarray = tempdata.split("\n\n\n") #yep
    dataarray.pop()
    for i in dataarray:
        Data.append(Numimg(i)) #you know it


def printData():
    global Data
    for i in Data:
        i.print()


class NeuralNet:
    def __init__(self, inputs, input_layer_size, hidden_layer_size, outputs, eta):
        self.num_inputs = inputs
        self.L1_size = input_layer_size
        self.L2_size = hidden_layer_size
        self.L3_size = outputs
        self.L1 = []
        self.L2 = []
        self.L3 = []
        for i in range(input_layer_size):
            self.L1.append(Perceptron(inputs))
        for i in range(hidden_layer_size):
            self.L2.append(Perceptron(input_layer_size))
        for i in range(outputs):
            self.L3.append(Perceptron(hidden_layer_size))

    def run_forward(self, input, isTraining):
        if len(input) != self.num_inputs:
            print("Wrong number of inputs for neural network, expected", self.num_inputs, "got", len(input))
        L1_outputs = [0.0]*self.L1_size
        L2_outputs = [0.0]*self.L2_size
        L3_outputs = [0.0]*self.L3_size
        for i in range(self.L1_size):
            L1_outputs[i] = self.L1[i].predict(input)
        for i in range(self.L2_size):
            L2_outputs[i] = self.L2[i].predict(L1_outputs)
        for i in range(self.L3_size):
            L3_outputs[i] = self.L3[i].predict(L2_outputs)
        if not isTraining:
            return L3_outputs
        else:
            return L1_outputs, L2_outputs, L3_outputs

    def run_backwards(self, input, L1_outputs, L2_outputs, L3_outputs, expected):
        # compute error in each output neuron
        L3_error = [0.0]*self.L3_size
        L2_error = [0.0]*self.L2_size
        L1_error = [0.0]*self.L1_size

        for i in range(self.L3_size):
            L3_error[i] = (expected[i] - L3_outputs[i])*L3_outputs[i]*(1-L3_outputs[i])

        for i in range(self.L2_sizeL):
            sum = 0.0
            for j in range(self.L3_size):
                sum += self.L3[j].weights[i] * L3_error[j]
            L2_error[i] = (L2_outputs[i] * sum)

        for i in range(self.L1_size):  # identical to above
            sum = 0.0
            for j in range(self.L2_size):
                sum += self.L2[j].weights[i] * L2_error[j]
            L1_error[i] = (L1_outputs[i] * sum)

        # should have error for everything now. Now to update values
        for i in range(self.L3_size):  # update output layer
            self.L3[i].train(L2_outputs, L3_error)
        for i in range(self.L2_size):  # update hidden layer
            self.L2[i].train(L1_outputs, L2_error)
        for i in range(self.L1_size):  # update input layer
            self.L1[i].train(input, L1_error)

    def train(self, input, expectedout):
        L1out, L2out, L3out = self.run_forward(input, True)
        self.run_backwards(input, L1out, L2out, L3out, expectedout)

    def test(self, input):
        output = self.run_forward(input, False)
        return output

def main():
    #global Epochs
    #global TrainingPortion
    #global Data
    #ErrorArray = []
    #PArray = []
    #parseData()
    #TrainingSize = int(len(Data) * TrainingPortion)
    #BestP = None
    #BestError = sys.maxsize
    #CurrP = Perceptron(7, 0.05)
    #CurrError = 0
    #for i in range(TrainingSize, len(Data)):
    #    correct_out = (-1 if Data[i].label == '5' else 1)
    #    if not (CurrP.test(Data[i].inputs(), correct_out)):
    #        CurrError += 1
    #ErrorArray.append(CurrError)
    #PArray.append(CurrP.copy())
    #if CurrError < BestError:
    #    BestError = CurrError
    #    BestP = CurrP.copy()
    #for e in range(Epochs):
    #    CurrError = 0
    #    for i in range(TrainingSize):
    #        correct_out = (-1 if Data[i].label == '5' else 1)
    #        CurrP.learn(Data[i].inputs(), correct_out)
#
    #    for i in range(TrainingSize, len(Data)):
    #        correct_out = (-1 if Data[i].label == '5' else 1)
    #        if not(CurrP.test(Data[i].inputs(), correct_out)):
    #            CurrError += 1
    #    ErrorArray.append(CurrError)
    #    PArray.append(CurrP.copy())
    #    if CurrError < BestError:
    #        BestError = CurrError
    #        BestP = CurrP.copy()
    #print("The best perceptron of", Epochs, "epochs has the following weights, where w0 is the threshold")
    #BestP.listWeights()
    #print()
    #print("This results in", BestError, "Errors out of", len(Data) - TrainingSize, "Test Cases")
    #print("Which is an accuracy of", round(1 - (BestError/(len(Data) - TrainingSize)), 2))
    #return 0
    array = getData()
    print("hell")


def getData():

    # get training set
    with open('train-labels-idx1-ubyte','rb') as fl:
        magic,num = struct.unpack(">II",fl.read(8))
        label = np.fromfile(fl,dtype=np.int8)
    with open('train-images-idx3-ubyte','rb') as fi:
        magic, num, rows, cols = struct.unpack(">IIII",fi.read(16))
        image = np.fromfile(fi,dtype=np.uint8).reshape(len(label),rows,cols)
        get_image = lambda idx: (label[idx],image[idx])
    ret = []
    for i in range(len(label)):
        ret.append(get_image(i))
    return ret

if __name__ == "__main__":
    main()

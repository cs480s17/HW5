
import struct
import random
import numpy as np
Data = []
import sys
import copy
#import mnist
Epochs = 50
TrainingPortion = 0.9
MnistSize = 1000
Eta = 0.15
HiddenLayerSize = 50

class Perceptron:
    def __init__(self, num_inputs):
        self.weights = [0.0]*(num_inputs + 1)  # +1 is for W0, the threshold weight
        for i in range(num_inputs + 1):  # randomize all weights
            self.weights[i] = random.uniform(-0.1, 0.1)


    def copy(self):
        p = Perceptron(len(self.weights)-1)
        for i in range(len(self.weights)):
            p.weights[i] = self.weights[i]
        return p

    #def learn(self, inputs, correct_output):  # useful for training single perceptron, not so much for neural net
    #    prediction = self.predict(inputs)
    #    if prediction * correct_output <= 0:  # if prediction and the correct output have a different sign:
    #        for i in range(len(self.weights)):
    #            new = self.weights[i] - (0.15 * (prediction - correct_output) * inputs[i])
    #            self.weights[i] = new
    #    else:
    #        return

    def train(self, inputs, error, eta):  # used by neural net
        for i in range(len(self.weights)):
            new = self.weights[i] + (eta * error * inputs[i])
            self.weights[i] = new
        return

    #def test(self, inputs, correct_output):
    #    prediction = self.predict(inputs)
    #    return not(prediction * correct_output <= 0)
    #    # if prediction and the correct output have a different sign, return false

    def predict(self, inputs): #remember to always have threshold input as -1
        total = 0.0
        for i in range(len(self.weights)):
            total += self.weights[i] * inputs[i]
        sigmoid = 1/(1 + np.exp(-total))
        return sigmoid


    def listWeights(self):
        for i in range(len(self.weights)):
            sys.stdout.write("w" + str(i) + " = " + str(self.weights[i]) + " ")
        sys.stdout.flush()
        return


class Numimg:
    def __init__(self, label, array):
        #self.label = data[0]
        #temp = data[1:].split("\n")  # like firewood
        #self.Array = []
        #for i in temp:
        #    self.Array.append(i.split(" ")[1:])  # because
        self.label = label
        self.Array = array
        self.rows = len(self.Array)
        self.cols = len(self.Array[0])
        num_ones = self.num_ones()
        self.density = num_ones / (self.rows * self.cols)
        self.h_symmetry = self.__h_symmetry__() / num_ones
        self.v_symmetry = self.__v_symmetry__() / num_ones
        self.max_h_intercepts = self.__h_intercepts__()
        self.max_v_intercepts = self.__v_intercepts__()
        self.__toBinaryArray__()
        self.gridIntensities = self.__gridIntensities__()

    #def print(self):
    #    print("Label:", self.label)
    #    print("density:", self.density)
    #    print("h_symmetry:", self.h_symmetry)
    #    print("v_symmetry:", self.v_symmetry)
    #    print("h intercepts(max):", self.max_h_intercepts)
    #    print("v intercepts(max):", self.max_v_intercepts)
    #    for i in self.Array:
    #        print(i)

    def __toBinaryArray__(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if self.Array[i][j] != 0:
                    self.Array[i][j] = 1

    def num_ones(self):
        counter = 0
        for i in self.Array:
            for j in i:
                if(j != 0):
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
            prevIs0 = True  # prev = '0'
            for j in range(self.cols):  # following will count the number of 1 to 0 borders in current row
                curr = self.Array[i][j]
                if prevIs0 != (curr == 0):  # if prev != curr:
                    if not prevIs0:
                        count += 1
                    prevIs0 = (curr == 0)  # True if curr == 0
            if count < min:
                min = count
            if count > max:
                max = count
        return max

    def __v_intercepts__(self):
        min = sys.maxsize
        max = 0
        for i in range(self.cols):
            count = 0
            prevIs0 = True  # prev = '0'
            for j in range(self.rows):  # following will count the number of 1 to 0 borders in current col
                curr = self.Array[j][i]
                if prevIs0 != (curr == 0):
                    if not prevIs0:
                        count += 1
                    prevIs0 = (curr == 0)  # True if curr == 0
            if count < min:
                min = count
            if count > max:
                max = count
        return max

    def __gridIntensities__(self):
        if(self.rows != 28 or self.cols != 28):
            print('MNIST data has wrong number of rows and columns. Expected 28x28, instead got '
                  + str(self.rows) + 'x' + str(self.cols))
            return []
        gridIntensities = []
        for gi in range(4):
            for gj in range(4):
                total = 0
                for i in range(7):
                    for j in range(7):
                        total += self.Array[7*gi + i][7*gj + j]
                gridIntensities.append(total)
        return gridIntensities

    def inputs(self):
        features = [-1, self.density, self.h_symmetry, self.v_symmetry, self.max_h_intercepts, self.max_v_intercepts]
        features.extend(self.gridIntensities)
        return features
        #return self.gridIntensities


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


#def printData():
#    global Data
#    for i in Data:
#        i.print()


class NeuralNet:
    def __init__(self, inputs=0, input_layer_size=0, hidden_layer_size=0, outputs=0):
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

    def copy(self):
        #nn = NeuralNet(self.num_inputs, self.L1_size, self.L2_size, self.L3_size, self.eta)
        nn = NeuralNet()
        for i in self.L1:
            nn.L1.append(i.copy())
        for i in self.L2:
            nn.L2.append(i.copy())
        for i in self.L3:
            nn.L3.append(i.copy())

        nn.num_inputs = self.num_inputs
        nn.L1_size = self.L1_size
        nn.L2_size = self.L2_size
        nn.L3_size = self.L3_size
        return nn

    def run_forward(self, input, isTraining):
        if len(input) != self.num_inputs + 1:
            print("Wrong number of inputs for neural network, expected", self.num_inputs, "got", len(input))
        L1_outputs = [0.0]*self.L1_size
        L2_outputs = [0.0]*self.L2_size
        L3_outputs = [0.0]*self.L3_size
        for i in range(self.L1_size):
            L1_outputs[i] = self.L1[i].predict(input)
        L1_outputs.insert(0, -1)
        for i in range(self.L2_size):
            L2_outputs[i] = self.L2[i].predict(L1_outputs)
        L2_outputs.insert(0, -1)
        for i in range(self.L3_size):
            L3_outputs[i] = self.L3[i].predict(L2_outputs)
        if not isTraining:
            return L3_outputs
        else:
            return L1_outputs, L2_outputs, L3_outputs

    def run_backwards(self, input, L1_outputs, L2_outputs, L3_outputs, expected, eta):
        # compute error in each output neuron
        L3_error = [0.0]*self.L3_size
        L2_error = [0.0]*self.L2_size
        L1_error = [0.0]*self.L1_size

        for i in range(self.L3_size):
            #if():  # output is incorrect
            L3_error[i] = (expected[i] - L3_outputs[i])*L3_outputs[i]*(1-L3_outputs[i])


        for i in range(self.L2_size):
            sum = 0.0
            for j in range(self.L3_size):
                sum += self.L3[j].weights[i+1] * L3_error[j]
            L2_error[i] = (L2_outputs[i+1] * sum)

        for i in range(self.L1_size):  # identical to above
            sum = 0.0
            for j in range(self.L2_size):
                sum += self.L2[j].weights[i+1] * L2_error[j]
            L1_error[i] = (L1_outputs[i+1] * sum)

        # should have error for everything now. Now to update values
        for i in range(self.L3_size):  # update output layer
            self.L3[i].train(L2_outputs, L3_error[i], eta)
        for i in range(self.L2_size):  # update hidden layer
            self.L2[i].train(L1_outputs, L2_error[i], eta)
        for i in range(self.L1_size):  # update input layer
            self.L1[i].train(input, L1_error[i], eta)

    def train(self, inputs, expectedout, eta):
        L1out, L2out, L3out = self.run_forward(inputs, True)
        self.run_backwards(inputs, L1out, L2out, L3out, expectedout, eta)

    def test(self, input):
        output = self.run_forward(input, False)
        return output

def main():
    global MnistSize
    global Epochs
    global TrainingPortion
    global Eta
    global HiddenLayerSize
    if len(sys.argv) == 2:
        HiddenLayerSize = int(sys.argv[1])
    dataArray = getData()

    dataSize = MnistSize #len(dataArray)
    trainingSize = int(dataSize * TrainingPortion)
    testingSize = dataSize - trainingSize
    trainingArray = []
    testingArray = []
    print('extracting features for training set')
    for i in range(0,trainingSize):
        label = dataArray[i][0]
        array = dataArray[i][1]
        trainingArray.append(Numimg(label, array))
    print('extracting features for test set')
    for i in range(trainingSize, dataSize):
        label = dataArray[i][0]
        array = dataArray[i][1]
        testingArray.append(Numimg(label, array))
    thing = trainingArray[0].inputs()
    numFeatures = len(thing) - 1
    CurrNeuralNet = NeuralNet(numFeatures, HiddenLayerSize, HiddenLayerSize, 10)
    ErrorArray = []
    NetArray = [] #temp
    BestNeuralNet = CurrNeuralNet.copy()
    BestError = sys.maxsize #number it got wrong
    BestEpoch = 0
    rounds = 0
    while True:
        for epoch in range(Epochs):
            print("Begining epoch", Epochs * rounds + epoch)
            random.shuffle(trainingArray)  # shuffling order in which training takes place per ravi's recommendation
            for i in range(trainingSize):  # training loop
                expectedOutput = [0]*10
                expectedOutput[trainingArray[i].label] = 1
                CurrNeuralNet.train(trainingArray[i].inputs(), expectedOutput, Eta)
            CurrError = 0
            for i in range(testingSize):  # testing loop
                outputs = CurrNeuralNet.test(testingArray[i].inputs())
                maxval = 0.0
                maxindex = 0
                for j in range(10):
                    if outputs[j] > maxval:
                        maxval = outputs[j]
                        maxindex = j
                if testingArray[i].label != maxindex:
                    CurrError += 1
                #for j in range(10):
                #    if testingArray[i].label == j and outputs[j] <= 0.5:
                #        CurrError += 1
                #        break
                #    if testingArray[i].label != j and outputs[j] > 0.5:
                #        CurrError += 1
                #        break
            print('Error is', CurrError)
            if CurrError < BestError:
                BestError = CurrError
                BestNeuralNet = CurrNeuralNet.copy()
                BestEpoch = Epochs * rounds + epoch
            ErrorArray.append(CurrError)
            NetArray.append(CurrNeuralNet.copy())
        print("Best Error is", BestError, "out of", testingSize, "at Epoch", BestEpoch)
        print("That's an accuracy of", str(100*(1.0-float(BestError)/testingSize))+"%")
        input("continue?")
        rounds += 1






def getData():

    # get training set
    with open('train-labels-idx1-ubyte','rb') as fl:
        magic,num = struct.unpack(">II",fl.read(8))
        label = np.fromfile(fl,dtype=np.int8)
    with open('train-images-idx3-ubyte','rb') as fi:
        magic, num, rows, cols = struct.unpack(">IIII",fi.read(16))
        image = np.fromfile(fi,dtype=np.uint8).reshape(len(label), rows, cols)
        get_image = lambda idx: (label[idx],image[idx])
    ret = []
    for i in range(len(label)):
        ret.append(get_image(i))
    return ret

if __name__ == "__main__":
    try:
        input = raw_input
    except NameError:
        pass
    main()

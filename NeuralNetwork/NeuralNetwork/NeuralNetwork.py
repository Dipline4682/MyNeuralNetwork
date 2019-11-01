import cv2
import random
import numpy as np

def CreatWeights(l):#h, w):
    mas = []
    for j in range(l):
        #mas.append(random.randint(a, b)) #Рандом Bias
        mas.append(float("{0:.1f}".format(random.uniform(-5.0, 5.0))))#[random.uniform(0.0, 2.0) for j in range(w)])
        #mas.append([float("{0:.1f}".format(random.uniform(-2.0, 2.0))) for i in range(l)])#[random.uniform(0.0, 2.0) for j in range(w)])
    #print(len(mas))
    #mas = np.array(mas)
    return mas

def SetWeightFile(SaveWeight):
    with open('bias_weights.txt', 'w') as filehandle:  
        for listitem in SaveWeight:
            filehandle.write('%s\n' % listitem)

def GetWeightFile (puty):
    if puty == 'bias_weights':
    # определим пустой список
        weights = []
        weights.clear()
        input_weights = []
        output_weights = []
        # откроем файл и считаем его содержимое в список
        with open('bias_weights.txt', 'r') as filehandle:  
            for line in filehandle:
                # удалим заключительный символ перехода строки
                currentPlace = line[:-1]
                # добавим элемент в конец списка
                weights.append(float(currentPlace))
        weights = np.array(weights)
        return weights.mean()
    elif puty == 'input_weights':
    # определим пустой список
        weights = []
        input_weights = []
        input_weights.clear()
        output_weights = []
        # откроем файл и считаем его содержимое в список
        with open('input_weights.txt', 'r') as filehandle:  
            for line in filehandle:
                # удалим заключительный символ перехода строки
                currentPlace = line[:-1]
                # добавим элемент в конец списка
                weights.append(float(currentPlace))
        input_weights = np.reshape(weights, (100, 100))
        return input_weights

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    def feedforward(self, inputs, l):
        input_layer = []
        input_layer.clear()
        for i in range(l):
            input_layer.append(np.dot(self.weights[i], inputs) + self.bias)
        #print(input_layer)
        input_layer = np.array(input_layer)
        print(input_layer.mean())
        #print(input_layer)
        return sigmoid(input_layer)
 
class OurNeuralNetwork:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        #self.h1 = Neuron(weights, bias)
        #self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(np.array([random.uniform(0, 10), random.uniform(0, 10)]), bias)
    def feedforward(self, x):
        #out_h1 = self.h1.feedforward(x)
        #out_h2 = self.h2.feedforward(x)
        #out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))
        return out_o1


img = cv2.imread('2.png', 0)
h, w = img.shape
cv2.imshow('qwe', img)

img1d = []
for i  in range(h):
    for j in range(w):
        img1d.append(img[i][j])

# Передача веса и смещения в нейрон
neuron = Neuron(GetWeightFile('input_weights'), GetWeightFile('bias_weights'))
print(neuron.feedforward(img1d, h*w))


#SetWeightFile(CreatWeights(h*w)) #Генерация и запись в файл случайных значений для весов
#print(GetWeightFile('input_weights')) # Считывание с файла массива весов //bias_weights //input_weights
#print(CreatWeights(h*w))
#network = OurNeuralNetwork(weights, bias)
#print(network.feedforward(mas))
#random.uniform(0, 10)
#w = w.reshape((10, 10))
#print(w)
cv2.waitKey(0)
cv2.destroyAllWindows()

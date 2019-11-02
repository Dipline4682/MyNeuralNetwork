import cv2
import random
import numpy as np

def CreatWeights(l):#h, w):
    mas = []
    for j in range(l):
        mas.append(np.random.normal()) #Рандом Bias
        #mas.append(float("{0:.1f}".format(random.uniform(-1.0, 1.0))))#[random.uniform(0.0, 2.0) for j in range(w)])
        #mas.append([float("{0:.1f}".format(random.uniform(-2.0, 2.0))) for i in range(l)])#[random.uniform(0.0, 2.0) for j in range(w)])
    #print(len(mas))
    #mas = np.array(mas)
    return mas

def SetWeightFile(SaveWeight):
    with open('output_bias_weight.txt', 'w') as filehandle:  
        for listitem in SaveWeight:
            filehandle.write('%s\n' % listitem)

def GetWeightFile (puty):
    # Загрузка выходных весов
    if puty == 'output_weights':
    # определим пустой список
        output_weights = []
        output_weights.clear()
        # откроем файл и считаем его содержимое в список
        with open('output_weights.txt', 'r') as filehandle:  
            for line in filehandle:
                # удалим заключительный символ перехода строки
                currentPlace = line[:-1]
                # добавим элемент в конец списка
                output_weights.append(float(currentPlace))
        #weights = np.array(weights)
        output_weights = np.reshape(output_weights, (10, 100))
        return output_weights
    # Загрузка входных весов смещения
    elif puty == 'input_bias_weight':
    # определим пустой список
        input_bias_weight = []
        input_bias_weight.clear()
        # откроем файл и считаем его содержимое в список
        with open('input_bias_weight.txt', 'r') as filehandle:  
            for line in filehandle:
                # удалим заключительный символ перехода строки
                currentPlace = line[:-1]
                # добавим элемент в конец списка
                input_bias_weight.append(float(currentPlace))
        return input_bias_weight
    # Загрузка весов смещения скрытого слоя
    elif puty == 'leyar_bias_weight':
    # определим пустой список
        leyar_bias_weight = []
        leyar_bias_weight.clear()
        # откроем файл и считаем его содержимое в список
        with open('leyar_bias_weight.txt', 'r') as filehandle:  
            for line in filehandle:
                # удалим заключительный символ перехода строки
                currentPlace = line[:-1]
                # добавим элемент в конец списка
                leyar_bias_weight.append(float(currentPlace))
        return leyar_bias_weight
    # Загрузка весов выходного слоя
    elif puty == 'output_bias_weight':
    # определим пустой список
        output_bias_weight = []
        output_bias_weight.clear()
        # откроем файл и считаем его содержимое в список
        with open('output_bias_weight.txt', 'r') as filehandle:  
            for line in filehandle:
                # удалим заключительный символ перехода строки
                currentPlace = line[:-1]
                # добавим элемент в конец списка
                output_bias_weight.append(float(currentPlace))
        return output_bias_weight
    # Загрузка входных весов
    elif puty == 'input_weights':
    # определим пустой список
        input_weights = []
        input_weights.clear()
        # откроем файл и считаем его содержимое в список
        with open('input_weights.txt', 'r') as filehandle:  
            for line in filehandle:
                # удалим заключительный символ перехода строки
                currentPlace = line[:-1]
                # добавим элемент в конец списка
                input_weights.append(float(currentPlace))
        input_weights = np.reshape(input_weights, (100, 100))
        return input_weights
    # Загрузка весов скрытого слоя
    elif puty == 'leyar_weights':
    # определим пустой список
        leyar_weights = []
        leyar_weights.clear()
        # откроем файл и считаем его содержимое в список
        with open('leyar_weights.txt', 'r') as filehandle:  
            for line in filehandle:
                # удалим заключительный символ перехода строки
                currentPlace = line[:-1]
                # добавим элемент в конец списка
                leyar_weights.append(float(currentPlace))
        leyar_weights = np.reshape(leyar_weights, (100, 100))
        return leyar_weights

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    # input - изображение; l - размер матрицы весов для цикла 
    def feedforward(self, inputs, l):
        input_layer = []
        input_layer.clear()
        weights = []
        weights.clear()
        for i in range(l):
            input_layer.append(np.dot(self.weights[i], inputs))
        input_layer = np.array(input_layer)
        for i in range(l):
            weights.append(input_layer[i] + self.bias[i])
        weights = np.array(weights)
        return sigmoid(weights)
 
class OurNeuralNetwork:
    def __init__(self):
        #self.weights = weights
        #self.bias = bias
        self.h1 = Neuron(GetWeightFile('input_weights'), GetWeightFile('input_bias_weight'))
        self.h2 = Neuron(GetWeightFile('leyar_weights'), GetWeightFile('leyar_bias_weight'))
        self.o1 = Neuron(GetWeightFile('output_weights'), GetWeightFile('output_bias_weight'))
    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x, len(GetWeightFile('input_weights')))
        print(out_h1)
        print()
        out_h2 = self.h2.feedforward(out_h1, len(GetWeightFile('leyar_weights')))
        print(out_h2)
        print()
        out_o1 = self.o1.feedforward(out_h2, len(GetWeightFile('output_weights')))
        return out_o1


img = cv2.imread('2.png', 0)
h, w = img.shape
ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
cv2.imshow('qwe', img)
#print(img)

img1d = []
for i  in range(h):
    for j in range(w):
        if img[i][j] == 255: img1d.append(img[i][j] - 254)
        else: img1d.append(img[i][j])
#print(img1d)
# Передача веса и смещения в нейрон
#imput_layer = Neuron(GetWeightFile('input_weights'), GetWeightFile('bias_weights'))
#in_l = imput_layer.feedforward(img1d, len(GetWeightFile('input_weights')))
#print(in_l)
#print()

#slayer_l = Neuron(GetWeightFile('leyar_weights'), GetWeightFile('bias_weights'))
#slay_l = slayer_l.feedforward(img1d, len(GetWeightFile('leyar_weights')))
#print(slay_l)
#print()

#out_layer = Neuron(GetWeightFile('output_weights'), GetWeightFile('bias_weights'))
#print(out_layer.feedforward(slay_l, len(GetWeightFile('output_weights'))))



# Тренируем нашу нейронную сеть!
network = OurNeuralNetwork()
print(network.feedforward(img1d))



#SetWeightFile(CreatWeights(h*w)) #Генерация и запись в файл случайных значений для весов
#print(GetWeightFile('leyar_weights')) # Считывание с файла массива весов 
#//bias_weights //input_weights //output_weights 
#//input_bias_weight //leyar_bias_weight //output_bias_weight
cv2.waitKey(0)
cv2.destroyAllWindows()

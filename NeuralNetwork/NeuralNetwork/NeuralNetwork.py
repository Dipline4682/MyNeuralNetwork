#https://python-scripts.com/intro-to-neural-networks
#https://habr.com/ru/post/313216/
#https://habr.com/ru/post/312450/

import cv2
import random
import numpy as np

def CreatWeights(l):#h, w):
    mas = []
    for j in range(l):
        #mas.append(np.random.normal()) #Рандом Bias
        #mas.append(float("{0:.1f}".format(random.uniform(-1.0, 1.0))))#[random.uniform(0.0, 2.0) for j in range(w)])
        mas.append(float("{0:.1f}".format(random.uniform(-1.0, 1.0))))# for i in range(l)])#[random.uniform(0.0, 2.0) for j in range(w)])
    #print(len(mas))
    #mas = np.array(mas)
    return mas

def SetWeightFile(SaveWeight):
    with open('input_bias_weights.txt', 'w') as filehandle:  
        for listitem in SaveWeight:
            filehandle.write('%s\n' % listitem)
#-------------------------------------------------------------------------
#---------------------Загрузка весов--------------------------------------
def GetWeightFile (puty):
    #---------------------------------------------------------------------
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
    #---------------------------------------------------------------------
    # Загрузка весов выходного слоя
    elif puty == 'output_bias_weights':
    # определим пустой список
        output_bias_weight = []
        output_bias_weight.clear()
        # откроем файл и считаем его содержимое в список
        with open('output_bias_weights.txt', 'r') as filehandle:  
            for line in filehandle:
                # удалим заключительный символ перехода строки
                currentPlace = line[:-1]
                # добавим элемент в конец списка
                output_bias_weight.append(float(currentPlace))
        return output_bias_weight
    #---------------------------------------------------------------------
    # Загрузка входных весов смещения
    elif puty == 'input_bias_weights':
    # определим пустой список
        input_bias_weight = []
        input_bias_weight.clear()
        # откроем файл и считаем его содержимое в список
        with open('input_bias_weights.txt', 'r') as filehandle:  
            for line in filehandle:
                # удалим заключительный символ перехода строки
                currentPlace = line[:-1]
                # добавим элемент в конец списка
                input_bias_weight.append(float(currentPlace))
        return input_bias_weight
    #---------------------------------------------------------------------
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
    #---------------------------------------------------------------------
    # Загрузка весов смещения скрытого слоя
    elif puty == 'leyar_bias_weights':
    # определим пустой список
        leyar_bias_weight = []
        leyar_bias_weight.clear()
        # откроем файл и считаем его содержимое в список
        with open('leyar_bias_weights.txt', 'r') as filehandle:  
            for line in filehandle:
                # удалим заключительный символ перехода строки
                currentPlace = line[:-1]
                # добавим элемент в конец списка
                leyar_bias_weight.append(float(currentPlace))
        return leyar_bias_weight
    #---------------------------------------------------------------------
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

#код среднеквадратической ошибки (MSE)
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

#Вычесление производной функции активации
def Proizvodnaya(aktiv):
    x = []
    for i  in range(len(aktiv)):
        x.append((1 - aktiv[i]) * aktiv[i])
    return x
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    # input - изображение; l - размер матрицы весов для цикла 
    def feedforward(self, inputs, l):
        input_layer = []
        weights = []
        for i in range(l):
            weights.append(np.dot(self.weights[i], inputs))
        weights = np.array(weights)
        for i in range(l):
            input_layer.append(weights[i] + self.bias[i])
        input_layer = np.array(input_layer)
        return sigmoid(input_layer)
 
class OurNeuralNetwork:
    def __init__(self):
        #self.weights = weights
        #self.bias = bias
        self.h1 = Neuron(GetWeightFile('input_weights'), GetWeightFile('input_bias_weights'))
        self.h2 = Neuron(GetWeightFile('leyar_weights'), GetWeightFile('leyar_bias_weights'))
        self.o1 = Neuron(GetWeightFile('output_weights'), GetWeightFile('output_bias_weights'))
    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x, len(GetWeightFile('input_weights')))
        print('out_h1')
        print(out_h1)
        print()

        out_h2 = self.h2.feedforward(out_h1, len(GetWeightFile('leyar_weights')))
        print('out_h2')
        print(out_h2)
        print()

        out_o1 = self.o1.feedforward(out_h2, len(GetWeightFile('output_weights')))
        print('out_o1')

        return out_o1

    def train(self, data, ideal):
        """
        - data is a (n x 2) numpy array, n = # of samples in the dataset.
        - all_y_trues is a numpy array with n elements.
            Elements in all_y_trues correspond to those in data.
        """
        learn_rate = 0.1
        epochs = 1000 # количество циклов во всём наборе данных
 
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # --- Выполняем обратную связь (нам понадобятся эти значения в дальнейшем)
                h1 = self.h1.feedforward(x, len(GetWeightFile('input_weights')))
 
                h2 = self.h2.feedforward(out_h1, len(GetWeightFile('leyar_weights')))
 
                o1 = self.o1.feedforward(out_h2, len(GetWeightFile('output_weights')))
                y_pred = o1

                #Производная от функции активации
                P = Proizvodnaya(y_pred)
                #Вычесляем среднеквадратичню ошибку идеал минус получившийся ответ
                
                '''MSE = []
                #for i in range(len(pol)):
                    MSE.append(mse_loss(ideal[i], y_pred[i])/1)
                print('MSE = ', MSE)'''
                
                #Вычесляем дельта выход (разница между нужным и получившимся ответом 
                #умноженная на производную от функции активации)
                dO = []
                for i in range(len(MSE)):
                    dO.append(y_pred[i] * P[i])
                #print(dO)

                dH = []
                P = Proizvodnaya(y_pred)
 
                # --- Подсчет частных производных
                # --- Наименование: d_L_d_w1 представляет "частично L / частично w1"
                d_L_d_ypred = -2 * (y_true - y_pred)
 
                # Нейрон o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)
 
                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)
 
                # Нейрон h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)
 
                # Нейрон h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)
 
                # --- Обновляем вес и смещения
                # Нейрон h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1
 
                # Нейрон h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2
 
                # Нейрон o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3
 
            # --- Подсчитываем общую потерю в конце каждой фазы
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))



#Основное тело программы
img = cv2.imread('2.png', 0)
img = cv2.resize(img, (10, 10), interpolation = cv2.INTER_AREA)

h, w = img.shape
ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
cv2.imshow('qwe', img)
#print(img)

img1d = []
for i  in range(h):
    for j in range(w):
        if img[i][j] == 255: img1d.append(img[i][j]-254)
        else: img1d.append(img[i][j])

#Передаём изображение в нейронную сеть
network = OurNeuralNetwork()
#print(network.feedforward(img1d))
otv = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
otv = np.array(otv)
max = 0
count = 0

pol = []
pol = network.feedforward(img1d)
print(pol)
print()

#print(Proizvodnaya(pol))




for i in range(len(pol)):
    if pol[i] > max:
        max = pol[i]
        count = i


if count == 0: 
    print('otvet 0; Probability', pol[count])
    count = 0
elif count == 1: 
    print('otvet 1; Probability', pol[count])
    count = 0
elif count == 2: 
    print('otvet 2; Probability', pol[count])
    count = 0
elif count == 3: 
    print('otvet 3; Probability', pol[count])
    count = 0
elif count == 4: 
    print('otvet 4; Probability', pol[count])
    count = 0
elif count == 5: 
    print('otvet 5; Probability', pol[count])
    count = 0
elif count == 6: 
    print('otvet 6; Probability', pol[count])
    count = 0
elif count == 7: 
    print('otvet 7; Probability', pol[count])
    count = 0
elif count == 8: 
    print('otvet 8; Probability', pol[count])
    count = 0
elif count == 9: 
    print('otvet 9; Probability', pol[count])
    count = 0

#SetWeightFile(CreatWeights(h*w)) #Генерация и запись в файл случайных значений для весов
#print(GetWeightFile('leyar_weights')) # Считывание с файла массива весов 
#//bias_weights //input_weights //output_weights 
#//input_bias_weight //leyar_bias_weight //output_bias_weight
cv2.waitKey(0)
cv2.destroyAllWindows()

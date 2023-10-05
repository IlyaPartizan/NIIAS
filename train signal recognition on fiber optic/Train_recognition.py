import numpy as np
import json
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.utils import to_categorical
from keras.models import load_model
from random import shuffle
#подключение библиотек

def parsing(start_data):
    data_set = []
    for i in range(len(start_data)):
        file = open(start_data[i])
        info = file.read().split('\n')[:-1]
        info = list(map(lambda x: list(map(float, x.split(';'))), info))
        file.close()
        data_set.append(info)
    return(data_set)
#функция парсинга данных из файлов

def abs_nums(data):
    data_abs = []
    for i in range(len(data)):
        mas_no_abs = data[i]
        mas_with_abs = []
        for j in range(len(mas_no_abs)):
            nubers_with_abs = []
            numbers_no_abs = mas_no_abs[j]
            for n in range(len(numbers_no_abs)):
                nubers_with_abs.append(abs(numbers_no_abs[n]))
            mas_with_abs.append(nubers_with_abs)
        data_abs.append(mas_with_abs)  
    return data_abs
#функция взятия модуля

def imput_datas_for_neurons(data):
    imput_datas_neuron = []
    for i in range(len(data)):
        raw_data = data[i]
        imput_date_for_n = []
        for j in range(len(raw_data)):
            imput_numbers = []
            raw_numbers = raw_data[j]
            if len(raw_numbers) > 10:
                if len(raw_numbers) % 2 == 0:
                    count_delete = (len(raw_numbers) - 10) // 2
                    for n in range(count_delete, len(raw_numbers) - count_delete):
                        imput_numbers.append(raw_numbers[n])
                else: 
                    count_delete = (len(raw_numbers) - 10) // 2
                    for n in range(count_delete, len(raw_numbers) - count_delete + 1):
                        imput_numbers.append(raw_numbers[n])
            else:
                imput_numbers.extend(raw_numbers)
            imput_date_for_n.append(imput_numbers)
        imput_datas_neuron.append(imput_date_for_n)
    return imput_datas_neuron
#приседение данных к одному формату

def raw_datas_for_neuron(data):
    imput_for_neurons = []
    metrix = []
    for i in range(len(data)):
        mas_of_images = []
        raw_data_imput = data[i]
        for j in range(0, len(raw_data_imput) - 10, 10):
            image_of_ten_lines = []
            for n in range(j, j + 10):
                image_of_ten_lines.append(raw_data_imput[n])
            mas_of_images.append(image_of_ten_lines)
        mx = 0
        mas_of_summ_vsego = []
        metrix_one = []
        for r in range(len(mas_of_images)):
            sum = 0
            for a in range(len(mas_of_images[r])):
                for v in range(len(mas_of_images[r][a])):
                    sum = sum + mas_of_images[r][a][v]
            if mx < sum:
                mx = sum
            mas_of_summ_vsego.append(sum)
        for l in range(len(mas_of_summ_vsego)):
            mas_of_summ_vsego[l] = mas_of_summ_vsego[l]/mx*510
        for u in range(len(mas_of_summ_vsego)):
            if mas_of_summ_vsego[u] < 97:
                metrix_one.append(0)
            else:
                metrix_one.append(1)
        metrix.append(metrix_one)
        imput_for_neurons.append(mas_of_images)
    return([imput_for_neurons, metrix])
#подготовка тестовых и тренировочных данных

def decode_output_neuronet(mas):
    for i in range(len(mas)):
        print(mas[i])
    new_mas=[]
    for i in range(len(mas)):
        if mas[i][0] > mas[i][1]:
            new_mas.append(0)
        if mas[i][0] < mas[i][1]:
            new_mas.append(1)
    return new_mas
#функция декодирования выхода нейросети

def cut_binnary(data):
    data_cut = []
    start = 0
    for i in range(len(data)):
        if data[i] == 1:
            start = i
            break
    stop = 0
    for j in range(len(data) - 1, -1, -1):
        if data[j] == 1:
            stop = j
            break
    for index in range(start, stop + 1):
        data_cut.append(data[index])
    print(data_cut)
    return data_cut
#функция обрезающая значения бинарной последовательности с начала и с конца доходя до значений 1

def graphics(data):
    y1 = []
    y2 = []
    x1 = []
    x2 = []
    for i in range(len(data)):
        if data[i] == 1:
            x1.append(i)
            y1.append(0)
        else:
            x2.append(i)
            y2.append(0)
    plt.title("График выходов сверточной нейронной сети")
    plt.scatter(x1, y1, color="red", s=40, marker=".")
    plt.plot(x2, y2)
    plt.show()
#визуализация бинарной последовательности на графике

def startt(mas):
    s = 0
    e = len(mas)
    for i in range(len(mas)):
        if mas[i] >0.5:
            s = i

    for i in range(len(mas)):
        if mas[len(mas)-1-i] >0.5:
            e = len(mas)-1-i
    print(s)
    h = mas[e - 3:s +20]
    return h[:]
#функция алгоритма оптимизации выходов сверточной неронной сети

def urez(mas):
    new_mas = []
    if len(mas) % 2 == 0:
        for i in np.arange(0,len(mas), 2):
            if mas[i] == 0 and mas[i+1] == 0:
                new_mas.append(0)
                new_mas.append(0)
                new_mas.append(0)
                new_mas.append(0)
                new_mas.append(0)
            else:
                new_mas.append(1)
                new_mas.append(1)
                new_mas.append(1)
    elif len(mas) % 2 == 1:
        for i in np.arange(0, len(mas)-1, 2):
            if mas[i] == 0 and mas[i + 1] == 0:
                new_mas.append(0)
                new_mas.append(0)
                new_mas.append(0)
                new_mas.append(0)
                new_mas.append(0)
            else:
                new_mas.append(1)
                new_mas.append(1)
                new_mas.append(1)
    else:
        print("Ошибка")
    return new_mas[:]
#функция алгоритма оптимизации выходов сверточной неронной сети

def urez1(mas):
    new_mas = []
    if len(mas) % 2 == 0:
        for i in np.arange(0, len(mas)-2, 3):
            if mas[i] + mas[i + 1]+ mas[i + 2] < 0.5:
                new_mas.append(0)
                new_mas.append(0)
                new_mas.append(0)
            else:
                new_mas.append(1)
    elif len(mas) % 2 == 1:
        for i in np.arange(0, len(mas)-3, 3):
            if mas[i] + mas[i + 1]+ mas[i + 2] < 0.5:
                new_mas.append(0)
                new_mas.append(0)
                new_mas.append(0)
            else:
                new_mas.append(1)
    else:
        print("Ошибка")
    return new_mas[:]
#функция алгоритма оптимизации выходов сверточной неронной сети

def krasnoe(mas):
    new_mas = []
    if len(mas) % 2 == 0:
        for i in np.arange(0, len(mas)-1, 2):
            if mas[i] + mas[i + 1] > 0.5:
                new_mas.append(1)
                new_mas.append(1)
            else:
                new_mas.append(0)
                new_mas.append(0)
    elif len(mas) % 2 == 1:
        for i in np.arange(0, len(mas)-2, 2):
            if mas[i] + mas[i + 1] > 0.5:
                new_mas.append(1)
                new_mas.append(1)
            else:
                new_mas.append(0)
                new_mas.append(0)
    else:
        print("Ошибка")
    return new_mas[:]
#функция алгоритма оптимизации выходов сверточной неронной сети

def ris(mas_answers):
    y1 = []
    y2 = []
    x1 = []
    x2 = []
    re = 0
    for i in range(len(mas_answers)):
        if mas_answers[i] > 0.000005:
            x1.append(i)
            y1.append(0)
            re = re+1
        else:
            x2.append(i)
            y2.append(0)
    plt.scatter(x1, y1, color="red", s=40, marker=".")
    plt.plot(x2, y2)
    plt.title("График оптимизации выходов сверточной нейронной сети")
    plt.show()
#функция алгоритма оптимизации выходов сверточной неронной сети

def prividenie(mas):
    new_mas = [mas]
    i = 1
    while len(new_mas[i-1]) <len(mas)*2 :
        new_mas.append(urez(new_mas[i-1]))
        i = i + 1
    return new_mas[i-1]
#функция алгоритма оптимизации выходов сверточной неронной сети

def zikl(mas):
    new_mas = [mas]
    i = 1
    while len(new_mas[i-1]) > 2*len(mas)/3 :
        new_mas.append(urez1(new_mas[i-1]))
        i = i + 1
    return new_mas[i-1]
#функция алгоритма оптимизации выходов сверточной неронной сети

def zikl2(mas):
    new_mas = [mas]
    i = 1
    while len(new_mas[i-1]) > 3*len(mas)/4 :
        new_mas.append(urez1(new_mas[i-1]))
        i = i + 1
    return new_mas[i-1]
#функция алгоритма оптимизации выходов сверточной неронной сети

start_data = ['5598_13_10_27_1246-13_.csv','5598_13_10_27_1404-33_.csv','5598_14_04_32_1247-13_.csv','5598_14_04_32_1684-19_.csv']
#загрузка начальных данных из CSV

data_set_signal = raw_datas_for_neuron(imput_datas_for_neurons(abs_nums(parsing(start_data))))
#формирование общего датасета

x_test = np.array(data_set_signal[0][2])
y_test = np.array(data_set_signal[1][2])
#создание тестовой выборки

x_2 = np.array(data_set_signal[0][1])
y_2 = np.array(data_set_signal[1][1])
x_3 = np.array(data_set_signal[0][2])
y_3 = np.array(data_set_signal[1][2])
x_4 = np.array(data_set_signal[0][3])
y_4 = np.array(data_set_signal[1][3])
#разделение данных для дальнейшей проверки

x_2 = x_2.reshape(5870, 10, 10, 1)
x_3 = x_3.reshape(5656, 10, 10, 1)
y_2 = to_categorical(y_2)
y_3 = to_categorical(y_3)
x_4 = x_4.reshape(5893, 10, 10, 1)
y_4 = to_categorical(y_4)
#подготовка данных на вход нейросети для проверки

x_train = []
y_train = []
#массивы для обучающей выборки

x_train.extend(data_set_signal[0][0])
y_train.extend(data_set_signal[1][0])
x_train.extend(data_set_signal[0][1])
y_train.extend(data_set_signal[1][1])
x_train.extend(data_set_signal[0][3])
y_train.extend(data_set_signal[1][3])
#создание обучающей выборки

x_train = np.array(x_train)
y_train = np.array(y_train)
#перевод в массивы numpy

x_train = x_train.reshape(17678, 10, 10, 1)
x_test = x_test.reshape(5656, 10, 10, 1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
#подготовка данных к обучению

model = Sequential()
model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(10, 10, 1)))
model.add(Conv2D(16, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
hist = model.fit(x_train, y_train, validation_data= (x_test, y_test), epochs=15)
print(hist.history)
model.save('model_test.h5')
#создание и компиляция модели сверточной нейросети использующей сверточные слои с функцией активации relu и softmax на выходе

model = load_model('model_final.h5')
#загрузка оптимальной модели полученной экспериментально

y_1_pr = model.predict(x_test[:])
y_2_pr = model.predict(x_2[:])
y_3_pr = model.predict(x_3[:])
y_4_pr = model.predict(x_4[:])
#распознавание данных с помощью модели

first_data = decode_output_neuronet(y_1_pr)
second_data = decode_output_neuronet(y_2_pr)
third_data = decode_output_neuronet(y_3_pr)
fourth_data = decode_output_neuronet(y_4_pr)
#декодирование выхода нейронной сети в бинарную последовательность

first_data_cut = cut_binnary(first_data)
second_data_cut = cut_binnary(second_data)
third_data_cut = cut_binnary(third_data)
fourth_data_cut = cut_binnary(fourth_data)
#образка данных

graphics(first_data_cut)
graphics(third_data_cut)
#визуализация

datas = [first_data, third_data]
#данные для нейронной сети для оптимизации

for i in range(len(datas)):
    kik = startt(datas[i])
    a1 = krasnoe(krasnoe(krasnoe(kik)))
    a1_1 = zikl(a1)
    kik3_1 = krasnoe(krasnoe(a1_1))
    kik2 = prividenie(kik3_1)
    a12 = krasnoe(krasnoe(a1_1))
    a1_12 = prividenie(a12)
    ris(a1_12)
    kik3_12 = zikl(a1_12)
    ris(zikl(kik3_12))
#реализация для алгоритма оптимизации выходов сверточной нейронной сети
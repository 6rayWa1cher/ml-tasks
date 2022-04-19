import csv

import matplotlib.pyplot as plt
import numpy as np
import json


def add_intercept_fn(x):
    """Добавить столбец с коэффициентами для свободных членов.

    Аргументы:
        x: 2D NumPy array.

    Возвращаемое значение:
        Новая матрица, являющаяся конкатенацией столбца из единиц и матрицы x.
    """
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x

    return new_x

def load_csv(csv_path, label_col='y', add_intercept=False):
    """Загрузить датасет из CSV файла

    Аргументы:
         csv_path: Пусть к CSV файлу с датасетом.
         label_col: Имя столбца, содержащего классы (должно быть 'y' или 'l').
         add_intercept: Добавить столбец из 1 к матрице x.

    Возвращаемое значение:
        xs: Numpy array со входными значениями x.
        ys: Numpy array с выходными значениями y.
    """

    # Load headers
    with open(csv_path, 'r', newline='') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    l_cols = [i for i in range(len(headers)) if headers[i] == label_col]
    inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols)
    labels = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=l_cols)

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)

    if add_intercept:
        inputs = add_intercept_fn(inputs)

    return inputs, labels

def load_spam_dataset(tsv_path):
    """Загрузить датасет для обучения спам-фильтра из TSV файла

    Аргументы:
         tsv_path: Пусть к TSV файлу с датасетом.

    Возвращаемое значение:
        messages: Список строк, содержащих текст каждого сообщения.
        labels: Бинаные метки (0 или 1) для каждого сообщения. Метка 1 означает спам.
    """

    messages = []
    labels = []

    with open(tsv_path, 'r', newline='', encoding='utf8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')

        for label, message in reader:
            messages.append(message)
            labels.append(1 if label == 'spam' else 0)

    return messages, np.array(labels)

def plot(x, y, theta, save_path, correction=1.0):
    """Построить график датасета и решающей границы регрессии Пуассона.

    Аргументы:
        x: Матрица обучающих примеров, по одному на строке.
        y: Вектор классов из {0, 1}.
        theta: Вектор параметров регрессионной модели.
        save_path: Имя файла для сохранения графика.
        correction: Коррекционный фактор.
    """
    # Plot dataset
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)

    # Plot decision boundary (found by solving for theta^T x = 0)
    x1 = np.arange(min(x[:, -2]), max(x[:, -2]), 0.01)
    x2 = -(theta[0] / theta[2] * correction + theta[1] / theta[2] * x1)
    plt.plot(x1, x2, c='red', linewidth=2)

    # Add labels and save to disk
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(save_path)


def plot_contour(predict_fn):
    """Построить контур для заданной функции гипотезы"""
    x, y = np.meshgrid(np.linspace(-10, 10, num=20), np.linspace(-10, 10, num=20))
    z = np.zeros(x.shape)

    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            z[i, j] = predict_fn([x[i, j], y[i, j]])

    plt.contourf(x, y, z, levels=[-float('inf'), 0, float('inf')], colors=['orange', 'cyan'])

def plot_points(x, y):
    """Построить точечный график, где x - координаты, а y - метки"""
    x_one = x[y == 0, :]
    x_two = x[y == 1, :]

    plt.scatter(x_one[:,0], x_one[:,1], marker='x', color='red')
    plt.scatter(x_two[:,0], x_two[:,1], marker='o', color='blue')

def write_json(filename, value):
    """Записать переданное значение в JSON формате в файл с заданным именем"""
    with open(filename, 'w') as f:
        json.dump(value, f)

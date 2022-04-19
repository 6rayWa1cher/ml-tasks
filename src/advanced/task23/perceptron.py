import math

import matplotlib.pyplot as plt
import numpy as np

import util


class PerceptronState:
    def __init__(self, dataset, alpha, dict_index):
        self.dataset = dataset
        self.alpha = alpha
        self.dict_index = dict_index


def initial_state(train_x):
    """Возвращает начальное состояние персептрона.

    Эта функция вычисляет и возвращает начальное состояние персептрона.
	Вы можете использовать любые типа данных (словари, списки, кортежи или свои собственные классы),
	чтобы представлять состояне персептрона.
    """

    # *** НАЧАЛО ВАШЕГО КОДА ***
    index = -1

    def get_next_index():
        nonlocal index
        index += 1
        return index

    return PerceptronState(
        train_x,
        np.zeros(len(train_x)),
        {tuple(train_x[i]): get_next_index() for i in range(len(train_x))}
    )
    # *** КОНЕЦ ВАШЕГО КОДА ***


def predict(state, kernel, x_i):
    """Возвращает прогноз на значении x_i на основе текущего состояния и ядра.

    Аргументы:
        state: Состояние в том же самом формате, который использует функция initial_state()
        kernel: Бинарная функция ядра, возвращающая его значение от этих аргументов
        x_i: Входной пример, для которого нужно вернуть прогноз

    Возвращаемое значение:
        Возвращает прогноз (т.е. 0 или 1)
    """
    # *** НАЧАЛО ВАШЕГО КОДА ***
    alpha = state.alpha
    dataset = state.dataset
    sum_linear = np.sum(alpha * kernel(dataset, x_i))
    return sign(sum_linear)
    # *** КОНЕЦ ВАШЕГО КОДА ***


def update_state(state, kernel, learning_rate, x_i, y_i) -> None:
    """Обновляет состояние персептрона.

    Аргументы:
        state: Состояние в том же самом формате, который использует функция initial_state()
        kernel: Бинарная функция ядра, возвращающая его значение от этих аргументов
        learning_rate: Скорость обучения (коэфициент alpha)
        x_i: Вектор признаков образца
        y_i: Метка 0 или 1, указывающая класс образца
    """
    # *** НАЧАЛО ВАШЕГО КОДА ***
    alpha = state.alpha
    dict_index = state.dict_index
    a = predict(state, kernel, x_i)
    if a != y_i:
        alpha[dict_index[tuple(x_i)]] += learning_rate * (y_i - a)
    # *** КОНЕЦ ВАШЕГО КОДА ***


def sign(a):
    """Возвращает знак скалярного аргумента."""
    if a >= 0:
        return 1
    else:
        return 0


def dot_kernel(a, b):
    """Реализация линейного ядра.

    Аргументы:
        a: вектор
        b: вектор
    """
    return np.dot(a, b)


def rbf_kernel(a, b, sigma=1):
    """Реализация гауссова ядра.

    Аргументы:
        a: вектор
        b: вектор
        sigma: радиус ядра
    """
    distance = np.sum((a - b) * (a - b), axis=1)
    scaled_distance = -distance / (2 * (sigma) ** 2)
    return np.exp(scaled_distance)


def train_perceptron(kernel_name, kernel, learning_rate):
    """Обучает персептрон, используя переданное ядро.

	This function trains a perceptron with a given kernel and then
    uses that perceptron to make predictions.
    The output predictions are saved to src/perceptron/perceptron_{kernel_name}_predictions.txt.
    The output plots are saved to src/perceptron/perceptron_{kernel_name}_output.pdf.

    Аргументы:
        kernel_name: Имя ядра
        kernel: Функция ядра
        learning_rate: Скорость обучения (коэфициент alpha)
    """
    prefix_path = r"models/task23/"
    train_x, train_y = util.load_csv(prefix_path + 'train.csv')

    state = initial_state(train_x)

    for x_i, y_i in zip(train_x, train_y):
        update_state(state, kernel, learning_rate, x_i, y_i)

    test_x, test_y = util.load_csv(prefix_path + 'test.csv')

    plt.figure(figsize=(12, 8))
    util.plot_contour(lambda a: predict(state, kernel, a))
    util.plot_points(test_x, test_y)
    plt.savefig('stats/task23/perceptron_{}_output.png'.format(kernel_name))

    predict_y = [predict(state, kernel, test_x[i, :]) for i in range(test_y.shape[0])]

    print("Точность = " + str(np.mean(np.array(predict_y) == test_y)))

    np.savetxt('stats/task23/perceptron_{}_predictions'.format(kernel_name), predict_y)


def main():
    train_perceptron('dot', dot_kernel, 0.5)
    train_perceptron('rbf', rbf_kernel, 0.5)


if __name__ == "__main__":
    main()

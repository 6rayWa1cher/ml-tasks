import numpy as np
import matplotlib.pyplot as plt
import argparse


def softmax(x):
    """
    Вычисляет значения функции софтмакса для пакета примеров.
	Количество строк матрицы x равно размеру пакета, столбцов -
	количеству классов на выходе модели.

    Важное замечание: имейте в виду, что данная функция подвержена проблеме переполнения. Это происходит
    из-за вычисления очень больших чисел вроде e^10000.
    Вы можете считать, что ваша реализация устойчива к вышеуказанной проблеме, если она сможет
	обработать вход np.array([[10000, 10010, 10]]) без ошибок.

    Аргументы:
        x: Матрица numpy вещественных чисел размерности batch_size x number_of_classes

    Возвращаемое значение:
        Матрица numpy вещественных чисел, содержащая результаты вычисления софтмакса размерности batch_size x number_of_classes
    """
    # *** НАЧАЛО ВАШЕГО КОДА ***
    x_max = np.atleast_2d(np.max(x, axis=1)).T
    x_sub = x - x_max
    x_log = np.atleast_2d(np.log(np.sum(np.exp(x_sub), axis=1))).T
    return np.exp(x_sub - x_log)
    # *** КОНЕЦ ВАШЕГО КОДА ***


def sigmoid(x):
    """
    Вычисляет значение логистической функции.

    Аргументы:
        x: Массив numpy вещественных чисел

    Возвращаемое значение:
        Массив numpy вещественных чисел, содержащих значения функции
    """
    # *** НАЧАЛО ВАШЕГО КОДА ***
    return 1 / (1 + np.exp(-x))
    # *** КОНЕЦ ВАШЕГО КОДА ***


def sigmoid_diff(x):
    return sigmoid(x) * (1 - sigmoid(x))


def ce(y, y_cap):
    return -np.sum(y * np.log(y_cap))


def get_initial_params(input_size, num_hidden, num_output) -> dict:
    """
    Производит инициализацию параметров нейронной сети.

    Эта функция должна возвращать словарь, в котором ключом являются имена параметров,
	а значениями сами numpy массивы.

    Для нашей модели должно быть четыре массива параметров:
    W1 - матрица весов для скрытого слоя
    b1 - вектор смещений для скрытого слоя
    W2 - матрица весов для выходного слоя
    b2 - вектор весов для выходного слоя

    В соответствии с заданием параметры сети должны быть инициализированы случайными значениями,
	распределенными по стандартному нормальному закону. Смещения должны быть инициализированы нулями.

    Аргументы:
        input_size: Размер входа нейронной сети
        num_hidden: Количество нейронов скрытого слоя
        num_output: Размер выхода нейронной сети

    Возвращаемое значение:
        Словарь, содержащий отображения имен параметров в numpy матрицы
    """

    # *** НАЧАЛО ВАШЕГО КОДА ***
    w1 = np.random.normal(size=(input_size, num_hidden))
    b1 = np.zeros(num_hidden)
    w2 = np.random.normal(size=(num_hidden, num_output))
    b2 = np.zeros(num_output)
    return {
        'W1': w1,
        'b1': b1,
        'W2': w2,
        'b2': b2
    }
    # *** КОНЕЦ ВАШЕГО КОДА ***


def forward_prop(data, labels, params):
    """
    Осуществляет прямое распространение на основе входных данных, меток и весов сети.

    Аргументы:
        data: Массив numpy, содержащий входные данные
        labels: Двухмерный массив numpy, содержащий метки в one-hot формате
        params: Словарь, содержащий отображения имен параметров в numpy матрицы
				(см. описание функции get_initial_params).

    Возвращаемое значение:
        Кортеж из трех значений:
            1. Массив numpy значений функции активации (после вычисления сигмоиды) скрытого слоя сети
            2. Массив numpy значений выходного слоя (после софтмакса)
            3. Усредненное значение функции потерь для данного входа
    """
    # *** НАЧАЛО ВАШЕГО КОДА ***
    w1, b1, w2, b2 = params['W1'], params['b1'], params['W2'], params['b2']
    m, _ = data.shape
    hidden = sigmoid(data.dot(w1) + b1)
    output = softmax(hidden.dot(w2) + b2)
    ce_sum = np.sum(ce(labels, output))
    cost = ce_sum / m
    return hidden, output, cost
    # *** КОНЕЦ ВАШЕГО КОДА ***


def backward_prop(data, labels, params, forward_prop_func):
    """
    Осуществляет обратное распространение ошибки в сети.

    Аргументы:
        data: Массив numpy, содержащий входные данные
        labels: Двухмерный массив numpy, содержащий метки в one-hot формате
        params: Словарь, содержащий отображения имен параметров в numpy матрицы
				(см. описание функции get_initial_params)
        forward_prop_func: Функция прямого распространения, описанная выше

    Возвращаемое значение:
	   Словарь с результатами обратного распространения, в котором ключом является имя параметра,
	   а значением - градиент функции потерь по данному параметру

        В частности, в нем должно быть 4 элемента:
            W1, W2, b1 и b2
    """

    # *** НАЧАЛО ВАШЕГО КОДА ***

    w1, b1, w2, b2 = params['W1'], params['b1'], params['W2'], params['b2']
    hidden, output, _ = forward_prop_func(data, labels, params)
    sigma_k = output - labels
    dw2 = hidden.T.dot(sigma_k)
    db2 = np.sum(sigma_k, axis=0)
    sigma_j = sigma_k.dot(w2.T) * sigmoid_diff(hidden)
    dw1 = data.T.dot(sigma_j)
    db1 = np.sum(sigma_j, axis=0)
    return {
        'W1': dw1,
        'b1': db1,
        'W2': dw2,
        'b2': db2
    }
    # *** КОНЕЦ ВАШЕГО КОДА ***


def backward_prop_regularized(data, labels, params, forward_prop_func, reg):
    """
    Осуществляет регуляризованное обратное распространение ошибки в сети

    Аргументы:
        data: Массив numpy, содержащий входные данные
        labels: Двухмерный массив numpy, содержащий метки в one-hot формате
        params: Словарь, содержащий отображения имен параметров в numpy матрицы
				(см. описание функции get_initial_params)
        forward_prop_func: Функция прямого распространения, описанная выше
        reg: коэффициент регуляризации (lambda)

    Возвращаемое значение:
        Словарь с результатами обратного распространения, в котором ключом является имя параметра,
	    а значением - градиент функции потерь по данному параметру

        В частности, в нем должно быть 4 элемента:
            W1, W2, b1 и b2
    """
    # *** НАЧАЛО ВАШЕГО КОДА ***
    w1, b1, w2, b2 = params['W1'], params['b1'], params['W2'], params['b2']
    m, _ = data.shape
    hidden, output, _ = forward_prop_func(data, labels, params)
    sigma_k = output - labels
    dw2 = hidden.T.dot(sigma_k) + (reg / m) * w2
    db2 = np.sum(sigma_k, axis=0)
    sigma_j = sigma_k.dot(w2.T) * sigmoid_diff(hidden)
    dw1 = data.T.dot(sigma_j) + (reg / m) * w1
    db1 = np.sum(sigma_j, axis=0)
    return {
        'W1': dw1,
        'b1': db1,
        'W2': dw2,
        'b2': db2
    }
    # *** КОНЕЦ ВАШЕГО КОДА ***


def gradient_descent_epoch(train_data, train_labels, learning_rate, batch_size, params, forward_prop_func,
                           backward_prop_func):
    """
    Реализует одну эпоху обучения нейронной сети на предоставленных данных.

    Этот код должен обновлять значения параметров, сохраненных в params.
    Функция ничего не возвращает.

    Аргументы:
        train_data: Массив numpy, содержащий входные данные
        train_labels: Массив numpy, содержащий метки для входных данных
        learning_rate: Скорость обучения
        batch_size: Размер мини-пакета
		forward_prop_func: Функция прямого распространения
        backward_prop_func: Функция обратного распространения ошибки

    Возвращаемое значение: функция ничего не возвращает.
    """

    # *** НАЧАЛО ВАШЕГО КОДА ***
    w1, b1, w2, b2 = params['W1'], params['b1'], params['W2'], params['b2']
    m, _ = train_data.shape
    for batch_index in range(m // batch_size):
        batch_start = batch_index * batch_size
        batch_end = (batch_index + 1) * batch_size
        data = train_data[batch_start:batch_end]
        labels = train_labels[batch_start:batch_end]
        error_corrections = backward_prop_func(data, labels, params, forward_prop_func)
        dw1, db1, dw2, db2 = error_corrections['W1'], error_corrections['b1'], \
                             error_corrections['W2'], error_corrections['b2']
        w1 -= learning_rate / batch_size * dw1
        b1 -= learning_rate / batch_size * db1
        w2 -= learning_rate / batch_size * dw2
        b2 -= learning_rate / batch_size * db2
    # *** КОНЕЦ ВАШЕГО КОДА ***

    # Функция ничего не возвращает
    return


def nn_train(
    train_data, train_labels, dev_data, dev_labels,
    get_initial_params_func, forward_prop_func, backward_prop_func,
    num_hidden=300, learning_rate=5, num_epochs=30, batch_size=1000):
    (nexp, dim) = train_data.shape

    params = get_initial_params_func(dim, num_hidden, 10)

    cost_train = []
    cost_dev = []
    accuracy_train = []
    accuracy_dev = []
    for epoch in range(num_epochs):
        gradient_descent_epoch(train_data, train_labels, learning_rate, batch_size, params, forward_prop_func,
                               backward_prop_func)

        h, output, cost = forward_prop_func(train_data, train_labels, params)
        cost_train.append(cost)
        accuracy_train.append(compute_accuracy(output, train_labels))
        h, output, cost = forward_prop_func(dev_data, dev_labels, params)
        cost_dev.append(cost)
        accuracy_dev.append(compute_accuracy(output, dev_labels))

    return params, cost_train, cost_dev, accuracy_train, accuracy_dev


def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy


def compute_accuracy(output, labels):
    accuracy = (np.argmax(output, axis=1) ==
                np.argmax(labels, axis=1)).sum() * 1. / labels.shape[0]
    return accuracy


def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size), labels.astype(int)] = 1
    return one_hot_labels


def read_data(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y


def run_train_test(name, all_data, all_labels, backward_prop_func, num_epochs, plot=True):
    params, cost_train, cost_dev, accuracy_train, accuracy_dev = nn_train(
        all_data['train'], all_labels['train'],
        all_data['dev'], all_labels['dev'],
        get_initial_params, forward_prop, backward_prop_func,
        num_hidden=300, learning_rate=5, num_epochs=num_epochs, batch_size=1000
    )

    t = np.arange(num_epochs)

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1)

        ax1.plot(t, cost_train, 'r', label='train')
        ax1.plot(t, cost_dev, 'b', label='dev')
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss')
        if name == 'baseline':
            ax1.set_title('Без регуляризации')
        else:
            ax1.set_title('С регуляризацией')
        ax1.legend()

        ax2.plot(t, accuracy_train, 'r', label='train')
        ax2.plot(t, accuracy_dev, 'b', label='dev')
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('accuracy')
        ax2.legend()

        fig.savefig('./stats/task24/' + name + '.pdf')

        for param_name in 'W1', 'b1', 'W2', 'b2':
            np.savetxt('stats/task24/{}_{}'.format(name, param_name), params[param_name])

    accuracy = nn_test(all_data['test'], all_labels['test'], params)
    print('Для модели %s получена точность: %f' % (name, accuracy))

    return accuracy


def main(plot=True):
    parser = argparse.ArgumentParser(description='Train a nn model.')
    parser.add_argument('--num_epochs', type=int, default=30)

    args = parser.parse_args()

    np.random.seed(100)
    train_data, train_labels = read_data('models/task24/images_train.csv', 'models/task24/labels_train.csv')
    train_labels = one_hot_labels(train_labels)
    p = np.random.permutation(60000)
    train_data = train_data[p, :]
    train_labels = train_labels[p, :]

    dev_data = train_data[0:10000, :]
    dev_labels = train_labels[0:10000, :]
    train_data = train_data[10000:, :]
    train_labels = train_labels[10000:, :]

    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = (train_data - mean) / std
    dev_data = (dev_data - mean) / std

    test_data, test_labels = read_data('models/task24/images_test.csv', 'models/task24/labels_test.csv')
    test_labels = one_hot_labels(test_labels)
    test_data = (test_data - mean) / std

    all_data = {
        'train': train_data,
        'dev': dev_data,
        'test': test_data
    }

    all_labels = {
        'train': train_labels,
        'dev': dev_labels,
        'test': test_labels,
    }

    baseline_acc = run_train_test('baseline', all_data, all_labels, backward_prop, args.num_epochs, plot)
    reg_acc = run_train_test('regularized', all_data, all_labels,
                             lambda a, b, c, d: backward_prop_regularized(a, b, c, d, reg=0.0001),
                             args.num_epochs, plot)

    return baseline_acc, reg_acc


if __name__ == '__main__':
    main()

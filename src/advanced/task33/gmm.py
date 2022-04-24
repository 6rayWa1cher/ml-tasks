import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Цвета для ваших графиков
K = 4  # Количество Гауссиан в модели смеси
NUM_TRIALS = 3  # Количество запусков алгоритма
UNLABELED = -1  # Метка кластера для неразмеченных данных (не меняйте)


def main(is_semi_supervised, trial_num):
    """Гибридная версия EM-алгоритма"""
    print('Выполняем {} версию EM алгоритма...'
          .format('гибридную' if is_semi_supervised else 'классическую'))

    # Загружаем датасет
    train_path = os.path.join('./models/task33/', 'train.csv')
    x_all, z_all = load_gmm_dataset(train_path)

    # Разделяем на размеченные и неразмеченные данные
    labeled_idxs = (z_all != UNLABELED).squeeze()
    x_tilde = x_all[labeled_idxs, :]  # Размеченные примеры
    z_tilde = z_all[labeled_idxs, :]  # Соответствующие метки
    x = x_all[~labeled_idxs, :]  # Неразмеченные примеры

    # *** НАЧАЛО ВАШЕГО КОДА ***
    # (1) Инициализируйте mu и sigma путем разбиения n_examples точек данных равномерно
    # K групп, затем посчитайте выборочно среднее и ковариацию для каждой группы
    # (2) Инициализируйте phi, чтобы оно давало равную вероятность каждой Гауссиане
    # phi должен быть numpy массивом размерности (K,)
    # (3) Инициализируйте значения w, чтобы они давали равную вероятность каждой Гауссиане
    # w должен быть numpy массивом размерности (m, K)
    m, n = x.shape
    mu = np.zeros((K, n))
    random_indices = np.random.permutation(m)
    for i in range(K):
        x_local = x[(random_indices % K == i).flatten()]
        mu[i] = np.mean(x_local, axis=0)
    sigma = np.zeros((K, n, n))
    for i in range(K):
        sigma[i] = np.eye(n)
    phi = np.atleast_2d(np.array([1 / K for _ in range(K)]))
    w = phi.repeat(m, axis=1).reshape((m, K))
    # *** КОНЕЦ ВАШЕГО КОДА ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Нарисуем предсказания алгоритма
    z_pred = np.zeros(m)
    if w is not None:
        for i in range(m):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def get_ro(x, m, mu, sigma):
    inv_sigma = np.linalg.inv(sigma)
    ro_mult = 1 / ((2 * np.pi) ** (K / 2) * np.sqrt(np.linalg.det(sigma)))
    ro_mat = np.zeros((m, K))
    for j in range(K):
        x_sub = x - mu[j]
        for i, x_sub_i in enumerate(x_sub):
            x_sub_i_2d = np.atleast_2d(x_sub_i)
            ro_mat[i, j] = (ro_mult[j] * np.exp(-1 / 2 * (x_sub_i @ inv_sigma[j]).dot(x_sub_i_2d.T))).item()
    return ro_mat


def get_ro_tilde(x, z, m, mu, sigma):
    inv_sigma = np.linalg.inv(sigma)
    ro_mult = 1 / ((2 * np.pi) ** (K / 2) * np.sqrt(np.linalg.det(sigma)))
    ro_mat = np.zeros((m, 1))
    for i, x_i in enumerate(x):
        z_i = int(z[i].item())
        x_sub_i = x_i - mu[z_i]
        x_sub_i_2d = np.atleast_2d(x_sub_i)
        ro_mat[i] = (ro_mult[z_i] * np.exp(-1 / 2 * (x_sub_i @ inv_sigma[z_i]).dot(x_sub_i_2d.T))).item()
    return ro_mat


def run_em(x, w, phi, mu, sigma):
    """Подзадача 4: классический EM-алгоритм.

    Инструкции указаны в комментариях.

    Аргументы:
        x: Матрица данных размерности (n_examples, dim).
        w: Первоначальная матрица весов размерности (n_examples, k).
        phi: Первоначальное априорное распределение размерности (k,).
        mu: Первоначальные средние значения кластеров, список из k массивов размерности (dim,).
        sigma: Первоначальные ковариации кластеров, список из k массивов размерности (dim, dim).

    Возвращаемое значение:
        Обновленная матрица весов размерности (n_examples, k), полученная в результате применения EM-алгоритма.
        Более конкретно, w[i, j] должен содержать вероятность того,
        что пример x^(i) принадлежит j-й Гауссиане в смеси.
    """
    # Менять эти параметры не требуется
    eps = 1e-3  # Порог сходимости
    max_iter = 1000

    # Останавливаемся, когда абсолютное значение изменения log-правдоподобия меньше eps
    # Смотрите ниже объяснение схождения
    it = 0
    ll = prev_ll = None
    m, n = x.shape
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # *** НАЧАЛО ВАШЕГО КОДА
        # (1) E-шаг: Обновите ваши оценки в w
        # x_sub_3d = np.atleast_3d(x).reshape((1, m, n)) - mu.reshape((K, 1, n))
        # semi_probs = np.exp(-1/2 * x_sub_3d.transpose((0, 2, 1)) @ np.linalg.inv(sigma) @ x_sub_3d)
        # inv_sigma = np.linalg.inv(sigma)
        # ro_mult = 1 / ((2 * np.pi) ** (K / 2) * np.sqrt(np.linalg.det(sigma)))
        # ro_mat = np.zeros((m, K))
        # for j in range(K):
        #     x_sub = x - mu[j]
        #     for i, x_sub_i in enumerate(x_sub):
        #         x_sub_i_2d = np.atleast_2d(x_sub_i)
        #         ro_mat[i, j] = (ro_mult[j] * np.exp(-1/2 * (x_sub_i @ inv_sigma[j]).dot(x_sub_i_2d.T))).item()
        # prob_ro = phi * ro_mat
        prob_ro = phi * get_ro(x, m, mu, sigma)
        w = prob_ro / np.sum(prob_ro, axis=1).reshape((m, 1))
        # (2) M-step: Обновите параметры модели phi, mu и sigma
        phi = 1 / m * np.sum(w, axis=0)
        mu = np.sum(w.reshape((m, K, 1)) * x.reshape((m, 1, n)), axis=0) / np.sum(w, axis=0).reshape((K, 1))
        sigma = np.zeros((K, n, n))
        for j in range(K):
            for i, x_i in enumerate(x):
                x_sub = np.atleast_2d(x_i - mu[j])
                product = x_sub.T @ x_sub
                sigma[j] += w[i, j] * product
        sigma /= np.sum(w, axis=0).reshape((K, 1, 1))
        # (3) Вычислите log-правдоподобие данных для определения схождения
        # Под log-правдоподобием мы понимаем `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # Под схождением мы понимаем первую итерацию, когда abs(ll - prev_ll) < eps.
        # Подсказка: Для отладки вспомните подзадание №1. В нем мы показали, что ll должна монотонно возрастать.
        new_prob_ro = phi * get_ro(x, m, mu, sigma)
        prev_ll = ll
        ll = np.sum(np.log(np.sum(new_prob_ro, axis=1)))
        it += 1
        # *** КОНЕЦ ВАШЕГО КОДА ***

    return w


def eval_gaussian(x, mu, sigma):
    norm_factor = 1.0 / (np.power(2 * np.pi, sigma.shape[0] / 2) * np.sqrt(np.linalg.det(sigma)))
    diff = np.expand_dims(x - mu, 1)
    pdf = np.exp(-.5 * np.dot(np.dot(diff.T, np.linalg.inv(sigma)), diff))
    return norm_factor * pdf


def eval_gaussian_log(x, mu, sigma):
    norm_factor = 1.0 / (np.power(2 * np.pi, sigma.shape[0] / 2) * np.sqrt(np.linalg.det(sigma)))
    diff = x - mu
    pdf = -.5 * np.dot(np.dot(diff.T, np.linalg.inv(sigma)), diff)
    return np.log(norm_factor) + pdf


def m_step_hybrid(x, x_tilde, z_tilde, alpha, w):
    m, n = x.shape
    m_tilde, _ = x_tilde.shape
    mask = np.arange(0, K).reshape((1, K)).repeat(m_tilde, axis=0)
    elements = z_tilde.repeat(K, axis=1)
    z_tilde_count = np.sum((mask == elements).astype(int), axis=0)
    phi = (np.sum(w, axis=0) + alpha * z_tilde_count) / (m + alpha * m_tilde)

    w_x_sum = np.sum(w.reshape((m, K, 1)) * x.reshape((m, 1, n)), axis=0)
    w_sum = np.sum(w, axis=0).reshape((K, 1))
    x_tilde_sum = np.sum(x_tilde.reshape((m_tilde, 1, n)) * (mask == elements).reshape((m_tilde, K, 1)), axis=0)
    mu = (w_x_sum + alpha * x_tilde_sum) / (w_sum + alpha * z_tilde_count.reshape((K, 1)))

    sigma = np.zeros((K, n, n))
    for j in range(K):
        for i, x_i in enumerate(x):
            x_sub = np.atleast_2d(x_i - mu[j])
            product = x_sub.T @ x_sub
            sigma[j] += w[i, j] * product
        for i, x_tilde_i in enumerate(x_tilde):
            if z_tilde[i] == j:
                x_tilde_sub = np.atleast_2d(x_tilde_i - mu[j])
                product = x_tilde_sub.T @ x_tilde_sub
                sigma[j] += alpha * product
        # sigma[j] /= w_sum[j] + alpha * z_tilde_count[j]
    sigma /= (w_sum + alpha * z_tilde_count.reshape((K, 1))).reshape((K, 1, 1))

    return mu, sigma, phi


def run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma):
    """Подзадача 5: Гибридный EM-алгоритм.

    Инструкции указаны в комментариях.

    Аргументы:
        x: Матрица неразмеченных данных размерности (n_examples_unobs, dim).
        x_tilde: Матрица размеченных данных размерности (n_examples_obs, dim).
        z_tilde: Массив меток размерности (n_examples_obs, 1).
		w: Первоначальная матрица весов размерности (n_examples, k).
        phi: Первоначальное априорное распределение размерности (k,).
        mu: Первоначальные средние значения кластеров, список из k массивов размерности (dim,).
        sigma: Первоначальные ковариации кластеров, список из k массивов размерности (dim, dim).

    Возвращаемое значение:
        Обновленная матрица весов размерности (n_examples, k), полученная в результате применения
		гибридного EM-алгоритма.
        Более конкретно, w[i, j] должен содержать вероятность того,
        что пример x^(i) принадлежит j-й Гауссиане в смеси.
    """
    # Менять эти параметры не требуется
    alpha = 20.  # Вес для размеченных примеров
    eps = 1e-3  # Порог сходимости
    max_iter = 1000

    # Останавливаемся, когда абсолютное значение изменения log-правдоподобия меньше eps
    # Смотрите ниже объяснение схождения
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # *** НАЧАЛО ВАШЕГО КОДА
        # (1) E-шаг: Обновите ваши оценки в w
        m, n = x.shape
        m_tilde, _ = x_tilde.shape
        prob_ro = phi * get_ro(x, m, mu, sigma)
        w = prob_ro / np.sum(prob_ro, axis=1).reshape((m, 1))
        # (2) M-step: Обновите параметры модели phi, mu и sigma
        mu, sigma, phi = m_step_hybrid(x, x_tilde, z_tilde, alpha, w)
        # (3) Вычислите log-правдоподобие данных для определения схождения.
        # Подсказка: Не забудьте про alpha при вычислении ll.
        # Подсказка: Для отладки вспомните подзадание №1. В нем мы показали, что ll должна монотонно возрастать.
        new_prob_ro = phi * get_ro(x, m, mu, sigma)
        new_prob_ro_tilde = phi * get_ro_tilde(x_tilde, z_tilde, m_tilde, mu, sigma)
        prev_ll = ll
        ll = np.sum(np.log(new_prob_ro / w) * w) + np.sum(np.log(new_prob_ro_tilde))
        it += 1
        # *** КОНЕЦ ВАШЕГО КОДА ***

    return w


# *** НАЧАЛО ВАШЕГО КОДА ***
# Вспомогательные функции
# *** КОНЕЦ ВАШЕГО КОДА ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Рисуем прогнозы GMM-модели на двухмерном датасете `x` с метками `z`.

    Записываем результат в выходной каталог, включив `plot_id`
    в имя файла, добавив 'ss', если результат получен гибридной версией алгоритма.

    Замечание: Вам не нужно править эту функцию.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('./stats/task33', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Загружаем данные для модели гауссовской смеси.

    Аргументы:
         csv_path: Пусть к CSV файлу, содержащему выборку.

    Возвращаемое значение:
        x: NumPy массив размерности (n_examples, dim)
        z: NumPy массив размерности (n_exampls, 1)

    Замечание: Вам не нужно править эту функцию.
    """

    # Загружаем заголовки
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Загружаем признаки и метки
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Запускаем NUM_TRIALS испытаний, чтобы посмотреть, как алгоритм ведет себя для разных начальных значений
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)

        # *** НАЧАЛО ВАШЕГО КОДА ***
        # После того как вы закончите реализацию гибридной версии EM-алгоритма,
        # раскомментируйте следующую строку.
        # Вам не нужно что-либо добавлять сюда еще.
        main(is_semi_supervised=True, trial_num=t)
        # *** КОНЕЦ ВАШЕГО КОДА ***

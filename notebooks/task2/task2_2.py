import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.optimize as op


def plot_data(x, y, title='Scatter plot of the training sample'):
    fig, ax = plt.subplots(figsize=(9, 7))
    data = np.hstack((x, y.reshape((len(y), 1))))
    false_data = data[np.in1d(data[:, 2], [0])]
    true_data = data[np.in1d(data[:, 2], [1])]
    ax.plot(false_data[:, 0], false_data[:, 1], 'o', color='red', label='y = 0')
    ax.plot(true_data[:, 0], true_data[:, 1], '+', color='green', label='y = 1')
    ax.set(xlabel='Microchip Test 1', ylabel='Microchip Test 2',
           title=title)
    ax.grid()
    ax.legend()
    plt.show()


def plot_data_with_decision(x, y, theta, title='Scatter plot of the training sample'):
    fig, ax = plt.subplots(figsize=(9, 7))
    data = np.hstack((x, y.reshape((len(y), 1))))
    false_data = data[np.in1d(data[:, 2], [0])]
    true_data = data[np.in1d(data[:, 2], [1])]
    ax.plot(false_data[:, 0], false_data[:, 1], 'o', color='red', label='y = 0')
    ax.plot(true_data[:, 0], true_data[:, 1], '+', color='green', label='y = 1')
    ax.set(xlabel='Microchip Test 1', ylabel='Microchip Test 2',
           title=title)

    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    # u, v = np.meshgrid(x[:, 0], x[:, 1])
    z = np.zeros((u.size, v.size))
    for i, u_i in enumerate(u):
        for j, v_j in enumerate(v):
            z[i, j] = map_feature(np.array([u_i]), np.array([v_j])) @ theta
    ax.contour(u, v, z.T, 0)

    ax.grid()
    ax.legend()
    plt.show()


def map_feature(x1, x2, degree=6):
    m, *_ = x1.shape
    out = np.ones(m).reshape((m, 1))
    for i in range(1, degree + 1):
        for j in range(i + 1):
            new_feature = (x1 ** (i - j) * x2 ** j).reshape((m, 1))
            out = np.hstack((out, new_feature))
    return out


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def h_func(theta, x):
    return sigmoid(theta.T @ x.T)


def predict(theta, x):
    return (h_func(theta, x) >= 0.5).astype(int)


#
#
# def cost_function_grad(theta, x, y):
#     theta = np.array(theta).reshape((len(theta), 1))
#     axis_m = np.sum((h_func(theta, x) - y) @ x, axis=0) / m
#     return axis_m
def cost_function_reg(theta, x, y, lambda_var):
    m, = y.shape
    theta = theta.reshape((len(theta), 1))
    cost_normal = np.sum(-y * np.log(h_func(theta, x)) - (1 - y) * np.log(1 - h_func(theta, x))) / m
    reg_coefficient = lambda_var / 2. / m * np.sum(theta[1:] ** 2)
    return cost_normal + reg_coefficient


def cost_function_grad_reg(theta, x, y, lambda_var):
    m, = y.shape
    theta = np.array(theta).reshape((len(theta), 1))
    axis_m = np.sum((h_func(theta, x) - y) @ x, axis=0) / m
    axis_m[1:] += (lambda_var / m * theta[1:]).flatten()
    return axis_m


def main():
    ex2data2 = np.loadtxt("./models/ex2data2.txt", delimiter=',')
    x_raw, y = ex2data2[:, 0:2], ex2data2[:, 2]
    m, n = x_raw.shape
    print(m, n)
    plot_data(x_raw, y)
    x = map_feature(x_raw[:, 0], x_raw[:, 1])
    _, features = x.shape
    initial_theta = np.zeros((features, 1))
    lambda_var = 1
    cost = cost_function_reg(initial_theta, x, y, lambda_var)
    grad = cost_function_grad_reg(initial_theta, x, y, lambda_var)
    print('Стоимость для начального значения theta (нулевого):', cost)
    print('Ожидаемая стоимость (примерно): 0.693')
    print('Градиент для начального значения theta (нулевого) - только первые значения:')
    print(' '.join(map(lambda num: str(round(float(num), 4)), grad[0:5])))
    print('Ожидаемые значения градиента (примерно) - только первые значения:')
    print('0.0085 0.0188 0.0001 0.0503 0.0115')
    print('===================================')
    initial_theta = np.ones((features, 1))
    lambda_var = 1
    cost = cost_function_reg(initial_theta, x, y, lambda_var)
    grad = cost_function_grad_reg(initial_theta, x, y, lambda_var)
    print('Стоимость для тестового значения theta (с lambda = 10):', cost)
    print('Ожидаемая стоимость (примерно): 3.16')
    print('Градиент для тестового значения theta - только первые значения:')
    print(' '.join(map(lambda num: str(round(float(num), 4)), grad[0:5])))
    print('Ожидаемые значения градиента (примерно) - только первые значения:')
    print('0.3460 0.1614 0.1948 0.2269 0.0922')

    theta = op.minimize(fun=cost_function_reg,
                        x0=initial_theta.T,
                        args=(x, y, lambda_var),
                        options={'maxiter': 400},
                        jac=cost_function_grad_reg).x
    plot_data_with_decision(x_raw, y, theta)

    print('lambda =', lambda_var)

    p = predict(theta, x)
    print('Точность на обучающей выборке:', np.mean(p == y) * 100)
    print('Ожидаемая точность (с lambda = 1): 83.1')


if __name__ == '__main__':
    main()

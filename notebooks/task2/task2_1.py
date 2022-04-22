import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.optimize as op

ex2data1 = np.loadtxt("../../models/ex2data1.txt", delimiter=',')

x, y = ex2data1[:, 0:2], ex2data1[:, 2]
m, n = x.shape
print(m, n)


def plot_data(x, y, title='Scatter plot of the training sample'):
    fig, ax = plt.subplots(figsize=(9, 7))
    data = np.hstack((x, y.reshape((len(y), 1))))
    false_data = data[np.in1d(data[:, 2], [0])]
    true_data = data[np.in1d(data[:, 2], [1])]
    ax.plot(false_data[:, 0], false_data[:, 1], 'o', color='red', label='Not admitted')
    ax.plot(true_data[:, 0], true_data[:, 1], '+', color='green', label='Admitted')
    ax.set(xlabel='Exam 1 score', ylabel='Exam 1 score',
           title=title)
    ax.grid()
    ax.legend()
    plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def vector_to_column(x):
    return x.reshape((x.shape[0], 1))


def vector_to_row(x):
    return x.reshape((1, x.shape[0]))


def predict(theta, x):
    return sigmoid(theta.T @ x.T)


def cost_function(theta, x, y):
    theta = theta.reshape((len(theta), 1))
    cost = np.sum(-y * np.log(predict(theta, x)) - (1 - y) * np.log(1 - predict(theta, x))) / m
    print(theta, cost, m)
    return cost


def cost_function_grad(theta, x, y):
    theta = np.array(theta).reshape((len(theta), 1))
    axis_m = np.sum((predict(theta, x) - y) @ x, axis=0) / m
    print('axis_m', axis_m)
    return axis_m


plot_data(x, y)


def intercept(t):
    return np.hstack((np.ones((m, 1)), t))


x = intercept(x)
initial_theta = np.zeros((n + 1, 1))
cost = cost_function(initial_theta, x, y)
grad = cost_function_grad(initial_theta, x, y)
print(f'Стоимость для начального значения theta (нулевого): {cost}')
print('Ожидаемая стоимость (примерно): 0.693')
print(f'Градиент для начального значения theta (нулевого): {grad}')
print('Ожидаемые значения градиента (примерно): -0.1000 -12.0092 -11.2628')

theta = op.minimize(fun=cost_function,
                    x0=initial_theta.T,
                    args=(x, y),
                    options={'maxiter': 400},
                    jac=cost_function_grad).x
cost = cost_function_grad(theta, x, y)
print(f'Стоимость для theta, найденная minimize: {cost}')
print('Ожидаемая стоимость (примерно): 0.203')
print(f'theta: {theta}')
print(f'Ожидаемое значение theta (примерно): -25.161 0.206 0.201')

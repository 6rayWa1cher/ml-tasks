from __future__ import division, print_function
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random


def init_centroids(num_clusters, image):
    """
    Инициализируйте np-массив размерности `num_clusters` x image_shape[-1] RGB цветами
    случайно выбранных пикселей картинки `image`

    Аргументы
    ----------
    num_clusters : int
        Количество центроидов/кластеров
    image : nparray
        (H, W, C) картинка, представленная в виде np-массива

    Возвращаемое значение
    -------
    centroids_init : nparray
        Случайным образом инициализированные центроиды
    """

    # *** НАЧАЛО ВАШЕГО КОДА ***
    chosen_indexes_h = np.atleast_2d(np.random.choice(image.shape[0], num_clusters, replace=False))
    chosen_indexes_w = np.atleast_2d(np.random.choice(image.shape[1], num_clusters, replace=False))
    centroids_init = image[chosen_indexes_h, chosen_indexes_w][0]
    # *** КОНЕЦ ВАШЕГО КОДА ***

    return centroids_init


def calc_distances(image, centroids):
    h, w, colors = image.shape
    centroid_count, _ = centroids.shape
    image_4d = image.reshape((h, w, colors, 1))
    centroids_4d = centroids.reshape((1, 1, centroid_count, colors)).transpose((0, 1, 3, 2))
    centroids_distance = np.sqrt(np.sum((image_4d - centroids_4d) ** 2, axis=2))
    return centroids_distance


def one_hot(x, depth: int):
    return np.take(np.eye(depth), x, axis=0)


def update_centroids(centroids, image, max_iter=30, print_every=10):
    """
    Выполните шаг обновления позиций центроидов алгоритма k-средних `max_iter` раз

    Аргументы
    ----------
    centroids : nparray
        np массив с центроидами
    image : nparray
        (H, W, C) картинка, представленная в виде np-массива
    max_iter : int
        Количество итераций алгоритма
    print_every : int
        Частота вывода диагностического сообщения

    Возвращаемое значение
    -------
    new_centroids : nparray
        Новые значения центроидов
    """

    # *** НАЧАЛО ВАШЕГО КОДА ***
    centroid_count, colors = centroids.shape
    h, w, _ = image.shape
    prev_centroids = centroids
    new_centroids = None
    for iteration in range(max_iter):
        if iteration % print_every == 0:
            print(prev_centroids)
        if new_centroids is not None:
            prev_centroids = new_centroids
        centroids_distance = calc_distances(image, prev_centroids)

        centroids_pixels = np.argmin(centroids_distance, axis=2)
        hot_centroid_pixels = one_hot(centroids_pixels, centroid_count)
        image_4d = image.reshape((h, w, colors, 1))
        hot_centroid_pixels_4d = hot_centroid_pixels.reshape((h, w, centroid_count, 1)).transpose((0, 1, 3, 2))
        prepared = image_4d * hot_centroid_pixels_4d
        new_centroids = np.mean(prepared, axis=(0, 1), where=prepared != 0).T
        if np.all(np.abs(prev_centroids - new_centroids) < 0.01):
            print("Stopped at", iteration + 1, "iteration")
            print(new_centroids)
            break
    # *** КОНЕЦ ВАШЕГО КОДА ***

    return new_centroids


def update_image(image, centroids):
    """
    Обновите RGB значения каждого пикселя картинки `image`, заменив его
    на значение ближайшего к нему центроида из `centroids`

    Аргументы
    ----------
    image : nparray
        (H, W, C) картинка, представленная в виде np-массива
    centroids : nparray
        np массив с центроидами

    Возвращаемое значение
    -------
    image : nparray
        Обновленное изображение
    """

    # *** НАЧАЛО ВАШЕГО КОДА ***
    centroids_distance = calc_distances(image, centroids)
    centroids_pixels = np.argmin(centroids_distance, axis=2)
    image = centroids[centroids_pixels].astype(int)
    # *** КОНЕЦ ВАШЕГО КОДА ***

    return image


def main(args):
    # Setup
    max_iter = args.max_iter
    print_every = args.print_every
    image_path_small = args.small_path
    image_path_large = args.large_path
    num_clusters = args.num_clusters
    figure_idx = 0

    # Load small image
    image = np.copy(mpimg.imread(image_path_small))
    print('[INFO] Loaded small image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original small image')
    plt.axis('off')
    savepath = os.path.join('./stats/task32/', 'orig_small.png')
    plt.savefig(savepath, transparent=True, format='png', bbox_inches='tight')

    # Initialize centroids
    print('[INFO] Centroids initialized')
    centroids_init = init_centroids(num_clusters, image)

    # Update centroids
    print(25 * '=')
    print('Updating centroids ...')
    print(25 * '=')
    centroids = update_centroids(centroids_init, image, max_iter, print_every)

    # Load large image
    image = np.copy(mpimg.imread(image_path_large))
    image.setflags(write=1)
    print('[INFO] Loaded large image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original large image')
    plt.axis('off')
    savepath = os.path.join('./stats/task32/', 'orig_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    # Update large image with centroids calculated on small image
    print(25 * '=')
    print('Updating large image ...')
    print(25 * '=')
    image_clustered = update_image(image, centroids)

    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image_clustered)
    plt.title('Updated large image')
    plt.axis('off')
    savepath = os.path.join('./stats/task32/', 'updated_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    print('\nCOMPLETE')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_path', default='./models/task32/peppers-small.tiff',
                        help='Path to small image')
    parser.add_argument('--large_path', default='./models/task32/peppers-large.tiff',
                        help='Path to large image')
    parser.add_argument('--max_iter', type=int, default=150,
                        help='Maximum number of iterations')
    parser.add_argument('--num_clusters', type=int, default=16,
                        help='Number of centroids/clusters')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    args = parser.parse_args()
    main(args)

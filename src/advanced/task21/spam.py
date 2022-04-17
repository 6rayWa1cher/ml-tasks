import numpy as np

import util
import svm


def get_words(message: str):
    """Возвращает список нормализованных слов, полученных из строки сообщения.

    Эта функция должна разбивать сообщение на слова, нормализовать их и возвращать
	получившийся список. Слова необходимо разбивать по пробелам. Под нормализацией
	понимается перевод всех букв в нижний регистр.

    Аргументы:
        message: Строка, содержащая SMS сообщение.

    Возвращаемое значение:
       Список нормализованных слов из текстового сообщения.
    """

    # *** НАЧАЛО ВАШЕГО КОДА ***
    def normalize(string: str):
        return ''.join(filter(lambda a: a.isalpha() or a == "'", string.lower()))

    def not_empty(string: str):
        return len(string) > 0

    return list(filter(not_empty, map(normalize, message.split())))
    # *** КОНЕЦ ВАШЕГО КОДА ***


def create_dictionary(messages):
    """Создает словарь, отображающий слова в целые числа.

    Данная функция должна создать словарь всех слов из сообщений messages,
	в котором каждому слову будет соответствовать порядковый номер.
	Для разделения сообщений на слова используйте функцию get_words.

    Редкие слова чаще всего бывают бесполезными при построении классификаторов. Пожалуйста,
	добавляйте слово в словарь, только если оно встречается минимум в пяти сообщениях.

    Аргументы:
        messages: Список строк с SMS сообщениями.

    Возвращаемое значение:
        Питоновская структура dict, отображающая слова в целые числа.
		Нумерацию слов нужно начать с нуля.
    """

    # *** НАЧАЛО ВАШЕГО КОДА ***
    stats = dict()
    threshold = 5
    for words in map(get_words, messages):
        for word in set(words):
            if word not in stats:
                stats[word] = 0
            stats[word] += 1

    last_index = -1

    def get_next_index():
        nonlocal last_index
        last_index += 1
        return last_index

    out = {word: get_next_index() for (word, count) in stats.items() if count >= threshold}
    return out
    # *** КОНЕЦ ВАШЕГО КОДА ***


def transform_text(messages, word_dictionary):
    """Трансформирует список текстовых сообщений в массив numpy, пригодный для дальнейшего использования.

    Эта функция должна создать массив numpy, содержащий количество раз, которое каждое слово
	словаря появляется в каждом сообщении.
	Строки в результирующем массиве должны соответствовать сообщениям из массива messages,
	а столбцы - словам из словаря word_dictionary.

    Используйте предоставленный словарь, чтобы сопоставлять слова с индексами столбцов.
	Игнорируйте слова, которых нет в словаре. Используйте get_words, чтобы разбивать сообщения на слова.

    Аргументы:
        messages: Список строк, в котором каждая строка является одним SMS сообщением.
        word_dictionary: Питоновский словарь dict, отображающий слова в целые числа.

    Возвращаемое значение:
        Массив numpy с информацией о том, сколько раз каждое слово встречается в каждом сообщении.
        Элемент (i,j) массива равен количеству вхождений j-го слова (по словарю) в i-м сообщении.
    """
    # *** НАЧАЛО ВАШЕГО КОДА ***
    matrix = np.zeros((len(messages), len(word_dictionary)))
    for i, words in enumerate(map(get_words, messages)):
        for word in words:
            if word in word_dictionary:
                word_index = word_dictionary[word]
                matrix[i, word_index] += 1
    return matrix
    # *** КОНЕЦ ВАШЕГО КОДА ***


def fit_naive_bayes_model(matrix, labels):
    """Обучает наивный байесовский классификатор.

    Эта функция должна обучить наивную байесовскую модель по переданной обучающей выборке.

    Функция должна возвращать построенную модель.

    Вы можете использовать любой тип данных, который пожелаете, для возвращения модели.

    Аргументы:
        matrix: Массив array, содержащий количества вхождений слов в сообщения из обучающей выборки.
        labels: Бинарные метки (0 или 1) для обучающей выборки.

    Возвращаемое значение: Обученный классификатор, использующий мультиномиальную модель событий и сглаживание Лапласа.
    ЗАМЕЧАНИЕ: обученная модель должна содержать два поля: vocab @ (2, V) и class @ (2,):
    vocab[i, k] хранит значения параметров P(x[j]=k | y=i), i принадлежит {0, 1},
    class[i] хранит значения параметров P(y=i), i принадлежит {0, 1}.
    """
    # *** НАЧАЛО ВАШЕГО КОДА ***
    m, v = matrix.shape
    labels = np.array(labels).reshape((len(labels), 1))
    phi_y = np.sum(labels) / m
    phi_k_1_dividend = 1 + np.sum(matrix, axis=0, where=labels == 1)
    phi_k_1_divisor = v + np.sum(matrix, where=labels == 1)
    phi_k_1 = phi_k_1_dividend / phi_k_1_divisor
    phi_k_0_dividend = 1 + np.sum(matrix, axis=0, where=labels == 0)
    phi_k_0_divisor = v + np.sum(matrix, where=labels == 0)
    phi_k_0 = phi_k_0_dividend / phi_k_0_divisor
    vocab = np.vstack((phi_k_0, phi_k_1))
    clazz = np.array([1 - phi_y, phi_y])
    return vocab, clazz
    # *** КОНЕЦ ВАШЕГО КОДА ***


def predict_from_naive_bayes_model(model, matrix):
    """Используя функцию гипотезы наивного байесовского классификатора, выдает прогнозы для матрицы с данными matrix.

    Данная функция должна выдавать прогнозы, используя передаваемую ей из fit_naive_bayes_model модель классификатора.

    Аргументы:
        model: Обученная функцией fit_naive_bayes_model модель.
        matrix: Массив numpy, содержащий количества слов.

    Возвращаемое значение: Массив numpy с прогнозами наивного байесовского классификатора.
    """
    # *** НАЧАЛО ВАШЕГО КОДА ***
    vocab, clazz = model
    # # message, word, class
    # vocab_3d = vocab.reshape((1, vocab.shape[1], vocab.shape[0]))
    # matrix_3d = matrix.reshape((*matrix.shape, 1))
    # log_voc_mat = np.log(1 + vocab_3d * matrix_3d)
    # false_sum_log = np.sum(log_voc_mat, axis=1)
    # false_total_score = false_sum_log[:, 0] - false_sum_log[:, 1]
    # # return (total_score > 0).astype(int)

    log_voc_mat_0 = np.log(1 + vocab[0] * matrix)
    sum_log_0 = np.sum(log_voc_mat_0, axis=1)
    log_voc_mat_1 = np.log(1 + vocab[1] * matrix)
    sum_log_1 = np.sum(log_voc_mat_1, axis=1)
    total_score = sum_log_1 * clazz[1] - sum_log_0 * clazz[0]
    return (total_score > 0).astype(int)
    # *** КОНЕЦ ВАШЕГО КОДА ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Определяет пять слов, наиболее характерных для спам сообщений.

    Используйте метрику, приведенную в теоретическом материале, чтобы понять, насколько данное
	конкретное слово хакактерно для того или иного класса.
    Верните список слов, отсортированный в порядке убывания "характерности".

    Аргументы:
        model: Обученная функцией fit_naive_bayes_model модель.
        dictionary: Питоновский словарь dict, отображающий слова в целые числа.

    Возвращаемое значение: список слов, отсортированный в порядке убывания "характерности".
    """
    # *** НАЧАЛО ВАШЕГО КОДА ***
    vocab, clazz = model
    stats = np.log(vocab[1] / vocab[0])
    word_stats = [(stats[index], word) for (word, index) in dictionary.items()]
    word_stats.sort(reverse=True)
    return list(map(lambda tpl: tpl[1], word_stats[:5]))
    # *** КОНЕЦ ВАШЕГО КОДА ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Вычисляет оптимальный SVM радиус, используя предоставленную обучающую и валидационную выборки.

    Вы должны исследовать только те значения радиусов, которые переданы в списке radius_to_consider.
	Вы должны использовать точность классификации в качестве метрики для сравнения различных значений радиусов.

    Аргументы:
        train_matrix: Матрица с частотами слов для обучающей выборки.
        train_labels: Метки "спам" и "не спам" для обучающей выборки.
        val_matrix: Матрица с частотами слов для валидационной выборки.
        val_labels: Метки "спам" и "не спам" для валидационной выборки.
        radius_to_consider: Значения радиусов, среди которых необходимо искать оптимальное.

    Возвращаемое значение:
        Значение радиуса, при котором SVM достигает максимальной точности.
    """
    # *** НАЧАЛО ВАШЕГО КОДА ***
    curr_max = radius_to_consider[0]
    curr_max_acc = 0
    for radius in radius_to_consider:
        predicted_labels = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, radius)
        accuracy = np.mean(val_labels == predicted_labels)
        if accuracy > curr_max_acc:
            curr_max_acc = accuracy
            curr_max = radius
    return curr_max
    # *** КОНЕЦ ВАШЕГО КОДА ***


def main():
    train_messages, train_labels = util.load_spam_dataset('models/spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('models/spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('models/spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    print('Размер словаря: ', len(dictionary))

    util.write_json('spam_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('spam_sample_train_matrix', train_matrix[:100, :])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Наивный Байес показал точность {} на тестовой выборке'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('Пять наиболее характерных для спама слов: ', top_5_words)

    util.write_json('spam_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('spam_optimal_radius', optimal_radius)

    print('Оптимальное значение SVM-радиуса {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('SVM модель имеет точность {} на тестовой выборке'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()

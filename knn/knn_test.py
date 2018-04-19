
from knn import Knn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.io.arff import loadarff


def get_accuracy(k, trainx, trainy, testx, testy):
    knn = Knn(k)
    knn.fit(trainx, trainy)
    hyp = knn.predict(testx)
    return accuracy_score(hyp, testy)


def extract_data(file):
    data, meta = loadarff(file)
    converted_data = np.ones((data.shape[0], len(data[0])))
    for i in range(data.shape[0]):
        for j in range(len(data[i])):
            converted_data[i, j] = data[i][j]

    return converted_data


if __name__ == '__main__':
    # legitimate: 1, suspicious: 0, phishy: -1
    data = extract_data('PhishingData.arff')
    train_len = int(data.shape[0] * .8)

    train_x = data[:train_len, :-1]
    train_y = data[:train_len, -1]
    test_x = data[train_len:, :-1]
    test_y = data[train_len:, -1]

    for k in range(2, 33):
        acc = get_accuracy(k, np.copy(train_x), np.copy(train_y), np.copy(test_x), np.copy(test_y))
        print('K: {0} Accuracy: {1:4f}'.format(k, acc))
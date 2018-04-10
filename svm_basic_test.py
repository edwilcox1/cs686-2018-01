from svm_basic import svm_basic
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


def get_data_and_labels(filename, sep=','):
    df = pd.read_csv(filename, header=None, sep=sep)
    x = df.iloc[:, :2]
    y = df.iloc[:, 2]
    return x, y


def plot_data(data_mat, labels, weights=None, b=None, plot_line=False, name='tmp'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    data_array = np.asarray(data_mat)
    n = data_array.shape[0]
    xcord1 = []
    xcord2 = []
    ycord1 = []
    ycord2 = []
    for i in range(n):
        if int(labels[i]) == 1:
            xcord1.append(data_array[i, 0])
            ycord1.append(data_array[i, 1])
        else:
            xcord2.append(data_array[i, 0])
            ycord2.append(data_array[i, 1])

    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    if weights is not None and plot_line:
        min_x = min(np.min(xcord1), np.min(xcord2))
        max_x = max(np.max(xcord1), np.max(xcord2))
        min_y = min(np.min(ycord1), np.min(ycord2))
        max_y = max(np.max(ycord2), np.max(ycord1))
        x = np.arange(-2, max_x + 3, 0.1)
        a = -weights[0]/weights[1]
        y = a * x - b[0,0]/weights[1]

        ax.fill_between(x, y, y2=min_y - 1, color='red', alpha=0.5)
        ax.fill_between(x, y, y2=max_y + 1, color='blue', alpha=0.5)

        plt.tight_layout()
        ax.axis([min_x - 1, max_x + 1, min_y - 1, max_y + 1])
        ax.plot(x, y)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(name)


if __name__ == '__main__':
    svm_classifier = svm_basic(maxIter=100)
    X, Y = get_data_and_labels('linearly_separable.csv')
    plot_data(X, Y, name='')
    weights, b = svm_classifier.fit(X, Y)
    plot_data(X, Y, weights, b, plot_line=True)



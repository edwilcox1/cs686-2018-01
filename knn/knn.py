
from ..classifier import classifier

class Knn(classifier):
    from scipy.spatial import distance, cKDTree
    import numpy as np

    def __init__(self, k=3):
        self.X = None
        self.Y = None
        self.unique_classes = None
        self.k = k

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.unique_classes = self.Y.unique()

    def predict(self, X):
        if self.X is None or self.Y is None:
            print('No training data has been provided through fit()')
            return

        hypotheses = []
        for t in X:
            points = self.__sort(self.__nearest_points(t))
            neighbors = self.__get_top(self.k, points)
            hyp = self.__majority_class(neighbors)
            hypotheses.append(hyp)
        return hypotheses

    def __sort(self, points):
        pass

    def __nearest_points(self, points):
        """

        :param points:
        :return: Indices of nearest points
        """
        pass

    def __get_top(self, k, points):
        try:
            return points[:k + 1]
        except IndexError:
            print('Invalid k provided: ' + k)
            raise IndexError

    def __majority_class(self, neighbors):
        class_count = {class_name: 0 for class_name in self.unique_classes}
        for neighbor in neighbors:
            class_count[self.Y[neighbor]] += 1

        return max

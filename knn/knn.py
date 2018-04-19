

import numpy as np

from classifier import classifier


class Knn(classifier):

    def __init__(self, k=3):
        self.X = None
        self.Y = None
        self.unique_classes = None
        self.k = k

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.unique_classes = np.unique(self.Y)

    def predict(self, X):
        from scipy.spatial import distance
        if self.X is None or self.Y is None:
            print('No training data has been provided through fit()')
            return

        hypotheses = []
        for t in X:
            distances = distance.cdist(np.array([t]), self.X)  # Need row t to be in 2d array for cdist
            neighbors = np.argpartition(distances, self.k)  # Returns indices of points
            neighbors = neighbors[0, :self.k]  # neighbors[0, :self.k]
            hyp = self.__majority_class(neighbors)
            hypotheses.append(hyp)
        return hypotheses

    def __get_distances(self, point):
        """Alternative function to cdist to check accuracy discrepancies"""
        return [np.linalg.norm(point - x) for x in self.X]

    def __majority_class(self, neighbors):
        import operator
        class_count = {class_name: 0 for class_name in self.unique_classes}
        for neighbor in neighbors:
            class_count[self.Y[neighbor]] += 1

        # Get majority class
        max_class = None
        max_count = 0
        for key, val in class_count.items():
            if val >= max_count:  # Taking last on tie
                max_count = val
                max_class = key

        return max_class

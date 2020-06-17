import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics


class LinearRegressor:
    def __init__(self, lr=0.01, num_iter=100000, normalize=True, weights=None, fit_intercept=True, log=False):
        self.weights = weights
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.log = log
        self.normalize = normalize
        self.cost_history = list()

    @staticmethod
    def __add_intercept(X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __cost_function(self, features, targets, weights):

        """returns average squared error among predictions"""
        N = len(targets)

        predictions = self.predict(features)

        # Matrix math lets use do this without looping
        sq_error = (predictions - targets) ** 2

        # Return average squared error among predictions
        return 1.0 / (2 * N) * sq_error.sum()

    @staticmethod
    def __normalize(features: pd.DataFrame) -> pd.DataFrame:
        """
        TODO: correct implementation(currently not working)
        :return: pd.DataFrame
        :type features: pd.DataFrame
        """
        # We transpose the input matrix, swapping
        # cols and rows to make vector math easier

        for feature in features:
            f = features[feature].values
            fMean = np.mean(f)
            fRange = np.amax(f) - np.amin(f)

            # Vector Subtraction (mean Normalization)
            f = f - fMean
            # np.subtract(f, fMean, out=f, casting='same_kind')

            # Vector Division (feature scaling)
            if fRange:
                f = f / fRange
                # np.divide(f, fRange, out=f, casting='unsafe')
            features.loc[:, feature] = f
        return features

    def __update_weights(self, features, targets, weights, lr):
        """gradient = features.T * (predictions - targets) / N"""

        noOfDataValues = len(features)
        # 1 - Get Predictions
        predictions = self.predict(features)
        # 2 - Calculate error/loss
        error = targets - predictions
        gradient = np.dot(-features.T, error)
        # 4 Take the average error derivative for each feature
        gradient /= noOfDataValues
        # 5 - Multiply the gradient by our learning rate
        gradient *= lr
        # 6 - Subtract from our weights to minimize cost
        weights -= gradient

        return weights

    def predict(self, features):
        if features.shape[1] != self.weights.shape[0]:
            features = self.__add_intercept(features)
        predictions = np.dot(features, self.weights)
        return predictions

    def train(self, features: np.ndarray, targets: np.ndarray) -> (
            np.ndarray, list):
        """
        fits the data into model and calculates weights to make accurate predictions
        :param features: list of independent_variable values
        :param targets: list of dependent_variable values
        """
        if self.fit_intercept:
            features = self.__add_intercept(features)

        self.weights = np.zeros((features.shape[1], 1)) if self.weights is None else self.weights

        if not self.normalize:
            features = self.__normalize(pd.DataFrame(features))

        for i in range(self.num_iter):

            self.weights = self.__update_weights(features, targets, self.weights, self.lr)

            # Calculate cost for auditing purposes
            cost = self.__cost_function(features, targets, self.weights)
            self.cost_history.append(cost)

            if self.log:
                # Log Progress
                if i % 100 == 0:
                    print("iter={0}    Weights={1}      cost={2}".format(i, self.weights, cost))
        breakpoint()
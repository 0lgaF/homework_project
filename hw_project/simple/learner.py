import pandas as pd
import numpy as np

class Train(object):
    def __call__(self, model, X, y):
        """learns model on dataset
        :param X: features matrix
        :param y: targets
        :param model: sklearn estimator
        :return: fitted model
        """
        return model.fit(X, y)


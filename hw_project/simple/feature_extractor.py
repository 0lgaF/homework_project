import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator

class Feature_extractor(BaseEstimator):
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, X, y=None):
        """Compute means and stds on train
        :param X: features matrix
        """
        self.mean = np.mean(X, axis = 0)
        self.std = np.std(X, axis = 0)
        pass
    
    def transform(self, X, y=None):
        """remove the mean and divide by std
        :param X: features matrix
        """
        return (X-self.mean)/self.std
  
    def fit_transform(self, X, y=None):
        """Compute means and stds on train and then
        remove the mean and divide by std
        :param X: features matrix
        """
        self.mean = np.mean(X, axis = 0)
        self.std = np.std(X, axis = 0)
        return (X-self.mean)/self.std


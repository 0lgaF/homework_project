#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

def make_preds(model, X):
    """make predictions on dataset
        :param X_data: features matrix
        :return: predictions
        """
    return model.predict(X)


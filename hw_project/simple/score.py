import pandas as pd
import numpy as np

def precision_score(true, preds):
    """Compute the precision"""
    return np.sum(true == preds)/len(preds)


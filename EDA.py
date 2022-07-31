import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from sklearn.datasets import *

def load_data(dataset):
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['Target']  = dataset.target
    return df

if __name__ == '__main__':
    print(load_data(load_iris()))
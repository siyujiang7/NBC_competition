import pandas as pd
import numpy as np
import math
import sys
import random
def nbc(percent):
    fileName = "HeadLine_Trainingdata.csv"
    data = pd.read_csv('HeadLine_Trainingdata.csv')
    X = data.as_matrix()
    random.shuffle(X)
    percent = percent/100
    n = len(X)
    train_set = X[:int(percent * n)]
    test_set = X[int(percent * n):]
    np.savetxt("train_nbc.csv", train_set, delimiter=",", fmt="%s")
    np.savetxt("test_nbc.csv", test_set, delimiter=",", fmt="%s")

nbc(80)
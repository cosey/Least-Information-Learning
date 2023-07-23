import numpy as np
import sklearn.metrics as MI
from sklearn.metrics.cluster.supervised import mutual_info_score

def mutual_information(fig1,fig2):
    x = np.reshape(fig1,-1)
    y = np.reshape(fig2,-1)
    return MI.mutual_info_score(x,y)



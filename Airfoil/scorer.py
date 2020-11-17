import numpy as np
from sklearn import metrics
class Scorer():
    def __init__(self,y_pre, y_true):
        self.y_pre = y_pre
        self.y_true = y_true
    def __call__(self):
        return self.MAPE(),self.RMSE()
    def MAPE(self):
        return (abs(self.y_pre - self.y_true) / self.y_true).sum() / len(self.y_pre) * 100
    def RMSE(self):
        return np.sqrt(metrics.mean_squared_error(self.y_true, self.y_pre))
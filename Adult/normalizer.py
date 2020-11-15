import pandas as pd
import numpy as np
class Normalizer(object):
    def __init__(self,data=[], tags=[]):
        self.data = data
        self.tags = tags

    def __call__(self):
        self.z_score()

    def set_data(self, data, tags):
        self.data = data

    def z_score(self):
        if(len(self.data) == 0):
            return []
        for i in range(len(self.data)):
            for c in self.tags:
                self.data[i][c] = (self.data[i][c] - self.data[i][c].mean()) / np.sqrt(self.data[i][c].var())
        return self.data

    def max_min(self):
        if(len(self.data) == 0):
            return []
        for i in range(len(self.data)):
            for c in self.tags:
                self.data[i][c] = (self.data[i][c] - self.data[i][c].min()) / (self.data[i][c].max() - self.data[i][c].min())
        return self.data
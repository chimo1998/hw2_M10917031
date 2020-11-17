import uci_loader
import pandas as pd
class Info():
    def __init__(self):
        self.url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat'
        self.names = ['freq','angle','length','velocity','thickness','sound']
        self.drop = []
        self.encode_tag = []
        self.label_tag = 'sound'
        self.normalize_tag = ['freq','angle','length','velocity','thickness',]
    def load(self):
        uci = uci_loader.Loader(self.url, self.names, self.label_tag, drop_tags=self.drop, 
                encode_tags=self.encode_tag, normal_tags=self.normalize_tag)
        return uci()
    def __call__(self):
        return self.load()
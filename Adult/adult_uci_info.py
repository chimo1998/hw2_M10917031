import uci_loader
import pandas as pd
class Adult():
    def __init__(self):
        self.url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
        self.names = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','salary']
        self.drop = []#'fnlwgt']
        self.encode_tag = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'salary']
        self.label_tag = 'hours-per-week'
        self.normalize_tag = ['age','fnlwgt','education-num','capital-gain','capital-loss']
    def load(self):
        uci = uci_loader.Loader(self.url, self.names, self.label_tag, drop_tags=self.drop, 
                encode_tags=self.encode_tag, normal_tags=self.normalize_tag)
        return uci()
    def __call__(self):
        return self.load()

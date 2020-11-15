import pandas as pd
from sklearn.preprocessing import OneHotEncoder as ohe
import numpy as np
from sklearn.model_selection import train_test_split as tts
from normalizer import Normalizer

class Loader(object):
    def __init__(self, url, names, label_tag, drop_tags=None, encode_tags=None, normalizer=Normalizer(), normal_tags=None, test_size=0.2):
        self.url = url
        self.names = names
        self.drop_tags = drop_tags
        self.encode_tags = encode_tags
        self.data = None
        self.label_tag = label_tag
        self.test_size = test_size
        self.enc = ohe(categories='auto')
        self.normal_tags = normal_tags
        self.normalizer = normalizer

    def __call__(self, normal=False):
        return self.load()

    def read_from_url(self):
        self.data = pd.read_csv(self.url, names=self.names)
 
    def drop(self):
        self.data = self.data.drop(self.drop_tags, axis=1)

    def encode(self):
        for tag in self.encode_tags:
            t = pd.DataFrame(data=(self.enc.fit_transform(self.data[tag].to_numpy().reshape(-1,1)).toarray()))
            names = t.columns.tolist()
            new_names = dict(zip(names, ["%s_%s" % (tag, x) for x in names]))
            t = t.rename(columns=new_names)
            self.data = pd.concat([self.data.drop(tag, axis=1), t], axis=1)

    def get_data(self):
        train, test = tts(self.data, test_size=0.2)
        trainX = train.drop(self.label_tag, axis=1).sort_index()
        trainY = train[self.label_tag].sort_index()
        testX = test.drop(self.label_tag, axis=1).sort_index()
        testY = test[self.label_tag].sort_index()
        if not self.is_num(trainY[0]):
            trainY = pd.DataFrame(data=(self.enc.fit_transform(trainY.to_numpy().reshape(-1,1)).toarray())).sort_index()
            testY = pd.DataFrame(data=(self.enc.fit_transform(testY.to_numpy().reshape(-1,1)).toarray())).sort_index()
        return trainX, trainY, testX, testY

    def normal(self):
        self.normalizer.set_data([self.data],self.normal_tags)
        self.normalizer()

    def load(self):
        self.read_from_url()
        if self.drop_tags:
            self.drop()
        if self.encode_tags:
            self.encode()
        if self.normal_tags:
            self.normal()
        return self.get_data()

    def is_num(self, v):
        result = True
        try:
            tmp = int(v)
            result = True
        except:
            result = False
        return result
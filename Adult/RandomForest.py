import numpy as np
import adult_uci_info
from sklearn.ensemble import RandomForestClassifier as RFC

uci = adult_uci_info.Adult()
train_x, train_y, test_x, test_y = uci()
train_y = train_y.iloc[:,0]
test_y = test_y.iloc[:,0]
rfc = RFC(n_estimators=110, max_depth=5, max_features=0.7, n_jobs=-1)
rfc.fit(train_x, train_y)
y_pre = rfc.predict(test_x)
temp = y_pre == test_y
acc = temp.sum() / len(test_y)
print(acc)

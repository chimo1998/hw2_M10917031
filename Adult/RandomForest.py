import numpy as np
import adult_uci_info
from sklearn.ensemble import RandomForestRegressor as RFR

uci = adult_uci_info.Adult()
train_x, train_y, test_x, test_y = uci()
rfr = RFR(n_estimators=30, max_depth=4, max_features=0.6, n_jobs=-1)
rfr.fit(train_x, train_y)
y_pre = rfr.predict(test_x)
print(y_pre, "===========", test_y)
temp = y_pre == test_y
acc = temp.sum() / len(test_y)
print(acc)
print("====================")
print("score:", rfr.score(test_x, test_y))
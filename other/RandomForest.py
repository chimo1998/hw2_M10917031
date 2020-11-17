import numpy as np
from uci_info import Info
from scorer import Scorer
from sklearn.ensemble import RandomForestRegressor as RFR

uci = Info()
train_x, train_y, test_x, test_y = uci()
rfr = RFR(n_estimators=30, max_depth=4, max_features=0.6, n_jobs=-1)
rfr.fit(train_x, train_y)
y_pre = rfr.predict(test_x)
scorer = Scorer(y_pre, test_y)
mape, rmse = scorer()
print("rmse %f" % rmse, "mape %f" % mape)
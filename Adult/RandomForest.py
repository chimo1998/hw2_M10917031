import numpy as np
from adult_uci_info import Adult
from scorer import Scorer
from sklearn.ensemble import RandomForestRegressor as RFR

uci = Adult()
train_x, train_y, test_x, test_y = uci()
rfr = RFR(n_estimators=30, max_depth=4, max_features=0.6, n_jobs=-1)
rfr.fit(train_x, train_y)
y_pre = rfr.predict(test_x)
scorer = Scorer(y_pre, test_y)
rmse, mape = scorer()
print("rmse %f" % rmse, "mape %f" % mape)
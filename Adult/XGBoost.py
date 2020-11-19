from xgboost import XGBRegressor as XGBR
from adult_uci_info import Adult
import numpy as np
from scorer import Scorer
xgb = XGBR(max_depth=4, learning_rate=0.15, n_estimators=150, silent=True, objective='reg:gamma')
uci = Adult()
train_x, train_y, test_x, test_y = uci()
xgb.fit(train_x, train_y)
y_pre = xgb.predict(test_x)
scorer = Scorer(y_pre, test_y)
mape, rmse = scorer()
print("rmse %f" % rmse, "mape %f" % mape)
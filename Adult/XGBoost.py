from xgboost import XGBRegressor as XGBR
from adult_uci_info import Adult
import numpy as np
from scorer import Scorer
# xgb = XGBR(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#        colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
#        max_depth=10, min_child_weight=1, missing=None, n_estimators=100,
#        n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
#        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#        silent=True, subsample=1)
#xgb = XGBC()
xgb = XGBR(max_depth=4, learning_rate=0.15, n_estimators=150, silent=True, objective='reg:gamma')
uci = Adult()
train_x, train_y, test_x, test_y = uci()
xgb.fit(train_x, train_y)
y_pre = xgb.predict(test_x)
scorer = Scorer(y_pre, test_y)
print(y_pre)
rmse, mape = scorer()
print("rmse %f" % rmse, "mape %f" % mape)
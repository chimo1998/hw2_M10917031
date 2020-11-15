from xgboost import XGBRegressor as XGBR
import adult_uci_info
import numpy as np
# xgb = XGBR(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#        colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
#        max_depth=10, min_child_weight=1, missing=None, n_estimators=100,
#        n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
#        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#        silent=True, subsample=1)
#xgb = XGBC()
xgb = XGBR(max_depth=3, learning_rate=0.1, n_estimators=50, silent=True, objective='reg:gamma')
uci = adult_uci_info.Adult()
train_x, train_y, test_x, test_y = uci()
xgb.fit(train_x, train_y)
y_pre = xgb.predict(test_x)
print(y_pre, "============", test_y)
print(xgb.score(test_x, test_y))

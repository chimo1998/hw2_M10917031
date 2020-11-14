from xgboost import XGBClassifier as XGBC
import adult_uci_info
import numpy as np
xgb = XGBC()
uci = adult_uci_info.Adult()
train_x, train_y, test_x, test_y = uci()
train_y = train_y.iloc[:,0]
test_y = test_y.iloc[:,0]
xgb.fit(train_x, train_y)
print(xgb.score(test_x, test_y))

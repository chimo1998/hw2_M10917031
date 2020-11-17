from adult_uci_info import Adult
from sklearn.svm import SVR
from scorer import Scorer

uci = Adult()
train_x, train_y, test_x, test_y = uci()
svr = SVR(kernel='poly',gamma='auto',C=10)
svr.fit(train_x, train_y)
y_pre = svr.predict(test_x)
scorer = Scorer(y_pre, test_y)
rmse, mape = scorer()
print("rmse %f" % rmse, "mape %f" % mape)
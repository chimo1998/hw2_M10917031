from uci_info import Info
from sklearn.svm import SVR
from scorer import Scorer

uci = Info()
train_x, train_y, test_x, test_y = uci()
svr = SVR(kernel='rbf',gamma='auto',C=1)
print("a")
svr.fit(train_x, train_y)
print("b")
y_pre = svr.predict(test_x)
scorer = Scorer(y_pre, test_y)
mape, rmse = scorer()
print("rmse %f" % rmse, "mape %f" % mape)

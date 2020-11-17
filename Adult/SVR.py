import adult_uci_info
from sklearn.svm import SVR

uci = adult_uci_info.Adult()
train_x, train_y, test_x, test_y = uci()
svr = SVR(kernel='poly',gamma='auto',C=10)
svr.fit(train_x, train_y)
y_pre = svr.predict(test_x)
u = ((test_y - y_pre)**2).sum()
v = (test_y - test_y.mean()).sum()
print(1-(u/v))
print(y_pre - test_y)

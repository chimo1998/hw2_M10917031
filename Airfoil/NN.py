from keras.models import Sequential
from keras.layers import Dense,Activation
from uci_info import Info
from scorer import Scorer
import pandas as pd

uci = Info()
train_x, train_y, test_x, test_y = uci()

model = Sequential()
model.add(Dense(32,activation='relu', input_dim=len(train_x.columns)))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(train_x, train_y, batch_size = 6, epochs = 150)
y_pre = model.predict(test_x).reshape(-1)
scorer = Scorer(y_pre, test_y)
mape, rmse = scorer()
print("rmse %f" % rmse, "mape %f" % mape)
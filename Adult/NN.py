from keras.models import Sequential
from keras.layers import Dense,Activation
import adult_uci_info

uci = adult_uci_info.Adult()
train_x, train_y, test_x, test_y = uci()

model = Sequential()
model.add(Dense(32,activation='relu', input_dim=len(train_x.columns)))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(train_x, train_y, batch_size = 10, epochs = 20)
y_pred = model.predict(test_x)

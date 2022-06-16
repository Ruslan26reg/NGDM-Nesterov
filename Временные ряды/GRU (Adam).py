import numpy as np
import tensorflow 
from tensorflow.keras import Sequential, layers
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 16, 10

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tensorflow.random.set_seed(RANDOM_SEED)

time = np.arange(0, 100, 0.1)
sin = np.sin(time) + np.random.normal(scale=0.5, size=len(time))

plt.plot(time, sin, 'b--',label='зашумленная синусоида')
plt.legend()
plt.show()

df = pd.DataFrame(dict(sine=sin), index=time, columns=['sine'])
df.head()

train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
print(len(train), len(test))

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)        
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 10

# reshape to [samples, time_steps, n_features]

X_train, y_train = create_dataset(train, train.sine, time_steps)
X_test, y_test = create_dataset(test, test.sine, time_steps)

print(X_train.shape, y_train.shape)

from Apollo_tf import*
from ngd import*

optimizer = tensorflow.keras.optimizers.Adam(0.001)
model = Sequential()
model.add(layers.GRU(128, activation = "tanh",input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(layers.Dense(1))
model.compile(loss='mean_squared_error', optimizer=optimizer)

history = model.fit(
    X_train, y_train, 
    epochs=30, 
    batch_size=16, 
    validation_split=0.1, 
    verbose=1, 
    shuffle=False
)

y_pred = model.predict(X_test)

plt.plot(np.arange(0, len(y_train)), y_train, 'g--', label="history")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test, 'b--', label="true")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_pred, 'r.-', label="prediction")
plt.ylabel('Значение')
plt.xlabel('Время')
plt.legend()
plt.show()

plt.plot(y_test, 'b--', label="зашумленная синусоида")
plt.plot(y_pred, 'r.-', label="прогноз")
plt.ylabel('Значение')
plt.xlabel('Время')
plt.legend()
plt.show();

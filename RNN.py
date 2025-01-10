import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam

def gen_data(x):
    return (x % pitch) / pitch

def convertToMatrix(data, step=1):
    X, Y = [], []
    for i in range(len(data) - step):
        d = i + step
        X.append(data[i:i + step])  
        Y.append(data[d])
    return np.array(X), np.array(Y)

pitch = 20  
step = 3
N = 100  
n_train = int(N * 0.7)  

t = np.arange(1, N + 1)

# EX 1
# y = np.sin(0.1 * t * 10) + 0.5 * np.random.rand(N)

# EX 2
# y = np.cos(0.05 * t * 10) + 0.8 * np.random.rand(N)

# EX 3
y = np.sin(0.05 * t * 10) + 0.5 * np.random.rand(N)

# EX 4
# y = t**2 + 0.8 * np.random.rand(N)

# EX 5
# y = np.sin(0.05 * t * 10) + 1.5 * np.random.rand(N)

train, test = y[:n_train], y[n_train:]
x_train, y_train = convertToMatrix(train, step)
x_test, y_test = convertToMatrix(test, step)

x_train = x_train.reshape((x_train.shape[0], step, 1))
x_test = x_test.reshape((x_test.shape[0], step, 1))

model = Sequential()
model.add(SimpleRNN(units=250, input_shape=(step, 1), activation="relu"))  
model.add(Dense(units=1))
model.compile(optimizer=Adam(learning_rate=0.00005), loss="mse", metrics=["accuracy"])
hist = model.fit(x_train, y_train, epochs=100, batch_size=5, verbose=1)
y_pred_train = model.predict(x_train).flatten()
y_pred_test = model.predict(x_test).flatten()

y_pred_combined = np.concatenate((
    [np.nan] * step, 
    y_pred_train,    
    [np.nan] * (len(y) - len(y_pred_train) - len(y_pred_test) - step),  
    y_pred_test       
))

def plot_prediction(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label="Original", color="blue")  
    plt.plot(y_pred, label="Predict", color="red", linestyle="--")  
    plt.axvline(x=len(y_true) * 0.7, color="magenta", linestyle="-") 
    plt.legend()
    plt.show()

plot_prediction(y, y_pred_combined)

plt.figure()
plt.plot(hist.history['loss'])
plt.legend()
plt.show()

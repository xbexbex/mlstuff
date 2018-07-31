from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pylab

pylab.rcParams['figure.figsize'] = [12.0, 8.0]

import keras

from keras import backend as K

# Load Sequential model architecture
from keras.models import Sequential

# Load Dense and Dropout layers ?
from keras.layers import Dense, Dropout

# Load RMSprop optimizer to minimize cost to train the network
from keras.optimizers import SGD, RMSprop, Adam

# Load data
boston = load_boston()

# Choose AVG number of rooms as feature
X = boston.data[:, boston.feature_names.tolist().index('RM')]

# Target / desired output
y = boston.target

def plot_prediction_error(x_test, y_test, y_test_pred):
    pylab.plot([x_test, x_test], [y_test, y_test_pred], 'r-')

def manual(w, b):
    pylab.plot(X, y, '.')
    pylab.grid()
    pylab.xlabel('Average number of rooms')
    pylab.ylabel('Average price of the house')
    pylab.savefig('house_num_rooms_vs_price.png')

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def predict(x, w, b):
        return w*x + b

    y_pred = predict(x_train, w, b)

    pylab.figure()
    xs = np.linspace(X.min(), X.max(), 100)
    pylab.plot(xs, predict(xs, w, b))
    pylab.plot(x_train, y_train, 'b+')
    #pylab.plot(x_test, y_test, '.')
    pylab.xlabel('Average number of rooms')
    pylab.ylabel('Average price of the house')
    plot_prediction_error(x_train, y_train, predict(x_train, w, b))
    pylab.legend(['Model', 'Train', 'Test', 'Error'])
    pylab.grid()
    ws = np.linspace(-10,40,100)
    for b in [0]: #[-20, -10, 0, 10, 30]:
        errs = []
        for w in ws:
            errs.append(mean_squared_error(y_train, predict(x_train, w, b)))
        # Visualise error function
        pylab.plot(ws, errs)
    #pylab.legend(['bias = -20', 'bias = -10', 'bias = 0', 'bias = 10', 'bias = 20'])
    pylab.legend(['bias = 0'])
    pylab.grid()
    pylab.xlabel('Weight w')
    pylab.ylabel('Cost: Mean squared error')
    pylab.show()

def auto1():
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(x_train.shape)
    clf = LinearRegression()
    clf.fit(x_train.reshape(-1,1), y_train)
    w = clf.coef_
    b = clf.intercept_
    print(w, " + ", b)
    train_error = mean_squared_error(y_train, clf.predict(x_train.reshape(-1,1)))
    test_error = mean_squared_error(y_test, clf.predict(x_test.reshape(-1,1)))
    print('Train error=%f test error=%f' % (train_error, test_error))

    pylab.figure()

    # Generate data to visualise model
    xs = np.linspace(X.min(), X.max(), 100)

    # Plot stuff
    pylab.plot(xs, clf.predict(xs.reshape(-1, 1)))
    pylab.plot(x_test, y_test, 'g.')
    pylab.plot(x_train, y_train, 'b+')
    plot_prediction_error(x_train, y_train, clf.predict(x_train.reshape(-1,1)))

    # Configure figure axes etc
    pylab.xlabel('Average number of rooms')
    pylab.ylabel('Average price of the house')
    pylab.legend(['Model', 'Test', 'Train', 'Error'])
    pylab.grid()
    pylab.show()

def auto2():
    K.clear_session()

    # Init new feedforward network model from keras
    model = Sequential()

    # In linear regression we don't have hidden layers. Just the output which is connected to input.
    model.add(Dense(1, activation='linear', input_shape=(1,)))

    model.summary()

    model.compile(loss='mean_squared_error',
              optimizer=RMSprop(lr=.1))
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    x_train.shape, y_train.shape, x_test.shape, y_test.shape
    history = model.fit(x_train, y_train,
                    batch_size=32,
                    epochs=100,
                    verbose=0)
    train_error = mean_squared_error(y_train, model.predict(x_train, verbose=0))
    test_error = mean_squared_error(y_test, model.predict(x_test, verbose=0))
    print('Train error=%f test error=%f' % (train_error, test_error))
    pylab.figure()
    xs = np.linspace(3, 10, 20)
    pylab.plot(xs, model.predict(xs.reshape(-1, 1)))
    pylab.plot(x_test, y_test, '.')
    pylab.plot(x_train, y_train, '+')
    pylab.legend(['Model', 'Test', 'Train'])
    pylab.xlabel('Average number of rooms')
    pylab.ylabel('Average price of the house')
    pylab.grid()
    pylab.show()
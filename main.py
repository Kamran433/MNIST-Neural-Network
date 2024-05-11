import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("train.csv")
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)
datadev = data[0:1000].T
datatrain = data[1000:m].T
Xdev = datadev[1:n]

Ydev = datadev[0]
Ytrain = datatrain[0]
Xtrain = datatrain[1:n]



def initparams():
    W1 = np.random.randn(10, 784) * 0.01
    b1 = np.random.randn(10, 1)
    W2 = np.random.randn(10, 10) * 0.01
    b2 = np.random.randn(10, 1)
    return W1, b1, W2, b2


def RelU(Z):
    return np.maximum(0, Z)

def SoftMax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Subtracting max(Z) for numerical stability
    A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    return A


def forward(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = RelU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = SoftMax(Z2)
    return A1, A2, Z1, Z2

def OneHot(Y):
    OneHotY = np.zeros((Y.size, Y.max() + 1))
    OneHotY[np.arange(Y.size), Y] = 1
    OneHotY = OneHotY.T
    return OneHotY

def derReLU(Z):
    return Z > 0

def back(A1, A2, Z1, W2, Y, X):
    m = Y.size
    OHY = OneHot(Y)
    dz2 = A2 - OHY
    dz1 = W2.T.dot(dz2) * derReLU(Z1)
    dw1 = 1 / m * dz1.dot(X.T)
    db1 = 1 / m * np.sum(dz1, axis=1, keepdims=True)
    dw2 = 1 / m * dz2.dot(A1.T)
    db2 = 1 / m * np.sum(dz2, axis=1, keepdims=True)
    return dw1, dw2, db1, db2

def errors(W1, W2, b1, b2 ,dw1, dw2, db1, db2, alpha):
    W1 = W1 - alpha * dw1
    W2 = W2 - alpha * dw2
    b1 = b1 - alpha * db1
    b2 = b2 - alpha * db2
    return W1, W2, b1, b2

def getpred(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def GradientDescend(X, Y, iterations, alpha):
    W1, b1, W2, b2 = initparams()
    for i in range(iterations):
        A1, A2, Z1, Z2 = forward(W1, b1, W2, b2, X)
        dw1, dw2, db1, db2 = back(A1, A2, Z1, W2, Y, X)
        W1, W2, b1, b2 = errors(W1, W2, b1, b2 ,dw1, dw2, db1, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = getpred(A2)
            print(f"Accuracy: {get_accuracy(predictions, Y) * 100} % ", )
    return W1, b1, W2, b2


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward(W1, b1, W2, b2, X)
    predictions = getpred(A2)
    return predictions


def test_prediction(index, W1, b1, W2, b2):
    current_image = Xtrain[:, index, None]
    prediction = make_predictions(Xtrain[:, index, None], W1, b1, W2, b2)
    label = Ytrain[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.imshow(current_image, cmap='gray', interpolation='nearest')
    plt.show()

W1, b1, W2, b2 = GradientDescend(Xdev, Ydev, 500, 0.001)
test_prediction(208, W1, b1, W2, b2)

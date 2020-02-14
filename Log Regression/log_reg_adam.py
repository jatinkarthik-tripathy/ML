import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split

alpha = 0.01
b1 = 0.9
b2 = 0.999
epsilon = 1e-8

m_t = 0
v_t = 0
t = 0


def adam(gradient, theta):
    global v_t, m_t, t
    t += 1
    g_t = gradient
    m_t = b1*m_t + (1-b1)*g_t
    v_t = b2*v_t + (1-b2)*(g_t*g_t)
    m_cap = m_t/(1-(b1**t))
    v_cap = v_t/(1-(b2**t))

    theta = theta - (alpha*m_cap)/(np.sqrt(v_cap) + epsilon)
    return theta

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train(x, y, theta):
    # predicting values
    z = np.dot(x, theta)
    h = sigmoid(z)

    # getting cost for prediction
    cost = (-y*np.log(h)-(1-y)*np.log(1-h)).mean()

    # optimization
    gradient = np.dot(x.T, (h - y)) / y.size
    theta = adam(gradient, theta)
    return theta, cost


def test(x, theta, threshold=0.5):
    return sigmoid(np.dot(x, theta)) > threshold


def run():
    # dataset
    df = pd.read_csv('data.txt', sep='\t')

    df.replace('?', -99999, inplace=True)
    df.drop(df.columns[0], axis=1, inplace=True)
    df.drop(['id'], 1, inplace=True)

    X = np.array(df.drop(['bought'], 1))
    y = np.array(df['bought'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)

    theta = np.zeros(X_train.shape[1])

   # training loop
    for _ in range(10):
        theta, cost = train(X_train, y_train, theta)

    predict = test(X_test, theta)

    print(f'confidence: {(predict==y_test).mean()}')


if __name__ == '__main__':
    run()

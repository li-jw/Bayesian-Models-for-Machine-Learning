import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.linalg import inv
import math

path = '/Users/jingweili/Documents/CU/Courses/EE6720/Homework/HW2/hw2_data_csv'
xtrain = np.array(pd.read_csv(path + '/Xtrain.csv', header = None))
ytrain = np.array(pd.read_csv(path + '/ytrain.csv', header = None))
xtest = np.array(pd.read_csv(path + '/Xtest.csv', header = None))
ytest = np.array(pd.read_csv(path + '/ytest.csv', header = None))

def expect(x, y, w):
    quantile = np.dot(x, w) / sigma
    if y == 1:
        expect = sigma * (quantile + norm.pdf(-quantile) / (1 - norm.cdf(-quantile)))
    else:
        expect = sigma * (quantile - norm.pdf(-quantile) / norm.cdf(-quantile))
    return(expect[0])

def updateW(phi):
    denominator = np.zeros((k, k))
    numerator = np.zeros((k, 1))
    for i in range(ntrain):
        x = np.array(np.mat(xtrain[i,]))
        denominator += np.dot(x.transpose(), x)
        numerator += x.transpose() * phi[i]
    denominator = denominator * sigma**(-2) + np.eye(k)
    numerator = numerator * sigma**(-2)
    w = np.dot(inv(denominator), numerator)
    return(np.array(w))

def object(w):
    quantile = np.dot(xtrain, w) / sigma
    prob = norm.cdf(quantile)
    obj = ((k/2) * math.log(1/(2*math.pi)) - np.dot(w.transpose(), w)[0][0]/2 + 
          np.dot(ytrain.transpose(), np.log(prob)) + 
          np.dot(1 - ytrain.transpose(), np.log(1 - prob)))
    return(obj)

def prob0(x, w):
    quantile = - np.dot(x, w)[0] / sigma
    cdf = norm.cdf(quantile)
    return([cdf, 1 - cdf])

def EM(xtrain, ytrain, xtest, T):
    ww = np.zeros((T, k))
    w = np.zeros((k,1))
    phi = np.array(ntrain)
    objectFunc = np.zeros(T)
    print('Iteration\tObject Function')
    for t in range(T):
        phi = [expect(xtrain[i, ], ytrain[i], w) for i in range(ntrain)]
        w = updateW(phi)
        ww[t, ] = w.transpose()
        objectFunc[t] = object(w)
        if (t + 1) % 10 == 0:
            print('{0}\t\t{1}'.format(t + 1, objectFunc[t]))
    prob = np.apply_along_axis(prob0, 1, xtest, w)
    pred = np.apply_along_axis(lambda x: 0 if x[0] > x[1] else 1, 1, prob)
    return(objectFunc, prob, pred, ww)

# Confustion matrix
from sklearn.metrics import confusion_matrix
sigma = 1.5
k = xtrain.shape[1]
ntrain = xtrain.shape[0]
T = 100
objectFunc, prob, pred, ww = EM(xtrain, ytrain, xtest, T)
confusion_matrix(pred, ytest)

# Plot log joint density
t = [i + 1 for i in range(T)]
plt.plot(t, objectFunc)
plt.xlabel('Iteration')
plt.ylabel('log joint density')
plt.ylim(ymax = -1900)

# Reconstruct misclassified images
mis_pred = [abs(ytest[i][0] - pred[i]) for i in range(len(pred))]
mis_list = [i for i, e in enumerate(mis_pred) if e == 1]
print('No.\tActual\tProb:4\tProb:9')
for i in mis_list[0:3]:
    actual = 4 if ytest[i][0] == 0 else 9
    print('{0}\t{1}\t{2:1.4f}\t{3:1.4f}'.format(i + 1, actual, prob[i, 0], prob[i, 1]))
Q = np.array(pd.read_csv(path + '/Q.csv', header = None))
x = xtest[mis_list[0:3],]
x2 = np.dot(x, Q.T)
for i in range(3):
    image = x2[i,].reshape((28, 28))
    plt.matshow(image, cmap=plt.cm.gray)

# Reconstruct most ambiguous images
ambig = [abs(prob[i, 0] - 0.5) for i in range(len(pred))]
ambig_top3 = list(np.argsort(ambig)[0:3])
print('No.\tActual\tProb:4\tProb:9')
for i in ambig_top3[0:3]:
    actual = 4 if ytest[i][0] == 0 else 9
    print('{0}\t{1}\t{2:1.4f}\t{3:1.4f}'.format(i + 1, actual, prob[i, 0], prob[i, 1]))
ambig_x = xtest[ambig_top3, ]
ambig_x2 = np.dot(x, Q.T)
for i in range(3):
    image = ambig_x2[i,].reshape((28, 28))
    plt.matshow(image, cmap=plt.cm.gray)

# Treat wt as digit and reconstruct images
t = [i - 1 for i in [1, 5, 10, 25, 50, 100]]
ww2 = np.dot(ww, Q.T)
for i in t:
    image = ww2[i,].reshape((28, 28))
    plt.matshow(image, cmap=plt.cm.gray)



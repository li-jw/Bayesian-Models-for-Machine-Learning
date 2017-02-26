
# coding: utf-8

import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib.pyplot as plt

path = '/Users/jingweili/Documents/CU/Courses/EE6720/Homework/HW1/hw1_data_csv'
xtrain = pd.read_csv(path + '/Xtrain.csv', header = None)
ytrain = pd.read_csv(path + '/ytrain.csv', header = None)
xtest = pd.read_csv(path + '/Xtest.csv', header = None)
ytest = pd.read_csv(path + '/ytest.csv', header = None)

def parameters(x):
    n = x.shape[0]
    mean = x.mean()
    var = x.var()
    mu = (n * mean) / (n + 1)
    kappa = n + 1
    alpha = 1 + n / 2
    beta = 1 + var * (n - 1) / 2 + (n * mean**2) / (2 * (n + 1))
    df = 2 * alpha
    scale = ((beta * (1 + kappa)) / (alpha * kappa))**(1/2)
    return(df, mu, scale)

def naiveBayes(xtrain, ytrain, xtest):
    ntrain = xtrain.shape[0]
    ntest = xtest.shape[0]
    log_prob = pd.DataFrame(0, index = np.arange(ntest), columns = [0, 1])
    for y in [0, 1]:
        xtrain_y = xtrain[ytrain[0] == y]
        n_y = xtrain_y.shape[0]
        prob_y = (1 + n_y) / (ntrain + 2)
        log_condi_prob = pd.DataFrame(0, index = np.arange(ntest), columns = range(15))
        for d in range(15):
            [df, loc, scale] = parameters(xtrain_y[d])
            log_condi_prob[d] = np.log(t.pdf(xtest[d], df, loc, scale))
        log_prob[y] = log_condi_prob.sum(axis = 1) + np.log(prob_y)
    prob = log_prob.apply(lambda x: np.exp(x), axis = 1)
    prob = prob.apply(lambda x: pd.Series([x[0]/x.sum(), x[1]/x.sum()]), axis = 1)
    pred = prob.apply(lambda x: 0 if x[0] > x[1] else 1, axis = 1)
    return(prob, pred)

from sklearn.metrics import confusion_matrix
[prob, pred] = naiveBayes(xtrain, ytrain, xtest)
confusion_matrix(pred, ytest[0])

# Misclassified digits
mis_pred = abs(ytest[0] - pred)
mis_list = list(mis_pred.index[mis_pred == 1])

for i in mis_list[0:3]:
    mis_prob = prob[pred[i]][i]
    print('No.{0} digit is misclassified as {1} with probability {2}'.format(i, pred[i], mis_prob))

# Reconstruct images
Q = pd.read_csv(path + '/Q.csv', header = None)
x = np.asarray(xtest)[mis_list[0:3]]
x2 = x.dot(np.asarray(Q).T)

for i in range(3):
    image = x2[i,].reshape((28, 28))
    plt.matshow(image, cmap=plt.cm.gray)

# Reconstruct most ambiguous images
ambig = abs(prob[0] - 0.5).sort_values()
ambig_top3 = list(ambig.index[0:3])
print('The most ambiguous images are {}'.format(ambig_top3))

print('No.\tActual\tProb:4\tProb:9')
for i in ambig_top3[0:3]:
    print('{0}\t{1}\t{2:1.4f}\t{3:1.4f}'.format(i, ytest[0][i], prob[0][i], prob[1][i]))

ambig_x = np.asarray(xtest)[ambig_top3]
ambig_x2 = ambig_x.dot(np.asarray(Q).T)
for i in range(3):
    image = ambig_x2[i,].reshape((28, 28))
    plt.matshow(image, cmap=plt.cm.gray)

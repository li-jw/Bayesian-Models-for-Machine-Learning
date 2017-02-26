import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.special import gammaln
from scipy.special import psi
from numpy import log


path = '/Users/jingweili/Documents/CU/Courses/EE6720/Homework/HW3/data_csv'
x1 = np.array(pd.read_csv(path + '/X_set1.csv', header = None))
x2 = np.array(pd.read_csv(path + '/X_set2.csv', header = None))
x3 = np.array(pd.read_csv(path + '/X_set3.csv', header = None))
y1 = np.array(pd.read_csv(path + '/y_set1.csv', header = None))
y2 = np.array(pd.read_csv(path + '/y_set2.csv', header = None))
y3 = np.array(pd.read_csv(path + '/y_set3.csv', header = None))
z1 = np.array(pd.read_csv(path + '/z_set1.csv', header = None))
z2 = np.array(pd.read_csv(path + '/z_set2.csv', header = None))
z3 = np.array(pd.read_csv(path + '/z_set3.csv', header = None))

T = 500
a0, b0, e0, f0 = 10e-16, 10e-16, 1, 1

def VI(x, y, T):
    # Initialize
    n, d = x.shape
    a, b, e, f = [a0] * d, [b0] * d, e0, f0
    ViObjFunc = [0] * T
    
    sum_yixi = np.zeros((d, 1))
    sum_xixi = np.zeros((d, d))
    for i in range(n):
        xi = x[i].reshape((d, 1))
        sum_yixi += y[i, 0] * xi
        sum_xixi += xi.dot(xi.T)
    
    for t in range(T): 
        # Update q(w)
        sigma_diag = np.diag([(a[i] / b[i]) for i in range(d)])
        sigma = inv(sigma_diag + sum_xixi * (e / f))
        mu = sigma.dot((e / f) * sum_yixi)
        
        # Update q(lambda)
        xsigmax = yminusxTmu = 0
        for i in range(n):
            xi = x[i].reshape((d, 1))
            xsigmax += (xi.T).dot(sigma).dot(xi)
            yminusxTmu += (y[i, 0] - (xi.T).dot(mu))**2
        e = e0 + n / 2
        f = f0 + 0.5 * (yminusxTmu + xsigmax)[0][0]
        
        # Update q(alpha)
        a = [a0 + 0.5] * d
        b = [(b0 + 0.5 * (sigma[k, k] + mu[k, 0]**2)) for k in range(d)]
        
        # Calculate variational object function
        p_y = 0.5 * n * (psi(e) - log(f)) - 0.5 * e / f * (yminusxTmu + xsigmax)
        p_lambda = (e0 - 1) * (psi(e) - log(f)) - f0 * e / f
        p_alpha = sum([(a0 - 1) * (psi(a[k]) - log(b[k])) - b0 * a[k] / b[k] for k in range(d)])
        diag_ab = np.diag([a[k] / b[k] for k in range(d)])
        p_w = 0.5 * (sum([psi(a[k]) - log(b[k]) for k in range(d)]) - 
              np.trace(diag_ab.dot(sigma + mu.dot(mu.T))))
        q_lambda = log(f) - e - gammaln(e) + (e - 1) * psi(e)
        q_alpha = sum([log(b[k]) - a[k] - gammaln(a[k]) + (a[k] - 1) * psi(a[k]) for k in range(d)])
        q_w = - 0.5 * np.linalg.slogdet(sigma)[1]
        obj_func = p_y + p_lambda + p_alpha + p_w - q_lambda - q_alpha - q_w
        ViObjFunc[t] = obj_func[0][0]
        
    return(a, b, e, f, mu, sigma, ViObjFunc)

def plotObjFunc(ObjFunc):
    plt.figure()
    plt.plot(ObjFunc)
    plt.title('Variational Objective Function')
    plt.xlabel('Iterations')
    plt.ylabel('Variational Objective Function')
    plt.grid()
    plt.show()

def plotAlpha(a, b):
    plt.figure()
    plt.stem([b[k] / a[k] for k in range(len(a))])
    plt.title(r'1/E[$\alpha_k$] as a function of k')
    plt.grid()
    plt.xlabel('k')
    plt.ylabel(r'1/E[$\alpha_k$]')
    plt.show()

def plot_yz(x, y, z, mu):    
    yhat = x.dot(mu)
    z_n = np.linspace(-6, 6, len(y))
    fz_n = 10 * np.sinc(z_n)
    plt.plot(z, yhat, color='red', label='Predicted')
    plt.scatter(z, y, color='blue', label='Actual', s = 5)
    plt.plot(z_n, fz_n, label='True')
    plt.xlabel('z')
    plt.ylabel('y')
    plt.legend()
    plt.title('y vs z')
    plt.grid()

def solution(x, y, z, T):
    a, b, e, f, mu, sigma, ViObjFunc = VI(x, y, T)
    plotObjFunc(ViObjFunc)
    plotAlpha(a, b)
    print('The value of 1/E[' + u'\u03BB' '] for the final iteration is ' + str(f / e))
    plot_yz(x, y, z, mu)

solution(x1, y1, z1, T)

solution(x2, y2, z2, T)

solution(x3, y3, z3, T)


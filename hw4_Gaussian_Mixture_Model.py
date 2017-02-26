
import pandas as pd
import numpy as np
import scipy
from scipy import stats
from scipy.special import gamma, psi, gammaln, multigammaln
from numpy.linalg import inv, det
from numpy import log
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# Gaussian Mixture Model

def GMM(X, K, T):
    np.random.seed(1)
    N, d = X.shape
    
    # Initialize parameters
    pi = np.repeat(1 / K, K)
    mu = np.random.rand(K, d)
    sigma = np.repeat([np.eye(d)], K, axis = 0)
    log_likelihood = [0] * T
    phi = np.empty((N, K))
    
    for t in range(T):
        # E-step
        for i in range(N):
            for j in range(K):
                phi[i, j] = pi[j] * stats.multivariate_normal.pdf(X[i, :], mean = mu[j, :], cov = sigma[j, :, :])
        phi = np.apply_along_axis(lambda x: x / x.sum(), 1, phi)
        
        # M-step
        n = np.sum(phi, 0)
        for j in range(K):
            mu[j, :] = np.sum(np.multiply(phi[:, j].reshape(N, 1), X), 0) / n[j]
            sigma[j, :, :] = np.multiply(phi[:, j].reshape(N, 1), (X - mu[j, :])).T.dot(X - mu[j, :]) / n[j]
            pi[j] = n[j] / n.sum()
        
        # Calculate objective function
        for i in range(N):
            p = 0
            for k in range(K):
                p += pi[k] * stats.multivariate_normal.pdf(X[i,:], mean = mu[k,:], cov = sigma[k, :,:])
            log_likelihood[t] += log(p)
    return(phi, log_likelihood)


# Variational Inference

def VI(X, K, T):
    np.random.seed(1)
    N, d = X.shape
    
    # Initialize parameters
    alpha0 = 1
    c0 = 10
    a0 = d
    B0 = np.cov(X.T) * d / 10
    
    alpha = np.random.rand(K) * N
    kmeans = KMeans(n_clusters = K).fit(X)
    m = kmeans.cluster_centers_
    sigma = np.repeat([np.eye(d) / c0], K, axis = 0)
    a = np.random.rand(K) * K**2
    B = np.array([(k + 1) * B0 for k in range(K)])
    L = np.repeat(0., T)
    
    for t in range(T):
        # Update q(c)
        t1 = np.sum(psi((a.reshape(1, K) - np.arange(d).reshape(d, 1)) / 2), 0) - log(det(B))
        t3 = np.array([np.trace(np.dot(a[j] * inv(B[j]), sigma[j])) for j in range(K)])
        t4 = psi(alpha) - psi(alpha.sum())
        t2 = np.zeros((N, K))
        for i in range(N):
            for j in range(K):
                xminusm = (X[i, :] - m[j, :]).reshape(d, 1)
                t2[i, j] = xminusm.T.dot(a[j] * inv(B[j])).dot(xminusm)
        phi = np.exp(0.5 * t1 - 0.5 * t2 - 0.5 * t3 + t4)
        phi = np.apply_along_axis(lambda x: x / x.sum(), 1, phi)
        n = np.sum(phi, 0)
        
        # Update q(pi)
        alpha = alpha0 + n
        
        # Update q(mu)
        sigma = np.array([inv(np.eye(d) / c0 + n[j] * a[j] * inv(B[j])) for j in range(K)])
        m = np.array([sigma[j].dot(a[j] * inv(B[j])).dot(np.sum(np.multiply(phi[:,j].reshape(N,1), X), 0)) for j in range(K)])
        
        # Update q(lambda)
        a = a0 + n
        for j in range(K):
            s = 0
            for i in range(N):
                xminusm = (X[i, :] - m[j, :]).reshape(d, 1)
                s += phi[i, j] * (xminusm.dot(xminusm.T) + sigma[j])
            B[j] = B0 + s
        
        # Calculate objective function
        Eln_lambda = d * log(2) - log(det(B)) + np.sum(psi(0.5 * (a.reshape(K, 1) - np.arange(d))), 1)
        Eln_pi = psi(alpha) - psi(alpha.sum())
        Eln_q_c = np.multiply(phi, log(phi)).sum()
        Eln_q_pi = np.multiply(alpha - 1, Eln_pi).sum() + gammaln(alpha.sum()) - gammaln(alpha).sum()
        Eln_q_mu = - sum([np.linalg.slogdet(sigma[j])[1] for j in range(K)]) / 2
        Eln_q_lambda = - sum([stats.wishart.entropy(a[j], inv(B[j])) for j in range(K)])
        Eln_p_pi = (alpha0 - 1) * Eln_pi.sum()
        Eln_p_mu = - 0.5 / c0 * sum([np.trace(sigma[j]) + m[j].dot(m[j]) for j in range(K)])
        Eln_p_lambda = 0.5 * (a0 - d - 1) * Eln_lambda.sum() - 0.5 * sum([np.trace(B0.dot(a[j] * inv(B[j]))) for j in range(K)])
        E_xmulambda = np.zeros((N,K))
        for i in range(N):
            for j in range(K):
                ximinusmj = (X[i, :] - m[j, :]).reshape(d,1)
                E_xmulambda[i, j] = ximinusmj.T.dot(a[j] * inv(B[j])).dot(ximinusmj) + np.trace(a[j] * inv(B[j]).dot(sigma[j]))
        Eln_p_xc = np.multiply(phi, 0.5 * Eln_lambda + Eln_pi - 0.5 * E_xmulambda).sum()
        L[t] = Eln_p_xc + Eln_p_pi + Eln_p_mu + Eln_p_lambda - Eln_q_c - Eln_q_pi - Eln_q_mu - Eln_q_lambda
    return(phi, L)


# Gibbs sampling

def Gibbs(X, T):
    np.random.seed(1)
    N, d = X.shape
    
    # Initialize parameters
    m0 = np.mean(X, 0)
    c0 = 0.1
    a0 = d
    A0 = np.cov(X.T)
    B0 = c0 * d * A0
    alpha0 = 1
    K = 20
    
    X_c = np.zeros(N).astype(int)
    mu = np.zeros((K, d))
    lamb = np.zeros((K, d, d))
    mu[0, :], lamb[0, :, :] = update_param(X, m0, c0, a0, B0, X_c, 0)
    top_clusters = np.zeros((T, 6))
    num_clusters = np.zeros(T)
    
    for t in range(T):
        phi = np.zeros((N, K))
        
        for i in range(N):
            all_clusters = np.array([len(np.where(np.delete(X_c, i) == k)[0]) for k in range(K)])
            clusters = np.where(all_clusters > 0)[0]
            for j in range(len(clusters)):
                if j != clusters[j]:
                    X_c[np.where(X_c == clusters[j])] = j
                    mu[j], mu[clusters[j]] = mu[clusters[j]], mu[j]
                    lamb[j], lamb[clusters[j]] = lamb[clusters[j]], lamb[j]
                phi[i, j] = stats.multivariate_normal.pdf(X[i, :], mean = mu[j], cov = inv(lamb[j])) * all_clusters[clusters[j]] / (alpha0 + N - 1)

            j_max = int(max(X_c) + 1)
            xminusm = (X[i, :] - m0).reshape(d, 1)
            marginal = ((c0 / ((c0 + 1) * np.pi))**(d / 2)
                        * det(B0 + c0 / (1 + c0) * xminusm.dot(xminusm.T))**(- 0.5 * (a0 + 1))
                        / det(B0)**(- 0.5 * a0)
                        * np.exp(multigammaln((a0 + 1) / 2, d)) / np.exp(multigammaln(a0 / 2, d)))
            phi[i, j_max] = alpha0 / (alpha0 + N - 1) * marginal
            phi[i] = phi[i] / np.sum(phi[i])
            cluster_list = np.where(phi[i] > 0)[0]
            discrete_dist = stats.rv_discrete(values = (range(len(cluster_list)), phi[i][cluster_list]))
            X_c[i] = discrete_dist.rvs(size = 1)[0]
            if X_c[i] == j_max:
                mu[j_max], lamb[j_max] = update_param(X, m0, c0, a0, B0, X_c, j_max)
        for j in np.unique(X_c):
            mu[j], lamb[j] = update_param(X, m0, c0, a0, B0, X_c, j)
        top_clusters[t, :] = np.array(sorted([len(np.where(X_c == j)[0]) for j in range(K)], reverse = True))[0:6]
        num_clusters[t] = len(np.unique(X_c))
    return top_clusters, num_clusters

def update_param(X, m0, c0, a0, B0, X_c, j):
    X_s = X[np.where(X_c == j)]
    s, d = X_s.shape
    m = c0 / (s + c0) * m0 + 1 / (s + c0) * np.sum(X_s, 0)
    c = s + c0
    a = a0 + s
    X_mean = np.mean(X_s, 0)
    X_centered = X_s - X_mean
    X_mean_minus_m = (X_mean - m0).reshape(d,1)
    B = B0 + X_centered.T.dot(X_centered) + s / (a0 * s + 1) * X_mean_minus_m.dot(X_mean_minus_m.T)
    lamb = stats.wishart.rvs(a, inv(B))
    mu = stats.multivariate_normal.rvs(mean = m, cov = inv(c * lamb))
    return mu, lamb


def plot_obj_func(obj_func, K, ylabel):
    plt.figure(figsize = (8, 6))
    plt.title('K = %s' % K, fontsize = 20)
    plt.xlabel('Iterations', fontsize = 20)
    plt.ylabel(ylabel, fontsize = 20)
    plt.plot(obj_func)


def scatter_plot(X, phi, K):
    pred = np.argmax(phi, 1)
    color_list = list("bgrcmyk") + ['darkorange', 'lime', 'dodgerblue', 'silver', 'fuchsia', 'pink', 'yellow']
    cluster_list = np.unique(pred)
    plt.figure(figsize = (8, 6))
    for k in range(len(cluster_list)):
        plt.scatter([x[0] for x in X[np.where(pred == cluster_list[k])]], [x[1] for x in X[np.where(pred == cluster_list[k])]], color = color_list[k])
    plt.legend(cluster_list, fontsize = 12, loc=4)
    plt.title('K = %s' % K, fontsize = 20)
    plt.xlabel('x1', fontsize = 20)
    plt.ylabel('x2', fontsize = 20)


def plot_top_clusters(top_clusters):
    plt.figure(figsize = (8, 6))
    plt.plot(top_clusters)
    plt.xlabel('Iteration', fontsize = 20)
    plt.ylabel('Number of points', fontsize = 20)
    plt.legend(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6'], fontsize = 12)


def plot_num_clusters(num_clusters):
    plt.figure(figsize = (8, 6))
    plt.plot(np.arange(1, len(num_clusters) + 1), num_clusters)
    plt.xlabel('Iteration', fontsize = 20)
    plt.ylabel('Number of clusters', fontsize = 20)
    plt.ylim(0, max(num_clusters) + 1)


def main():
    X = np.array(pd.read_csv('./data.txt', sep = ',', header = None))
    
    # Problem 1
    K1 = [2, 4, 8, 10]
    for k in K1:
        phi, log_likelihood = GMM(X, k, 100)
        plot_obj_func(log_likelihood, k, 'Log Likelihood')
        scatter_plot(X, phi, k)
    
    # Problem 2
    K2 = [2, 4, 10, 25]
    for k in K2:
        phi, L = VI(X, k, 100)
        plot_obj_func(L, k, 'L')
        scatter_plot(X, phi, k)
    
    # Problem 3
    top_clusters, num_clusters = Gibbs(X, 500)
    plot_top_clusters(top_clusters)
    plot_num_clusters(num_clusters)


if __name__ == '__main__':
    main()





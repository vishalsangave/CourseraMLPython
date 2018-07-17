from __future__ import division
import numpy as np

from computeCost import computeCost


def gradientDescent(X, y, theta, alpha, num_iters):
    m=len(y)
    finalTheta = theta
    J_history = np.zeros((num_iters, 1))
    for iter in range(num_iters):
        delta = ((X * (np.kron(np.ones((1,X.shape[1])),np.matmul(X,theta) - np.c_[y]))).sum(axis=0))/m
        theta = (theta.T - (alpha * delta)).T
        print("theta : {0}".format(theta))
        finalTheta=theta
        J_history[iter] = computeCost(X, y, theta)

    return finalTheta,J_history

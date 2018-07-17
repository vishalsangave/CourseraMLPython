from __future__ import division
import numpy as np
def computeCost(X,y,theta):
    # COMPUTECOST Compute cost for linear regression
    # J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    # parameter for linear regression to fit the data points in X and y
    m= len(y)
    # You need to return the following variables correctly

    return (1/(2*m))*( np.sum(np.square(np.subtract(np.matmul(X,theta), np.c_[y]))))

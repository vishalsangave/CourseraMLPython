import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gradientDescent import gradientDescent
from computeCost import computeCost
from plotData import plotData

data = pd.read_csv('ex1data1.txt', sep=',')
X = pd.DataFrame(data).ix[:,0]
y= pd.DataFrame(data).ix[:,1]

m = len(y)

plotData(X,y)

# Lets create matrix from arrays
# X = np.column_stack((np.ones((m,1)),np.asarray(X)))
X = np.c_[np.ones((m,1)),X]
theta = np.zeros((2,1))

iterations = 1500
alpha = 0.01

J = computeCost(X, y, theta)
print('With theta = [0 ; 0]\nCost computed = {0}\n'.format(J))
print('Expected cost value (approx) 32.07\n')

J = computeCost(X, y, np.matrix('-1;2'))
print('With theta = [-1 ; 2]\nCost computed = {0}\n'.format(J))
print('Expected cost value (approx) 54.24\n')

print('\nRunning Gradient Descent ...\n')
# run gradient descent
theta, JHistory = gradientDescent(X, y, theta, alpha, iterations)
print('Theta found by gradient descent:\n')
print('{0}\n'.format(theta) )
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')


plt.plot(X[:, 1], np.matmul(X,theta), 'r--', linewidth=1)
plt.show()

 # Predict values for population sizes of 35,000 and 70,000
predict1 = np.matrix('1, 3.5') *theta
print('For population = 35,000, we predict a profit of {0}\n'.format(predict1*10000))
predict2 = np.matrix('1, 7') * theta
print('For population = 70,000, we predict a profit of {0}\n'.format((
    predict2*10000)[0]))

input('Program paused. Press enter to continue.\n')
  
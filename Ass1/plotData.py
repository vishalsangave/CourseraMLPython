import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
def plotData(X,y):
    plt.scatter(X, y, marker="x", c=y)
    plt.xlabel('Profit in $10,000s')
    plt.ylabel('Population of City in 10,000s')
    # add a 'best fit' line
    # l = plt.plot(X, y, 'r--', linewidth=1)

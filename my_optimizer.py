import numpy as np
from numpy import arange
from numpy import meshgrid
from matplotlib import pyplot
import matplotlib.pyplot as plt

def objective(x, y):
    return x**2.0 + y**2.0

def objective_gradient(x, y):
    return x**2.0 + y**2.0

def show_3D():
    # define range for input
    r_min, r_max = -1.0, 1.0
    # sample input range uniformly at 0.1 increments
    xaxis = arange(r_min, r_max, 0.1)
    yaxis = arange(r_min, r_max, 0.1)
    # create a mesh from the axis
    x, y = meshgrid(xaxis, yaxis)
    # compute targets
    results = objective(x, y)
    # create a surface plot with the jet color scheme
    figure = pyplot.figure()
    axis = figure.gca(projection='3d')
    axis.plot_surface(x, y, results, cmap='jet')
    # show the plot
    pyplot.show()


def show_2D():
    # define range for input
    bounds = np.asarray([[-1.0, 1.0], [-1.0, 1.0]])
    # sample input range uniformly at 0.1 increments
    xaxis = arange(bounds[0, 0], bounds[0, 1], 0.1)
    yaxis = arange(bounds[1, 0], bounds[1, 1], 0.1)
    # create a mesh from the axis
    x, y = meshgrid(xaxis, yaxis)
    # compute targets
    results = objective(x, y)
    # create a filled contour plot with 50 levels and jet color scheme
    pyplot.contourf(x, y, results, levels=50, cmap='jet')
    # show the plot
    pyplot.show()

def test_func(x):
    return x**2

def test_gradient(x):
    return 2*x


def gradient_descent(gradient, start, learning_rate, n_iterations, tolerance=1e-02):
    x_axis = arange(-100, 101, 1)
    y_axis = test_func(x_axis)

    # plot base function in blue
    plt.plot(x_axis, y_axis, c="blue", linestyle=':')


    vector = start
    changed_x = []
    changed_y = []
    for i in range(n_iterations):
        diff = -learning_rate*gradient(vector)

        if np.all(np.abs(diff) <= tolerance):
            break
        print("Iteration {} | Past: {}, New: {}".format(i, vector, vector+diff))
        vector += diff
        print("(x,y: ({},{})".format(vector, test_func(vector)))
        changed_x.append(vector)
        changed_y.append(test_func(vector))

    print(x_axis)
    print(test_func(x_axis))
    # plot algo progress in red
    plt.scatter(changed_x, changed_y, 10, c='red')
    return vector


if __name__ == '__main__':
    print(gradient_descent(lambda v: 2*v, 100, learning_rate=0.05, n_iterations=100))
    #plt.show()

    plt.show()
# import numpy as np
# from scipy.optimize import leastsq
# import pylab as plt
# import scipy
# import matplotlib as plt
#
# N = 5
# t = np.arange(0, N)
# data = [0, 4.034, 7.982, 11.785, 15.609]
#
# guess_mean = np.mean(data)
# guess_std = 3*np.std(data)/(2**0.5)/(2**0.5)
# guess_phase = 0
# guess_freq = 1
# guess_amp = 1
#
# # we'll use this to plot our first estimate. This might already be good enough for you
# data_first_guess = guess_std*np.sin(t+guess_phase) + guess_mean
#
# # Define the function to optimize, in this case, we want to minimize the difference
# # between the actual data and our "guessed" parameters
# optimize_func = lambda x: x[0]*np.sin(x[1]*t+x[2]) + x[3] - data
# est_amp, est_freq, est_phase, est_mean = leastsq(optimize_func, [guess_amp, guess_freq, guess_phase, guess_mean])[0]
#
# # recreate the fitted curve using the optimized parameters
# data_fit = est_amp*np.sin(est_freq*t+est_phase) + est_mean
#
# # recreate the fitted curve using the optimized parameters
#
# fine_t = np.arange(0,max(t),0.1)
# data_fit=est_amp*np.sin(est_freq*fine_t+est_phase)+est_mean
#
# plt.plot(t, data, '.')
# plt.plot(t, data_first_guess, label='first guess')
# plt.plot(fine_t, data_fit, label='after fitting')
# plt.legend()
# plt.show()
#
# # x = np.array([0, 1, 2, 3, 4])
# # y = np.array([0, 4.034, 7.982, 11.785, 15.609])
# # ddd = np.polyfit(np.log(x), y, 1)
# # print(ddd)
#
# x = np.array([1, 2, 3, 4, 5])
# y = np.array([0, 4.034, 7.982, 11.785, 15.609])
# fsdf= np.polyfit(np.log(x), y, 1)
# print(fsdf)
#
# asd = scipy.optimize.curve_fit(lambda t, a, b: a+b*np.log(t),  x,  y)
# print(asd)

import numpy, scipy, matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress

# ignore any "invalid value in log" warnings internal to curve_fit() routine
import warnings


def func(x, a, b, c):  # x-shifted log
    return a * numpy.log(x + b) + c


##########################################################
# graphics output section
def ModelAndScatterPlot(xData, yData, fittedParameters, ref_data):
    f = plt.figure(figsize=(800 / 100.0, 600 / 100.0), dpi=100)
    axes = f.add_subplot(111)

    # first the raw data as a scatter plot
    axes.plot(xData, yData, 'D')

    # create data for the fitted equation plot
    xModel = numpy.linspace(min(xData), max(xData))
    yModel = func(xModel, *fittedParameters)

    # now the model as a line plot
    axes.plot(xModel, yModel)

    axes.set_xlabel('X Data')  # X axis data label
    axes.set_ylabel('Y Data')  # Y axis data label

    slope2, inter2, r_v2, p_v2, std_err2 = linregress(numpy.arange(0, 5), [0, 4.16, 8.36, 12.51, 16.7])

    line2 = slope2 * numpy.arange(0, 5) + inter2

    # print("sum_mean_list LR = %fx + %f" % (slope1, inter1))
    print("ref_mean_list LR = %fx + %f \r\n R-squared = %f" % (slope2, inter2, r_v2 ** 2))

    print("sum_mean_list Nonlinear regression = %fln(x + %f) %f"
          % (fittedParameters[0], fittedParameters[1], fittedParameters[2]))

    plt.plot(numpy.arange(1, 6), line2, '-y')

    plt.show()
    plt.close('all')  # clean up after using pyplot


def compare_rpm_LR_to_cell_data_NLR():
    warnings.filterwarnings("ignore")

    X = [1, 2, 3, 4, 5]
    Y = [0, 4.034, 7.982, 11.785, 15.609]

    # alias data to match previous example
    x_data = numpy.array(X, dtype=float)
    y_data = numpy.array(Y, dtype=float)

    # these are the same as the scipy defaults
    initialParameters = numpy.array([1.0, 1.0, 1.0])

    # curve fit the test data
    fittedParameters, pcov = curve_fit(func, x_data, y_data, initialParameters)

    modelPredictions = func(x_data, *fittedParameters)

    absError = modelPredictions - y_data

    SE = numpy.square(absError)  # squared errors
    MSE = numpy.mean(SE)  # mean squared errors
    RMSE = numpy.sqrt(MSE)  # Root Mean Squared Error, RMSE
    Rsquared = 1.0 - (numpy.var(absError) / numpy.var(y_data))

    print('Parameters:', fittedParameters)
    print('RMSE:', RMSE)
    print('R-squared:', Rsquared)

    print()

    ModelAndScatterPlot(x_data, y_data, fittedParameters)

compare_rpm_LR_to_cell_data_NLR()

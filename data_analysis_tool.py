import scipy
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import one_cell_imu_data_extraction
from scipy import spatial
from scipy import stats
from scipy.signal import butter, lfilter
import cross_corrlation_sync_point_finder
from scipy.optimize import curve_fit
from scipy.stats import linregress
import warnings

fs = 100
section_frame_cut = [0, 2125, 4140, 6110, 8100, 10070]


# accel / gyro
# 1. cosine similarity
# 2. pearson correlation

def butter_bandpss(lowcut, hicut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = hicut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpss(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def calculate_cosine_similarity(ref, sample_x, sample_y, sample_z, section_number):
    plt.plot(np.arange(0, section_frame_cut[section_number + 1] - section_frame_cut[section_number], 1),
             ref[section_frame_cut[section_number]:section_frame_cut[section_number + 1]], "-y")

    plt.plot(np.arange(0, section_frame_cut[section_number + 1] - section_frame_cut[section_number], 1),
             sample_x[section_frame_cut[section_number]:section_frame_cut[section_number + 1]], "-r")
    plt.plot(np.arange(0, section_frame_cut[section_number + 1] - section_frame_cut[section_number], 1),
             sample_y[section_frame_cut[section_number]:section_frame_cut[section_number + 1]], "-g")
    plt.plot(np.arange(0, section_frame_cut[section_number + 1] - section_frame_cut[section_number], 1),
             sample_z[section_frame_cut[section_number]:section_frame_cut[section_number + 1]], "-b")

    plt.show()

    result_x = 1 - spatial.distance.cosine(ref[section_frame_cut[section_number]:section_frame_cut[section_number + 1]],
                                           sample_x[
                                           section_frame_cut[section_number]:section_frame_cut[section_number + 1]])
    result_y = 1 - spatial.distance.cosine(ref[section_frame_cut[section_number]:section_frame_cut[section_number + 1]],
                                           sample_y[
                                           section_frame_cut[section_number]:section_frame_cut[section_number + 1]])
    result_z = 1 - spatial.distance.cosine(ref[section_frame_cut[section_number]:section_frame_cut[section_number + 1]],
                                           sample_z[
                                           section_frame_cut[section_number]:section_frame_cut[section_number + 1]])

    return result_x, result_y, result_z


def calculate_pearson_correlation(ref, sample_x, sample_y, sample_z, section_number):
    r_x, p_x = scipy.stats.pearsonr(ref[section_frame_cut[section_number]:section_frame_cut[section_number + 1]],
                                    sample_x[
                                    section_frame_cut[section_number]:section_frame_cut[section_number + 1]])
    r_y, p_y = scipy.stats.pearsonr(ref[section_frame_cut[section_number]:section_frame_cut[section_number + 1]],
                                    sample_y[
                                    section_frame_cut[section_number]:section_frame_cut[section_number + 1]])
    r_z, p_z = scipy.stats.pearsonr(ref[section_frame_cut[section_number]:section_frame_cut[section_number + 1]],
                                    sample_z[
                                    section_frame_cut[section_number]:section_frame_cut[section_number + 1]])

    return r_x, r_y, r_z


# Magnetometer
# 1. magnetic frequency depend on Motor RPM
def stft(sample):
    N = len(sample)

    reconstruct_x = []
    reconstruct_y = []

    f, t, Zxx = signal.stft(sample, fs=100)
    plt.subplot(311)
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=200, shading='gouraud')
    # print("@@@@@@@", t, len(t), f, len(f), len(Zxx))
    for z in range(0, 129):
        for j in range(0, 72):
            if np.abs(Zxx[z][j]) > 50:
                # print("%%%%%%%%", np.abs(Zxx[z][j]), z, j)
                reconstruct_x.append(j * 1.2622)
                reconstruct_y.append(z)
    # print(reconstruct_x, reconstruct_y)
    plt.plot(reconstruct_x, reconstruct_y)

    z = np.polyfit(reconstruct_x, reconstruct_y, 1)
    p = np.poly1d(z)
    plt.plot(reconstruct_x, p(reconstruct_x), "r-")

    movavg = cross_corrlation_sync_point_finder.moving_average_filter(reconstruct_x)

    plt.plot(movavg, reconstruct_y)

    plt.title('STFT Magnitude')
    plt.ylim(0, 50)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    # plt.plot(f, np.abs(Zxx))
    # plt.show()


def correct_bias(x, y, z, shifting_frame):
    # print("Correction bias")
    x_stationary_mean = np.mean(x[200 + shifting_frame:1800 + shifting_frame])
    y_stationary_mean = np.mean(y[200 + shifting_frame:1800 + shifting_frame])
    z_stationary_mean = np.mean(z[200 + shifting_frame:1800 + shifting_frame])

    print("XYZ stationary section mean", x_stationary_mean, y_stationary_mean, z_stationary_mean)

    correction_x = x - x_stationary_mean
    correction_y = y - y_stationary_mean
    correction_z = z - z_stationary_mean

    return correction_x, correction_y, correction_z


def mean_each_sector(name, sum_xyz, refer_data, saturation_sector):
    print(name)
    sum_mean_list = []
    ref_mean_list = []
    print(saturation_sector)

    sum_mean_list.append(0)
    ref_mean_list.append(0)
    for i in range(0, len(saturation_sector)):
        if i % 2 == 0:
            print(saturation_sector[i], saturation_sector[i + 1])
            sum_mean_list.append(np.mean(sum_xyz[saturation_sector[i]:saturation_sector[i + 1]]))
            ref_mean_list.append(np.mean(refer_data[saturation_sector[i]:saturation_sector[i + 1]]))

    print("Refer mean = ", ref_mean_list)
    print("Sum XYZ mean = ", sum_mean_list)

    return sum_mean_list, ref_mean_list


def func(x, a, b, c):  # x-shifted log
    return a * np.log(x + b) + c


##########################################################
# graphics output section
def ModelAndScatterPlot(xData, yData, fittedParameters, ref_data, exp_rsquared):
    common_x_data = np.arange(0, 5)

    f = plt.figure(figsize=(800 / 100.0, 600 / 100.0), dpi=100)
    axes = f.add_subplot(111)

    # first the raw data as a scatter plot
    axes.plot(xData, yData, 'D')

    # create data for the fitted equation plot
    xModel = np.linspace(min(xData), max(xData))
    yModel = func(xModel, *fittedParameters)

    # now the model as a line plot

    axes.set_xlabel('X Data')  # X axis data label
    axes.set_ylabel('Y Data')  # Y axis data label

    slope2, inter2, r_v2, p_v2, std_err2 = linregress(common_x_data, ref_data)

    line2 = slope2 * common_x_data + inter2

    axes.plot(xModel, yModel, '-r', label="y=%fln(x + %f) %f \n R-squared = %f"
                                          % (fittedParameters[0], fittedParameters[1], fittedParameters[2], exp_rsquared))
    plt.plot(common_x_data, line2, '-b', label="y=%fx + %f \n R-squared = %f" % (slope2, inter2, r_v2 ** 2))

    plt.legend(fontsize=9)
    plt.show()
    plt.close('all')  # clean up after using pyplot


def compare_rpm_LR_to_cell_data_NLR(ref_data_list, sum_data_list):
    warnings.filterwarnings("ignore")

    ref_data = [0, 4.16, 8.36, 12.51, 16.7]

    X = np.arange(0, 5)
    Y = sum_data_list

    # alias data to match previous example
    x_data = np.array(X, dtype=float)
    y_data = np.array(Y, dtype=float)

    # these are the same as the scipy defaults
    initialParameters = np.array([1.0, 1.0, 1.0])

    # curve fit the test data
    fittedParameters, pcov = curve_fit(func, x_data, y_data, initialParameters, maxfev=100000)

    modelPredictions = func(x_data, *fittedParameters)

    absError = modelPredictions - y_data

    SE = np.square(absError)  # squared errors
    MSE = np.mean(SE)  # mean squared errors
    RMSE = np.sqrt(MSE)  # Root Mean Squared Error, RMSE
    Rsquared = 1.0 - (np.var(absError) / np.var(y_data))

    print('Parameters:', fittedParameters)
    print('RMSE:', RMSE)
    print('R-squared:', Rsquared)

    print()

    ModelAndScatterPlot(x_data, y_data, fittedParameters, ref_data_list, Rsquared)

# da.compare_rpm_LR_to_cell_data_NLR([0, 4.16, 8.36, 12.51, 16.7], sum_mean_list)


#
# imuaccel_list, imugyro_list, imumag_list = one_cell_imu_data_extraction.loadv2("./cell data/cell_imu_data_test_14_0.im",
#                                                                                True)
#
# y_f = butter_bandpass_filter(imumag_list[:, 0], 1, 3, 100, 3)
#
# y_s1 = butter_bandpass_filter(imumag_list[39000:42000, 0], 1, 2.5, 100, 1)
# y_s2 = butter_bandpass_filter(imumag_list[42000:44300, 0], 2.5, 4.5, 100, 1)
# y_s3 = butter_bandpass_filter(imumag_list[44300:46000, 0], 4.5, 5, 100, 1)
# y_s4 = butter_bandpass_filter(imumag_list[46000:48000, 0], 5, 5.3, 100, 1)
#
# total_y = list(y_s1) + list(y_s2) + list(y_s3) + list(y_s4)
#
# plt.subplot(312)
# frame_length = np.arange(0, len(total_y), 1)
# plt.plot(frame_length, total_y, label="x")
# stft(total_y)
#
# # y_f1 = butter_bandpass_filter(imumag_list[40000:42000, 0], 1, 3, 100, 3)
# # plt.subplot(312)
# # frame_length = np.arange(0, len(imumag_list[40000:42000, 0]), 1)
# # plt.plot(frame_length, y_f1, label="x")
# # stft(y_f1)
#
# open_g_data = []
# # print("Drawing G graph and x/y/z cell acceleration graph ")
# with open('rpm_data.txt', 'rb') as fp:
#     barr = fp.readlines()
#     # print('total length', len(barr))
#
# for line in barr:
#     open_g_data.append(float(line[:-2].decode('utf-8')))
# length = len(barr)
# time = np.arange(0, length, 1)
#
# plt.subplot(313)
# plt.plot(time, open_g_data, label="RPM")
# plt.show()
import numpy as np
import serial
from scipy import signal
import math

hz_setup = 100
RADIUS = 10
file_path = "./cell data"


def clean_serial_data(data):
    """
    Given a list of serial lines (data). Removes all characters.
    Returns the cleaned list of lists of digits.
    Given something like: ['0.5000,33\r\n', '1.0000,283\r\n']
    Returns: [[0.5,33.0], [1.0,283.0]]
    """
    clean_data = []

    for line in data:
        try:
            line_data = line.decode("UTF-8")
        except:
            pass
        if len(line_data) >= 2:
            clean_data.append(line_data)

    return clean_data


def save_to_txt(data, filename, execution_time):
    """
    Saves a list of lists (data) to filename
    """
    redefine_data_list = []
    cut_float_resample_data = []
    for line in data:
        redefine_data_list.append(line[:-2])

    g_list = calculate_rpm_to_g(redefine_data_list)

    resampled_data = resampling_g_data(g_list, execution_time)
    resampled_data = np.round_(resampled_data, 4)
    # print(resampled_data)

    for line in resampled_data:
        cut_float_resample_data.append(line)

    with open(file_path + '/' + filename, 'w') as fp:
        for buf in cut_float_resample_data:
            fp.write(str(buf) + '\n')


def calculate_rpm_to_g(rpm_data):
    g_list = []
    # print("@@@@@@", rpm_data)
    for rpm in rpm_data:
        rpm = float(rpm)
        g = 1.11786 * math.pow(10, -5) * math.pow(rpm, 2) * RADIUS
        g_list.append(round(g, 4))

    return g_list


def resampling_g_data(data, sampling_time):
    """ Resampling 158.83Hz to about 100Hz """
    # print("resampled number = ", int(sampling_time * hz_setup))
    resampled = signal.resample(data, int(sampling_time * hz_setup))
    # print("Resample data length = ", len(resampled))

    return resampled


def load_rpm_to_g_data():
    g_data = []
    # print("Drawing G graph and x/y/z cell acceleration graph ")
    with open(file_path + '/rpm_to_g_data.txt', 'rb') as fp:
        barr = fp.readlines()
        # print('total length', len(barr))

    for line in barr:
        g_data.append(float(line[:-2].decode('utf-8')))
    length = len(barr)
    frame_length = np.arange(0, length, 1)

    return g_data, frame_length


def load_rpm_to_dps_data():
    dps_data = []
    with open('g_to_rpm.txt', 'rb') as fp:
        barr = fp.readlines()
        # print('total length', len(barr))

    for line in barr:
        dps_data.append((float(line[:-2].decode('utf-8'))) * 6)

    length = len(barr)
    frame_length = np.arange(0, length, 1)

    print(dps_data)

    return dps_data, frame_length


def check_absolute_val_and_change_value(sample_x, sample_y, sample_z):
    print("check1 ", sample_x[9000], sample_y[9000], np.mean(sample_z))

    if np.mean(sample_x) < 0:
        print("^^^^X", np.mean(sample_x))
        sample_x = -sample_x
    if np.mean(sample_y) < 0:
        print("^^^^Y", np.mean(sample_y))
        sample_y = -sample_y
    if np.mean(sample_z) < 0:
        print("^^^^Z", np.mean(sample_z))
        sample_z = -sample_z

    return sample_x, sample_y, sample_z


def moving_average_filter(before_filter_data):
    N = 100  # number of points to test on each side of point of interest, best if even
    padded_x = np.insert(
        np.insert(np.insert(before_filter_data, len(before_filter_data), np.empty(int(N / 2)) * np.nan), 0,
                  np.empty(int(N / 2)) * np.nan), 0, 0)
    # print(padded_x)
    n_nan = np.cumsum(np.isnan(padded_x))
    # print(n_nan)
    cumsum = np.nancumsum(padded_x)
    # print(cumsum)
    window_sum = cumsum[N + 1:] - cumsum[:-(
            N + 1)] - before_filter_data  # subtract value of interest from sum of all values within window
    # print(window_sum)
    window_n_nan = n_nan[N + 1:] - n_nan[:-(N + 1)] - np.isnan(before_filter_data)
    # print(window_n_nan)
    window_n_values = (N - window_n_nan)
    # print(window_n_values)
    movavg = (window_sum) / (window_n_values)
    # print(movavg)
    # print(len(movavg))

    time = np.arange(0, len(movavg), 1)

    return movavg


def time_shifting(ref_g_data, sample_x, sample_y, sample_z):
    # print("Start time sync process")
    check_x = sample_x > 10
    check_y = sample_y > 10
    check_z = sample_z > 10

    cnt_x = np.count_nonzero(check_x == True)
    cnt_y = np.count_nonzero(check_y == True)
    cnt_z = np.count_nonzero(check_z == True)

    if cnt_x > 1000:
        # print("X ref")
        x_frame = np.argmax(np.convolve(ref_g_data[::-1], sample_x, mode='valid'))
        return x_frame

    if cnt_y > 1000:
        # print("Y ref")
        y_frame = np.argmax(np.convolve(ref_g_data[::-1], sample_y, mode='valid'))
        return y_frame

    if cnt_z > 1000:
        # print("Z ref")
        z_frame = np.argmax(np.convolve(ref_g_data[::-1], sample_z, mode='valid'))
        return z_frame


def time_shifting_mag(ref_g_data, sample):
    print("Start time sync process")
    check_x = []
    print(sample)

    for i in range(0, len(sample)):
        if sample[i] > 0:
            print("here")
            check_x.append(True)
    # check_x = sample > 0

    print(check_x)

    cnt_x = np.count_nonzero(check_x == True)

    # if cnt_x > 1:
    print("X ref")
    x_frame = np.argmax(np.convolve(ref_g_data[::-1], sample, mode='valid'))
    return x_frame

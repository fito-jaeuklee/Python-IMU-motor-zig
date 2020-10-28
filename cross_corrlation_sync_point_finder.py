import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import one_cell_imu_data_extraction

def moving_average_filter(before_filter_data):
    N = 100  # number of points to test on each side of point of interest, best if even
    padded_x = np.insert(np.insert(np.insert(before_filter_data, len(before_filter_data), np.empty(int(N / 2)) * np.nan), 0,
                                   np.empty(int(N / 2)) * np.nan), 0, 0)
    print(padded_x)
    n_nan = np.cumsum(np.isnan(padded_x))
    print(n_nan)
    cumsum = np.nancumsum(padded_x)
    print(cumsum)
    window_sum = cumsum[N + 1:] - cumsum[:-(
                N + 1)] - before_filter_data  # subtract value of interest from sum of all values within window
    print(window_sum)
    window_n_nan = n_nan[N + 1:] - n_nan[:-(N + 1)] - np.isnan(before_filter_data)
    print(window_n_nan)
    window_n_values = (N - window_n_nan)
    print(window_n_values)
    movavg = (window_sum) / (window_n_values)
    print(movavg)
    print(len(movavg))

    time = np.arange(0, len(movavg), 1)

    return movavg

def drawing_xyz_accel(g, ax, ay, az, time, accel_len):
    frame = plt.gca()
    # N = 55
    # re = np.convolve(g, np.ones((N,))/N, mode='valid')
    # print(re)
    # plot.plot(time[:len(time) - N + 1], re, "-y", label="G-RPM")
    plt.plot(time, g, "-y", label="G-RPM")

    plt.plot(accel_len, ax, "-r", label="X")
    plt.plot(accel_len, ay, "-g", label="Y")
    plt.plot(accel_len, az, "-b", label="Z")

    plt.title('Gravitational acceleration from RPM')

    # Give x axis label for the sine wave plot
    plt.xlabel('frame')

    # Give y axis label for the sine wave plot
    plt.ylabel('g(9.8m/s2)')
    plt.ylim(-20, 20)
    plt.grid(True)
    # frame.axes.get_yaxis().set_visible(False)

    # plot.axhline(y=0, color='k')
    plt.axhline(y=0, color='k')
    plt.legend(frameon=False)
    plt.get_current_fig_manager().show()
    plt.show()


def lag_finder(y1, y2, sr):
    n = len(y1)

    corr = signal.correlate(y2, y1, mode='same') / np.sqrt(signal.correlate(y1, y1, mode='same')[int(n/2)] * signal.correlate(y2, y2, mode='same')[int(n/2)])

    delay_arr = np.linspace(-0.5*n/sr, 0.5*n/sr, n)
    delay = delay_arr[np.argmax(corr)]
    print('y2 is ' + str(delay) + ' behind y1')

    plt.figure()
    plt.plot(delay_arr, corr)
    plt.title('Lag: ' + str(np.round(delay, 3)) + ' s')
    plt.xlabel('Lag')
    plt.ylabel('Correlation coeff')
    plt.show()


open_g_data = []
print("Drawing G graph and x/y/z cell acceleration graph ")
with open('C:/Users/jaeuk/Downloads/rpm_data (1).txt', 'rb') as fp:
    barr = fp.readlines()
    print('total length', len(barr))

for line in barr:
    open_g_data.append(float(line[:-2].decode('utf-8')))
length = len(barr)
time = np.arange(0, length, 1)

open_g_data = moving_average_filter(open_g_data)

imuaccel_list = one_cell_imu_data_extraction.loadv2("C:/Users/jaeuk/Downloads/cell_imu_data_test (2).im", True)
length = len(imuaccel_list[:, 2])
print("IMU data length = ", length)
accel_len = np.arange(0, length , 1)


# Sine sample with some noise and copy to y1 and y2 with a 1-second lag
sr = 1024
# y = np.linspace(0, 2*np.pi, sr)
# y = np.tile(np.sin(y), 5)
# y += np.random.normal(0, 5, y.shape)
# y1 = y[sr:4*sr]
# y2 = y[:3*sr]

a= open_g_data
b= imuaccel_list[:, 1]

print("before shifting ")

drawing_xyz_accel(open_g_data, imuaccel_list[:, 0], imuaccel_list[:, 1], imuaccel_list[:, 2], time, accel_len)

print(type(a), type(b))

asd = np.argmax(np.convolve(a[::-1],b,mode='valid'))
print("Shifting frame = ", asd)
print("After shifting frame")

accel_len = np.arange(0, len(imuaccel_list[asd:, 1]) , 1)
plt.plot(time, open_g_data, "-y", label='RPM-G')
plt.plot(accel_len, imuaccel_list[asd:, 1], "-g", label='Y')
plt.plot(accel_len, imuaccel_list[asd:, 0], "-r", label='X')
plt.plot(accel_len, imuaccel_list[asd:, 2], "-b", label='Z')

plt.show()

# lag_finder(open_g_data, imuaccel_list[:, 2], sr)
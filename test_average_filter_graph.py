import numpy as np
import matplotlib.pyplot as plot
from sklearn.metrics import accuracy_score
# import similaritymeasures



def drawing_xyz_accel(g, g_wo_filter, time):
    frame = plot.gca()
    # N = 55
    # re = np.convolve(g, np.ones((N,))/N, mode='valid')
    # print(re)
    # plot.plot(time[:len(time) - N + 1], re, "-y", label="G-RPM")
    plot.plot(time, g, "-y", label="G-RPM")

    # plot.plot(time, g_wo_filter, "-r", label="X", alpha=0.5)
    # plot.plot(accel_len, ay, "-g", label="Y")
    # plot.plot(accel_len, az, "-b", label="Z")

    plot.title('Gravitational acceleration from RPM')

    # Give x axis label for the sine wave plot
    plot.xlabel('frame')

    # Give y axis label for the sine wave plot
    plot.ylabel('g(9.8m/s2)')
    plot.ylim(-20, 20)
    plot.grid(True)
    # frame.axes.get_yaxis().set_visible(False)

    # plot.axhline(y=0, color='k')
    plot.axhline(y=0, color='k')
    plot.legend(frameon=False)
    plot.get_current_fig_manager().show()
    plot.show()


open_g_data = []
print("Drawing G graph and x/y/z cell acceleration graph ")
with open('rpm_data.txt', 'rb') as fp:
    barr = fp.readlines()
    print('total length', len(barr))

for line in barr:
    open_g_data.append(float(line[:-2].decode('utf-8')))
length = len(barr)
time = np.arange(0, length, 1)


N=100 # number of points to test on each side of point of interest, best if even
padded_x = np.insert(np.insert(np.insert(open_g_data, len(open_g_data), np.empty(int(N/2))*np.nan), 0, np.empty(int(N/2))*np.nan ),0,0)
print(padded_x)
n_nan = np.cumsum(np.isnan(padded_x))
print(n_nan)
cumsum = np.nancumsum(padded_x)
print(cumsum)
window_sum = cumsum[N+1:] - cumsum[:-(N+1)] - open_g_data # subtract value of interest from sum of all values within window
print(window_sum)
window_n_nan = n_nan[N+1:] - n_nan[:-(N+1)] - np.isnan(open_g_data)
print(window_n_nan)
window_n_values = (N - window_n_nan)
print(window_n_values)
movavg = (window_sum) / (window_n_values)
print(movavg)
print(len(movavg))

time = np.arange(0, len(movavg), 1)

print(movavg, len(movavg))
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print(open_g_data, len(open_g_data))

print(type(movavg), type(open_g_data))

sim_x = time

sim_movavg = np.zeros((len(time), 2))
sim_movavg[:, 0] = sim_x
sim_movavg[:, 1] = movavg

sim_raw_g = np.zeros((len(time), 2))
sim_raw_g[:, 0] = sim_x
sim_raw_g[:, 1] = np.array(open_g_data)


# # quantify the difference between the two curves using PCM
# pcm = similaritymeasures.pcm(sim_movavg, sim_raw_g)
#
# # quantify the difference between the two curves using
# # Discrete Frechet distance
# df = similaritymeasures.frechet_dist(sim_movavg, sim_raw_g)
#
# # quantify the difference between the two curves using
# # area between two curves
# area = similaritymeasures.area_between_two_curves(sim_movavg, sim_raw_g)
#
# # quantify the difference between the two curves using
# # Curve Length based similarity measure
# cl = similaritymeasures.curve_length_measure(sim_movavg, sim_raw_g)
#
# # quantify the difference between the two curves using
# # Dynamic Time Warping distance
# dtw, d = similaritymeasures.dtw(sim_movavg, sim_raw_g)
#
# # print the results
# print(pcm, df, area, cl, dtw)

drawing_xyz_accel(movavg, open_g_data, time)
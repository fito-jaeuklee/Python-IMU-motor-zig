import serial
import time
from scipy import signal
import math
import matplotlib.pyplot as plot
import numpy as np
import os
import one_cell_imu_data_extraction
from scipy.stats import linregress
from tkinter import filedialog
import collections
import data_analysis_tool as da

hz_setup = 100
data_count = 20000
RESAMPLING_RATIO = 100 / 158.85
"""Distance between origin point of disk and imu sensor (cm)"""
RADIUS = 13
file_path = "./cell data/6-4"
cut_max_limit = 13000
saturation_sector = [3700, 4000, 5700, 6000, 7700, 8000, 9700, 10000]


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

    with open(filename, 'w') as fp:
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


def drawing_xyz_accel(g, ax, ay, az, time, accel_len, min, max):
    frame = plot.gca()
    # N = 55
    # re = np.convolve(g, np.ones((N,))/N, mode='valid')
    sum_xyz = ax + ay + az

    print(saturation_sector)

    sum_mean_list, ref_mean_list = da.mean_each_sector(file_name, sum_xyz, g, saturation_sector)
    print(sum_mean_list, ref_mean_list)
    plot.plot(time, g, "-y", label="G-RPM")

    plot.plot(accel_len, ax, "-r", label="X")
    plot.plot(accel_len, ay, "-g", label="Y")
    plot.plot(accel_len, az, "-b", label="Z")
    plot.plot(accel_len, sum_xyz, label="SUM XYZ")

    # slope, inter, r_v, p_v, std_err = linregress(time[:10000], g[:10000])
    # sum_sl, sum_inter, sum_r, sum_p, sum_str_err = linregress(accel_len[:10000], sum_xyz[:10000])
    # print(slope, inter, r_v, p_v)
    # print("RPM-G LR = %fx + %f" % (slope, inter))
    # print("XYZ LR = %fx + %f" % (sum_sl, sum_inter))
    #
    # line_total = slope*time + inter
    # sum_line = sum_sl * accel_len + sum_inter
    #
    # plot.plot(time[:10000], line_total[:10000], '-y')
    #
    # plot.plot(accel_len[:10000], sum_line[:10000], '-b')

    # slope1, inter1, r_v1, p_v1, std_err1 = linregress(time[2100:4130], g[2100:4130])
    # slope2, inter2, r_v2, p_v2, std_err2 = linregress(time[4130:6100], g[4130:6100])
    # slope3, inter3, r_v3, p_v3, std_err3 = linregress(time[6100:8090], g[6100:8090])
    # slope4, inter4, r_v4, p_v4, std_err4 = linregress(time[8090:10050], g[8090:10050])
    #
    # line1 = slope1 * time + inter1
    # line2 = slope2 * time + inter2
    # line3 = slope3 * time + inter3
    # line4 = slope4 * time + inter4
    #
    # print("RPM-G1 LR = %fx + %f" % (slope1, inter1))
    # print("RPM-G2 LR = %fx + %f" % (slope2, inter2))
    # print("RPM-G3 LR = %fx + %f" % (slope3, inter3))
    # print("RPM-G4 LR = %fx + %f" % (slope4, inter4))
    #
    # plot.plot(time[2100:4130], line1[2100:4130], '-y')
    # plot.plot(time[4130:6100], line2[4130:6100], '-y')
    # plot.plot(time[6100:8090], line3[6100:8090], '-y')
    # plot.plot(time[8090:10050], line4[8090:10050], '-y')

    # slope1, inter1, r_v1, p_v1, std_err1 = linregress(np.arange(0, 5), sum_mean_list)
    # slope2, inter2, r_v2, p_v2, std_err2 = linregress(np.arange(0, 5), ref_mean_list)
    #
    # line1 = slope1 * np.arange(0, 5) + inter1
    # line2 = slope2 * np.arange(0, 5) + inter2
    #
    # print("sum_mean_list LR = %fx + %f" % (slope1, inter1))
    # print("ref_mean_list LR = %fx + %f" % (slope2, inter2))

    # plot.plot(time[2100:4130], line1[2100:4130], '-y')
    # plot.plot(time[4130:6100], line2[4130:6100], '-y')

    plot.title('Gravitational acceleration from RPM')

    # Give x axis label for the sine wave plot
    plot.xlabel('frame')

    # Give y axis label for the sine wave plot
    plot.ylabel('g(9.8m/s2)')
    plot.ylim(min, max)
    plot.grid(True)
    # frame.axes.get_yaxis().set_visible(False)

    # plot.axhline(y=0, color='k')
    plot.axhline(y=0, color='k')
    plot.legend(frameon=False)
    plot.get_current_fig_manager().show()

    cut_section = da.section_frame_cut

    plot.axvline(x=cut_section[1], color='r')
    plot.axvline(x=cut_section[2], color='r')
    plot.axvline(x=cut_section[3], color='r')
    plot.axvline(x=cut_section[4], color='r')
    plot.axvline(x=cut_section[5], color='r')

    plot.show()

    da.compare_rpm_LR_to_cell_data_NLR(ref_mean_list, sum_mean_list)


def load_rpm_to_g_data():
    g_data = []
    # print("Drawing G graph and x/y/z cell acceleration graph ")
    with open(file_path + '/rpm_data.txt', 'rb') as fp:
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


if __name__ == '__main__':
    while 1:
        input_byte = ''
        input_byte = input(" ---------------------------------------------------------  "
                           "\r\n 1. Start RPM recording "
                           "\r\n 2. Extract CELL IMU data "
                           "\r\n 3. Drawing graph (0/30/60/90/120/150 degree)"
                           "\r\n 4. Time sync & Moving average filter apply to RPM data "
                           "\r\n 5. Data analysis Acceleration "
                           "\r\n 6. Data analysis Gyroscope "
                           "\r\n 7. Save g data to rpm data "
                           "\r\n ---------------------------------------------------------  "
                           "\r\n Enter your command = ")

        command = input_byte.encode('utf-8')

        data = []

        if command.decode('utf-8') == "1":
            dev = serial.Serial("/dev/tty.usbserial-1410", baudrate=2000000, timeout=0.3)
            time.sleep(1)

            # print("Start RPM recording")
            start = time.time()
            for _ in range(data_count):
                dev.write(command)
                line = dev.readline()
                data.append(line)

            # # print(data)
            # print(len(data))
            execution_time = time.time() - start
            # print("Execution time = ", execution_time)

            clean_data = clean_serial_data(data)

            save_to_txt(clean_data, 'rpm_data.txt', execution_time)
        elif command.decode("utf-8") == "2":
            # print("Extract cell imu data ")
            cell_com_port_name = one_cell_imu_data_extraction.get_cell_com_port(1155)
            # print(cell_com_port_name)
            nand_flag = one_cell_imu_data_extraction.read_and_save_imu_data(cell_com_port_name, file_path,
                                                                            "cell_imu_data_test", 4000)
            if nand_flag == True:
                one_cell_imu_data_extraction.erase_cell_nand_flash(cell_com_port_name)

        elif command.decode('utf-8') == "3":
            open_g_data, frame_length = load_rpm_to_g_data()

            file_list = os.listdir("./cell data")

            for file_name in file_list:
                # print(file_name)
                imuaccel_list, imugyro_list, imumag_list = one_cell_imu_data_extraction.loadv2(
                    "./cell data/" + file_name, True)
                length = len(imuaccel_list[:, 0])
                # print("IMU data length = ", length)
                accel_len = np.arange(0, length, 1)

                drawing_xyz_accel(open_g_data, imuaccel_list[:, 0], imuaccel_list[:, 1], imuaccel_list[:, 2],
                                  frame_length, accel_len)

        elif command.decode('utf-8') == "4":
            open_g_data, frame_length = load_rpm_to_g_data()
            # print("Apply moving average")
            movavg_g_data = moving_average_filter(open_g_data)

            file_list = os.listdir("./cell data")

            for file_name in file_list:
                # print(file_name)
                imuaccel_list, imugyro_list, imumag_list = one_cell_imu_data_extraction.loadv2(
                    "./cell data/" + file_name, True)
                length = len(imuaccel_list[:, 0])
                # print("IMU data length = ", length)
                accel_len = np.arange(0, length, 1)

                # print("Time shifting")
                shifting_frame = time_shifting(movavg_g_data, x, y, z)
                # print(shifting_frame)

                x, y, z = check_absolute_val_and_change_value(imuaccel_list[:, 0], imuaccel_list[:, 1],
                                                              imuaccel_list[:, 2])

                accel_len = np.arange(0, len(imuaccel_list[shifting_frame:, 1]), 1)

                drawing_xyz_accel(movavg_g_data, x[shifting_frame:], y[shifting_frame:], z[shifting_frame:],
                                  frame_length, accel_len)

        elif command.decode('utf-8') == "5":
            open_g_data, frame_length = load_rpm_to_g_data()
            # print("Apply moving average")
            movavg_g_data = moving_average_filter(open_g_data)

            file_list = os.listdir(file_path)

            for file_name in file_list:
                # print(file_name)
                if '.im' in file_name:
                    imuaccel_list, imugyro_list, imumag_list = one_cell_imu_data_extraction.loadv2(
                        file_path + "/" + file_name, True)
                    length = len(imuaccel_list[:, 0])
                    # print("IMU data length = ", length)
                    accel_len = np.arange(0, length, 1)

                    print("Change Minus value to ")
                    # x, y, z = check_absolute_val_and_change_value(imuaccel_list[:, 0], imuaccel_list[:, 1],
                    #                                               imuaccel_list[:, 2])

                    x, y, z = -imuaccel_list[:, 0], -imuaccel_list[:, 1], -imuaccel_list[:, 2]
                    # print("Time shifting")
                    shifting_frame = time_shifting(movavg_g_data, x, y, z)


                    # print(shifting_frame)

                    accel_len = np.arange(0, len(imuaccel_list[shifting_frame:, 1]), 1)

                    x, y, z = da.correct_bias(x, y, z, shifting_frame)

                    # x, y, z = np.abs(x), np.abs(y), np.abs(z)

                    drawing_xyz_accel(movavg_g_data, x[shifting_frame:], y[shifting_frame:], z[shifting_frame:],
                                      frame_length, accel_len, -20, 20)


                    # for i in range(0, 5):
                    #     # print("Calculate similarity for each section = ", i)
                    #     cos_x, cos_y, cos_z = da.calculate_cosine_similarity(movavg_g_data, x[shifting_frame:],
                    #                                                          y[shifting_frame:], z[shifting_frame:], i)
                    #     # print("cosine_result = ", cos_x, cos_y, cos_z)
                    #
                    #     r_x, r_y, r_z = da.calculate_pearson_correlation(movavg_g_data, x[shifting_frame:],
                    #                                                      y[shifting_frame:], z[shifting_frame:], i)
                    #     # print("pearson_correlation_result = ", r_x, r_y, r_z)
                else:
                    print("\r\n")

        elif command.decode('utf-8') == "6":
            print("")
            dps_data, dps_frame_length = load_rpm_to_dps_data()

            movavg_dps_data = moving_average_filter(dps_data)

            file_list = os.listdir(file_path)

            for file_name in file_list:
                # print(file_name)
                if '.im' in file_name:
                    imuaccel_list, imugyro_list, imumag_list = one_cell_imu_data_extraction.loadv2(
                        file_path + "/" + file_name, True)
                    length = len(imugyro_list[:, 0])
                    # print("IMU data length = ", length)
                    gyro_len = np.arange(0, length, 1)
                    print("----------------", len(gyro_len))

                    gy_x, gy_y, gy_z = check_absolute_val_and_change_value(imugyro_list[:, 0], imugyro_list[:, 1],
                                                                           imugyro_list[:, 2])
                    print(gy_x, gy_y, gy_z)

                    # print("Time shifting")
                    shifting_frame = time_shifting(movavg_dps_data, gy_x, gy_y, gy_z)
                    # print(shifting_frame)

                    gyro_len = np.arange(0, len(imugyro_list[shifting_frame:, 1]), 1)

                    gy_x, gy_y, gy_z = da.correct_bias(gy_x, gy_y, gy_z, shifting_frame)

                    drawing_xyz_accel(movavg_dps_data, gy_x[shifting_frame:], gy_y[shifting_frame:],
                                      gy_z[shifting_frame:],
                                      dps_frame_length, gyro_len, -10, 2100)

        elif command.decode('utf-8') == "7":
            rpm_data, frame_length = load_rpm_to_g_data()

            with open("g_to_rpm.txt", 'w') as fp:
                for buf in rpm_data:
                    buf = abs(buf)
                    rpm = round(math.sqrt(buf / (1.11876 * math.pow(10, -5) * RADIUS)), 4)
                    fp.write(str(rpm) + '\n')

        else:
            print("Entered wrong number")

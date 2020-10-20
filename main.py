import serial
import time
from scipy import signal
import math
import matplotlib.pyplot as plot
import numpy as np
import one_cell_imu_data_extraction
from scipy.signal import savgol_filter

offset = 618
hz_setup = 100
data_count = 10000
RESAMPLING_RATIO = 100 / 158.85
"""Distance between origin point of disk and imu sensor (cm)"""
RADIUS = 13
file_path = "C:/Users/jaeuk/workspace/IMU_calibration_with_motor/cell data"


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
    print(resampled_data)

    for line in resampled_data:
        cut_float_resample_data.append(line)

    with open(filename, 'w') as fp:
        for buf in cut_float_resample_data:
            fp.write(str(buf) + '\n')


def calculate_rpm_to_g(rpm_data):
    g_list = []
    print("@@@@@@", rpm_data)
    for rpm in rpm_data:
        rpm = float(rpm)
        g = 1.11786 * math.pow(10, -5) * math.pow(rpm, 2) * RADIUS
        g_list.append(round(g, 4))

    return g_list


def resampling_g_data(data, sampling_time):
    """ Resampling 158.83Hz to about 100Hz """
    print("resampled number = ", int(sampling_time * hz_setup))
    resampled = signal.resample(data, int(sampling_time * hz_setup))
    print("Resample data length = ", len(resampled))

    return resampled


def drawing_xyz_accel(g, ax, ay, az, time, accel_len):
    frame = plot.gca()
    N = 55
    re = np.convolve(g, np.ones((N,))/N, mode='valid')
    print(re)
    plot.plot(time[:len(time) - N + 1], re, "-y", label="G-RPM")


    # plot.plot(accel_len, ax, "-r", label="X")
    # plot.plot(accel_len, ay, "-g", label="Y")
    # plot.plot(accel_len, az, "-b", label="Z")

    plot.title('Gravitational acceleration from RPM')

    # Give x axis label for the sine wave plot
    plot.xlabel('frame')

    # Give y axis label for the sine wave plot
    plot.ylabel('g(9.8m/s2)')
    plot.ylim(0, 20)
    plot.grid(True)
    # frame.axes.get_yaxis().set_visible(False)

    # plot.axhline(y=0, color='k')
    plot.axhline(y=0, color='k')
    plot.legend(frameon=False)
    plot.get_current_fig_manager().show()
    plot.show()


if __name__ == '__main__':
    while 1:
        input_byte = ''
        input_byte = input(" -------------------------- \r\n 1. Start RPM recording \r\n 2. Extract CELL IMU data \r\n "
                           "3. Drawing graph \r\n -------------------------- \r\n Enter your command = ")
        command = input_byte.encode('utf-8')

        data = []

        if command.decode('utf-8') == "1":
            dev = serial.Serial("COM65", baudrate=2000000, timeout=0.3)
            time.sleep(1)

            print("Start RPM recording")
            start = time.time()
            for _ in range(data_count):
                dev.write(command)
                line = dev.readline()
                data.append(line)

            # print(data)
            print(len(data))
            execution_time = time.time() - start
            print("Execution time = ", execution_time)

            clean_data = clean_serial_data(data)

            save_to_txt(clean_data, 'rpm_data.txt', execution_time)
        elif command.decode("utf-8") == "2":
            print("Extract cell imu data ")
            cell_com_port_name = one_cell_imu_data_extraction.get_cell_com_port(1155)
            print(cell_com_port_name)
            one_cell_imu_data_extraction.read_and_save_imu_data(cell_com_port_name, file_path, "cell_imu_data_test", 4000)
            one_cell_imu_data_extraction.erase_cell_nand_flash(cell_com_port_name)

        elif command.decode('utf-8') == "3":
            open_g_data = []
            print("Drawing G graph and x/y/z cell acceleration graph ")
            with open('rpm_data.txt', 'rb') as fp:
                barr = fp.readlines()
                print('total length', len(barr))

            for line in barr:
                open_g_data.append(float(line[:-2].decode('utf-8')))
            length = len(barr)
            time = np.arange(0, length, 1)

            imuaccel_list = one_cell_imu_data_extraction.loadv2('./cell data/cell_imu_data_test.im', True)
            length = len(imuaccel_list[:, 0])
            print("IMU data length = ", length)
            accel_len = np.arange(0, length-offset, 1)

            drawing_xyz_accel(open_g_data, imuaccel_list[offset:, 0], imuaccel_list[offset:, 1], imuaccel_list[offset:, 2], time, accel_len)

        else:
            print("Entered wrong number")

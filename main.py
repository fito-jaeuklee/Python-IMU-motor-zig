import serial
import time
import math
import matplotlib.pyplot as plot
import numpy as np
import os
import one_cell_imu_data_extraction
import data_analysis_tool as da
import data_manage as dm

data_count = 20000
COM_PORT_NAME = "/dev/tty.usbserial-1440"
saturation_sector = [3700, 4000, 5700, 6000, 7700, 8000, 9700, 10000]


def drawing_xyz_accel(g, ax, ay, az, time, accel_len, min, max):
    frame = plot.gca()
    # N = 55
    # re = np.convolve(g, np.ones((N,))/N, mode='valid')
    sum_xyz = ax + ay + az

    print(saturation_sector)

    sum_mean_list, ref_mean_list = da.mean_each_sector(file_name, sum_xyz, g, saturation_sector)
    print(sum_mean_list, ref_mean_list)
    plot.plot(time, g, "-y", label="Temperature(cel)")
    # plot.plot(time, g, "-y", label="Degree Per Second[DPS]-RPM")
    #
    plot.plot(accel_len, ax, "-r", label="X")
    plot.plot(accel_len, ay, "-g", label="Y")
    plot.plot(accel_len, az, "-b", label="Z")
    plot.plot(accel_len, sum_xyz, label="SUM XYZ")

    plot.rcParams["figure.figsize"] = (12, 10)

    plot.title('Acceleration with temperature')

    # Give x axis label for the sine wave plot
    plot.xlabel('frame')

    # Give y axis label for the sine wave plot
    plot.ylabel('g(9.8m/s2) Or Temperature')
    # plot.ylabel('DPS(º/s)')
    plot.ylim(min, max)
    plot.grid(True)
    # frame.axes.get_yaxis().set_visible(False)

    # plot.axhline(y=0, color='k')
    plot.axhline(y=0, color='k')
    plot.legend(frameon=False)
    plot.get_current_fig_manager().show()

    cut_section = da.section_frame_cut

    # plot.axvline(x=cut_section[1], color='r')
    # plot.axvline(x=cut_section[2], color='r')
    # plot.axvline(x=cut_section[3], color='r')
    # plot.axvline(x=cut_section[4], color='r')
    # plot.axvline(x=cut_section[5], color='r')

    plot.rcParams["figure.figsize"] = (12, 6)

    plot.show()

    print("dfgjklahklgsd", ref_mean_list, sum_mean_list)

    # da.compare_rpm_LR_to_cell_data_NLR(ref_mean_list, sum_mean_list)


if __name__ == '__main__':
    while 1:
        input_byte = ''
        input_byte = input(" ---------------------------------------------------------  "
                           "\r\n 1. Start RPM recording & Save gravity and DPS data"
                           "\r\n 2. Extract CELL IMU data "
                           "\r\n 3. Drawing graph (0/30/60/90/120/150 degree)"
                           "\r\n 4. Time sync & Moving average filter apply to RPM data "
                           "\r\n 5. Data analysis Accelerator "
                           "\r\n 6. Data analysis Gyroscope "
                           "\r\n 7. Data analysis Magnetometer "
                           "\r\n ---------------------------------------------------------  "
                           "\r\n Enter your command = ")

        command = input_byte.encode('utf-8')

        data = []

        if command.decode('utf-8') == "1":
            dev = serial.Serial(COM_PORT_NAME, baudrate=2000000, timeout=0.3)
            time.sleep(1)

            print("Start RPM recording")
            start = time.time()
            for _ in range(data_count):
                dev.write(command)
                line = dev.readline()
                data.append(line)

            print(len(data))
            execution_time = time.time() - start
            print("Execution time = ", execution_time)

            clean_data = dm.clean_serial_data(data)

            dm.save_to_txt(clean_data, 'rpm_to_g_data.txt', execution_time)

            rpm_data, frame_length = dm.load_rpm_to_g_data()

            with open(dm.file_path + "/" + "g_to_rpm.txt", 'w') as fp:
                for buf in rpm_data:
                    buf = abs(buf)
                    rpm = round(math.sqrt(buf / (1.11876 * math.pow(10, -5) * dm.RADIUS)), 4)
                    fp.write(str(rpm) + '\n')

        elif command.decode("utf-8") == "2":
            print("Extract cell imu data ")
            user_type_insert = input("Type cell data information = ")

            cell_com_port_name = one_cell_imu_data_extraction.get_cell_com_port(1155)
            print(cell_com_port_name)
            nand_flag = one_cell_imu_data_extraction.read_and_save_imu_data(cell_com_port_name, dm.file_path,
                                                                            "cell_imu_data_test_" + user_type_insert,
                                                                            4000)
            print("imu done")
            time.sleep(1)
            nand_flag = one_cell_imu_data_extraction.read_and_save_gps_data(cell_com_port_name, dm.file_path,
                                                                            "cell_imu_data_test_" + user_type_insert,
                                                                            4000)
            print("gps done")

            nand_flag = one_cell_imu_data_extraction.read_and_save_temp_data(cell_com_port_name, dm.file_path,
                                                                             "cell_imu_data_test_" + user_type_insert,
                                                                             4000)
            if nand_flag == True:
                print("Nand flash erase")
                one_cell_imu_data_extraction.erase_cell_nand_flash(cell_com_port_name)

        elif command.decode('utf-8') == "3":
            open_g_data, frame_length = dm.load_rpm_to_g_data()

            file_list = os.listdir("./cell data")

            for file_name in file_list:
                print(file_name)
                imuaccel_list, imugyro_list, imumag_list = one_cell_imu_data_extraction.loadv2(
                    "./cell data/" + file_name, True)
                length = len(imuaccel_list[:, 0])
                print("IMU data length = ", length)
                accel_len = np.arange(0, length, 1)

                drawing_xyz_accel(open_g_data, imuaccel_list[:, 0], imuaccel_list[:, 1], imuaccel_list[:, 2],
                                  frame_length, accel_len)

        elif command.decode('utf-8') == "4":
            open_g_data, frame_length = dm.load_rpm_to_g_data()
            print("Apply moving average")
            movavg_g_data = dm.moving_average_filter(open_g_data)

            file_list = os.listdir("./cell data")

            for file_name in file_list:
                # print(file_name)
                imuaccel_list, imugyro_list, imumag_list = one_cell_imu_data_extraction.loadv2(
                    "./cell data/" + file_name, True)
                length = len(imuaccel_list[:, 0])
                print("IMU data length = ", length)
                accel_len = np.arange(0, length, 1)

                print("Time shifting")
                shifting_frame = dm.time_shifting(movavg_g_data, x, y, z)
                print(shifting_frame)

                x, y, z = dm.check_absolute_val_and_change_value(imuaccel_list[:, 0], imuaccel_list[:, 1],
                                                                 imuaccel_list[:, 2])

                accel_len = np.arange(0, len(imuaccel_list[shifting_frame:, 1]), 1)

                drawing_xyz_accel(movavg_g_data, x[shifting_frame:], y[shifting_frame:], z[shifting_frame:],
                                  frame_length, accel_len)

        elif command.decode('utf-8') == "5":
            open_g_data, frame_length = dm.load_rpm_to_g_data()
            print("Apply moving average")
            movavg_g_data = dm.moving_average_filter(open_g_data)

            file_list = os.listdir(dm.file_path)

            temp_list, temp_len = one_cell_imu_data_extraction.imu_temp_load(
                dm.file_path + "/" + "cell_imu_data_test_nh2.txt")
            temp_len = np.arange(0, temp_len, 1)

            print(temp_list[:2000])
            print(temp_len)

            for file_name in file_list:
                # print(file_name)
                if 'cell_imu_data_test_nh2.im' in file_name:
                    imuaccel_list, imugyro_list, imumag_list = one_cell_imu_data_extraction.loadv2(
                        dm.file_path + "/" + file_name, True)

                    length = len(imuaccel_list[:, 0])
                    # print("IMU data length = ", length)
                    accel_len = np.arange(0, length, 1)

                    print("Change Minus value to ")
                    # x, y, z = check_absolute_val_and_change_value(imuaccel_list[:, 0], imuaccel_list[:, 1],
                    #                                               imuaccel_list[:, 2])

                    x, y, z = -imuaccel_list[:, 0], imuaccel_list[:, 1], -imuaccel_list[:, 2]
                    print("Time shifting")
                    shifting_frame = dm.time_shifting(movavg_g_data, x, y, z)

                    if shifting_frame is None:
                        shifting_frame = 0
                    print(shifting_frame)
                    shifting_frame = 0

                    accel_len = np.arange(0, len(imuaccel_list[shifting_frame:, 1]), 1)

                    # x, y, z = da.correct_bias(x, y, z, shifting_frame)
                    print(temp_list, temp_len)
                    print(accel_len)

                    drawing_xyz_accel(temp_list, x[shifting_frame:], y[shifting_frame:], z[shifting_frame:],
                                      temp_len, accel_len, -40, 150)

                    # for i in range(0, 5):
                    #     # print("Calculate similarity for each section = ", i)
                    #     cos_x, cos_y, cos_z = da.calculate_cosine_similarity(movavg_g_data, x[shifting_frame:],
                    #                                                          y[shifting_frame:], z[shifting_frame:], i)
                    #     # print("cosine_result = ", cos_x, cos_y, cos_z)
                    #
                    #     r_x, r_y, r_z = da.calculate_pearson_correlation(movavg_g_data, x[shifting_frame:],
                    #                                                      y[shifting_frame:], z[shifting_frame:], i)
                    #                                                     y[shifting_frame:], z[shifting_frame:], i)
                    #     # print("pearson_correlation_result = ", r_x, r_y, r_z)
                else:
                    print("not im file\r\n")

        elif command.decode('utf-8') == "6":
            print("")
            dps_data, dps_frame_length = dm.load_rpm_to_dps_data()

            movavg_dps_data = dm.moving_average_filter(dps_data)

            file_list = os.listdir(dm.file_path)

            temp_list, temp_len = one_cell_imu_data_extraction.imu_temp_load(
                dm.file_path + "/" + "cell_imu_data_test_h5.txt")
            temp_len = np.arange(0, temp_len, 1)

            for file_name in file_list:
                # print(file_name)₩
                if 'cell_imu_data_test_h5.im' in file_name:
                    imuaccel_list, imugyro_list, imumag_list = one_cell_imu_data_extraction.loadv2(
                        dm.file_path + "/" + file_name, True)
                    length = len(imugyro_list[:, 0])
                    print("@@@@@@@@@@@@@@", imumag_list)
                    # print("IMU data length = ", length)
                    gyro_len = np.arange(0, length, 1)
                    print("----------------", len(gyro_len))

                    gy_x, gy_y, gy_z = dm.check_absolute_val_and_change_value(imugyro_list[:, 0], imugyro_list[:, 1],
                                                                              imugyro_list[:, 2])
                    print(gy_x, gy_y, gy_z)

                    # print("Time shifting")
                    shifting_frame = dm.time_shifting(movavg_dps_data, gy_x, gy_y, gy_z)
                    # print(shifting_frame)

                    gyro_len = np.arange(0, len(imugyro_list[shifting_frame:, 1]), 1)

                    # gy_x, gy_y, gy_z = da.correct_bias(gy_x, gy_y, gy_z, shifting_frame)

                    drawing_xyz_accel(temp_list, gy_x[shifting_frame:], gy_y[shifting_frame:],
                                      gy_z[shifting_frame:],
                                      temp_len, gyro_len, -10, 2100)

                    # for i in range(0, 5): print("Calculate similarity for each section = ", file_name, i) cos_x,
                    # cos_y, cos_z = da.calculate_cosine_similarity(movavg_dps_data, gy_x[shifting_frame:],
                    # gy_y[shifting_frame:], gy_z[shifting_frame:], i) print("cosine_result = ", cos_x, cos_y, cos_z)
                    #
                    # r_x, r_y, r_z = da.calculate_pearson_correlation(movavg_dps_data, gy_x[shifting_frame:],
                    # gy_y[shifting_frame:], gy_z[shifting_frame:], i) print("pearson_correlation_result = ", r_x,
                    # r_y, r_z)

        elif command.decode('utf-8') == "7":
            print("Analyse magnetometer data")

        else:
            print("Entered wrong number")

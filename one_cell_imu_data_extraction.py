import serial.tools.list_ports
import serial
import time
import numpy as np
import struct
import matplotlib.pyplot as plot

BAUDRATE = 230400
CELL_IMU_CAL_RESP_SIZE = 128
SYSCOMMAND_SET_READ_IMU_CAL = "AC C0 01 24 49"
SYSCOMMAND_OLD_UPLOAD_IMU_DATA = "AC C0 01 E1 8C"
SYSCOMMAND_OLD_UPLOAD_GPS_DATA = "AC C0 02 E0 00 8E"
SYSCOMMAND_OLD_UPLOAD_TEMP_DATA = "AC C0 01 E4 89"
CELL_GPS_IMU_READ_CHUCK_SIZE = 2048
SYSCOMMAND_ERASE_NAND_FLASH = "AC C0 01 15 78"
SYSCOMMAND_ERASE_NAND_FLASH_RESP_SIZE = 7
IMU_ERR_STR = "IMUERR"
GPS_ERR_STR = "GPSERR"
GPS_END_STR = "GPSEND"
file_path = "./cell data"
frame_to_time_scaler = 1 / 6000

# Add new command - Renewal data manager
DATA_READ_STOP = "AC C0 01 F3 9E"


def to_float(x):
    try:
        return float(x)
    except:
        return 0.


def get_cell_com_port(cell_vendor_id):
    cell_port_name = ''
    cell_port_list = []
    for port in serial.tools.list_ports.comports():
        if port.vid == cell_vendor_id:
            cell_port_name = port.device
            cell_port_list.append(cell_port_name)
    if cell_port_name == '':
        print("Please make sure Cell plug into docking")

    return cell_port_list


def read_and_save_imu_data(port, file_save_path, filename, imu_page_size):
    print(file_save_path)
    imu_data_chuck_validation_cnt = 0
    cell_read_error_serial = []
    imu_loop_cnt = 0
    # TODO : 플러그를 뽑는게 아니라 실제로 timeout 걸리는 데이터 플로우를 만들어서 체크해야함

    try:
        with serial.Serial(port[0], BAUDRATE, timeout=0.1) as ser, \
                open('%s/%s.im' % (file_save_path, filename), mode='w+b') as f:
            imu_cal = bytes.fromhex(SYSCOMMAND_SET_READ_IMU_CAL)
            ser.write(imu_cal)
            imu_cal_data = ser.read(CELL_IMU_CAL_RESP_SIZE)
            imu_cal_data = imu_cal_data[5:-3]
            f.write(imu_cal_data)

            imu = bytes.fromhex(SYSCOMMAND_OLD_UPLOAD_IMU_DATA)
            ser.write(imu)

            imu_data_reading_end_flag = 1
            imu_error_str_delete_flag = 0
            while imu_data_reading_end_flag:
                data = ser.read(CELL_GPS_IMU_READ_CHUCK_SIZE)
                str_data = str(data)

                if len(data) != CELL_GPS_IMU_READ_CHUCK_SIZE:
                    if (str_data.find(IMU_ERR_STR)) != -1:
                        print(" Found IMUERR ")
                        imu_data_reading_end_flag = 1
                        imu_error_str_delete_flag = 0
                    elif imu_data_chuck_validation_cnt > imu_page_size * 0.95:
                        print("IMU 0.95 Here")
                        imu_data_reading_end_flag = 0
                    elif (str_data.find("IMUEND")) != -1:
                        print("end done")
                        imu_data_reading_end_flag = 0
                    else:
                        print("IMU else length = ", len(data))
                        # imu_data_reading_end_flag = 0
                        # cell_read_error_serial.append(filename)

                # if (str_data.find(IMU_ERR_STR)) != -1:
                #     print("imuerr happen didn't write to file")
                #     imu_error_str_delete_flag = 1

                if imu_error_str_delete_flag == 0:
                    imu_loop_cnt += 1
                    print(imu_loop_cnt)
                    f.write(data)
                    imu_error_str_delete_flag = 0
                imu_data_chuck_validation_cnt += 1

            imu_err = bytes.fromhex("AC C0 01 25 48")
            ser.write(imu_err)
            error_count = ser.read(20)
            str_data = str(hex(int.from_bytes(error_count, byteorder='big')))
            cell_imu_error_count = int("0x" + str_data[16:18] + str_data[18:20], 0)
            print("IMUERR count = ", cell_imu_error_count)
            print("IMU loop cnt = ", imu_loop_cnt)

        return True
    except:
        # print("Not open cell serial com port.")
        return False
        pass


def read_and_save_gps_data(port, file_save_path, filename, gps_page_size):
    # TODO: Return reading result to userinterface each cell
    # TODO: Detect end of data

    gps_data_chuck_validation_cnt = 0
    cell_read_error_serial_g = []

    print(" Start reading GPS data")
    try:
        with serial.Serial(port[0], BAUDRATE, timeout=1) as ser, \
                open('%s/%s.gp' % (file_save_path, filename), mode='w+b') as f:

            t_end = time.time() + 40

            gps = bytes.fromhex(SYSCOMMAND_OLD_UPLOAD_GPS_DATA)
            ser.write(gps)

            upload_command_response = ser.read(6)
            print(upload_command_response)

            gps_data_reading_end_flag = 1
            gps_error_str_delete_flag = 0
            while gps_data_reading_end_flag:
                if time.time() < t_end:
                    data = ser.read(CELL_GPS_IMU_READ_CHUCK_SIZE)
                    str_data = str(data)

                    if len(data) != CELL_GPS_IMU_READ_CHUCK_SIZE:
                        if (str_data.find(GPS_END_STR)) != -1:
                            gps_data_reading_end_flag = 0
                        elif (str_data.find(GPS_ERR_STR)) != -1:
                            print("GPSERR", gps_data_chuck_validation_cnt)
                            gps_data_reading_end_flag = 1
                            gps_error_str_delete_flag = 0
                        elif gps_data_chuck_validation_cnt > gps_page_size * 0.95:
                            print("GPS 0.95 Here")
                            gps_data_reading_end_flag = 0
                        else:
                            print("GPS else not 2048")
                            # gps_error_str_delete_flag = 1
                            # gps_data_reading_end_flag = 0
                            cell_read_error_serial_g.append(filename)
                    if gps_error_str_delete_flag == 0:
                        f.write(data)
                        gps_error_str_delete_flag = 0
                    gps_data_chuck_validation_cnt += 1
                else:
                    print("TIMEOUT ! ")
                    # add cell data stream stop command
                    while True:
                        print("check1")
                        data_stop = bytes.fromhex(DATA_READ_STOP)
                        ser.write(data_stop)
                        read_stop_response = ser.read(128)
                        if "STOP" in str(read_stop_response):
                            print(read_stop_response)
                            print("check2")
                            break
                        else:
                            print(read_stop_response)
                            print("Not found STOP resp")
                    break

            gps_err = bytes.fromhex("AC C0 01 25 48")
            ser.write(gps_err)
            error_count = ser.read(20)
            str_data = str(hex(int.from_bytes(error_count, byteorder='big')))
            cell_gps_error_count = int("0x" + str_data[12:14] + str_data[14:16], 0)
            print("GPSERR count = ", cell_gps_error_count)

        return True
    except:
        print("Not open cell serial com port.")
        return False
        pass

        # TODO : After reading gp/im data, check ERR count here
        # Create a new file name with the number of ERR count added


def read_and_save_temp_data(port, file_save_path, filename, gps_page_size):
    gps_data_chuck_validation_cnt = 0
    cell_read_error_serial_g = []
    print(" Start reading Temp data")
    try:
        with serial.Serial(port[0], BAUDRATE, timeout=1) as ser, \
                open('%s/%s.txt' % (file_save_path, filename), mode='w+b') as f:

            temp = bytes.fromhex(SYSCOMMAND_OLD_UPLOAD_TEMP_DATA)
            ser.write(temp)

            temp_data_reading_end_flag = 1
            temp_error_str_delete_flag = 0
            while temp_data_reading_end_flag:
                data = ser.read(1)
                str_data = str(data)

                print(str_data)
                print(len(str_data))
                f.write(data)
                if len(str_data) == 3:
                    temp_data_reading_end_flag = 0

        return True
    except:
        print("Not open cell serial com port.")
        return False
        pass


def imuread_firmwarev2(fpath):
    print(fpath)
    with open(fpath, 'rb') as f:
        barr = f.read()
        # print('total bytes', len(barr))

        i = 0
        while True:
            if chr(barr[i]) == '\r' and chr(barr[i + 1]) == '\n':
                idx = i
                break
            i += 1
        # # print(idx)
        if idx is not 0:
            configs = barr[:idx]
            configs = configs.split(b',')
            configs = list(map(to_float, configs))
            # # print('parsed:', configs)
            print("pdf . com.  computer rule human indeed..!")
        hz = configs[0]
        saved_biases = configs[1:]

        i = idx
        pred_bidx = idx + 6
        for _ in range(4):
            if chr(barr[i]) == 'B' and chr(barr[i + 1]) == 'I' and chr(barr[i + 2]) == 'A' and chr(barr[i + 3]) == 'S':
                idx = i
                break
            i += 1

        bidx = idx + 4

        if pred_bidx != bidx:
            # print('prefix "BIAS" not found, using prediected bias idx', pred_bidx)
            bidx = pred_bidx

        i = bidx
        while True:
            if chr(barr[i]) == 'E' and chr(barr[i + 1]) == 'N' and chr(barr[i + 2]) == 'D' \
                    and chr(barr[i + 3]) == '\r' and chr(barr[i + 4]) == '\n':
                idx = i
                break
            i += 1

        beidx = idx

        biases = []
        for i in range(bidx, beidx, 18):  # parse bias
            bigendpart = struct.unpack('>hhhhhh', barr[i:i + 12])
            littleendpart = struct.unpack('<hhh', barr[i + 12:i + 18])
            biases.append(bigendpart + littleendpart)

        i = idx  #
        while True:
            if chr(barr[i]) == '2' and chr(barr[i + 1]) == '0':
                idx = i + 2
                break
            i += 1

        i = idx
        while True:
            if chr(barr[i]) == '@' and chr(barr[i + 1]) == '@' and chr(barr[i + 2]) == '@' and chr(barr[i + 3]) == '@':
                edidx = i + 2
                break
            i += 1

        barr = barr[idx + 5:edidx]  # offset 12 repeat 20 ( 2 18 )

        vals = []
        for i in range(0, len(barr) - 20, 20):
            bigendpart = struct.unpack('>BBhhhhhh', barr[i:i + 14])
            littleendpart = struct.unpack('<hhh', barr[i + 14:i + 20])
            vals.append(bigendpart + littleendpart)
    return hz, saved_biases, biases, vals


def __preprocess_imuraw_wbias(raw, bias, magscale, return_ref_angle=False):
    imuraw = raw[:, 2:].astype(np.float32)
    # print(imuraw)

    ratio_acc = 16. / 32768.
    # ratio_gyro = 200000.0 / 32768.0 * (np.pi / 180.)
    # ratio_gyro = 1 / 15.881081
    ratio_gyro = 0.061800486
    ratio_mag = 10 * 4912.0 / 32768.
    gpsec = 9.80665

    # print('magnetometer scale:', magscale)
    imuraw[:, 6:] *= magscale

    # _, magcal, mangle = __find_bias(imuraw)

    # imuac = imuraw[..., :3] * ratio_acc * gpsec
    imuac = imuraw[..., :3] * ratio_acc
    imugy = (imuraw[..., 3:6] - bias[3:6]) * ratio_gyro
    # imumg = (imuraw[..., 6:] - magcal) * ratio_mag

    imumg = (imuraw[..., 6:]) * ratio_mag

    imumg = imumg[:, [1, 0, 2]]
    imumg[:, 2] = -imumg[:, 2]

    if return_ref_angle:
        return imuac, imugy, imumg

    return imuac, imugy, imumg


def loadv2(fpath, return_all=False):
    samplefreq, saved_bias, bias, r = imuread_firmwarev2(fpath)
    r = np.array(r)

    bias = np.array(bias).mean(axis=0)
    # print('freq', samplefreq)
    # print('bias', bias)
    magscale = saved_bias[-3:]
    imuac, imugy, imumag = __preprocess_imuraw_wbias(r, bias, magscale, True)
    # rq = get_reference_axis(imuac[:100], imumg[:100])
    # ahrs = NewAHRS(samplefreq, rq, mangle)
    # imuq = ahrs.batch_calc(imuac, imugy, imumg)

    if return_all:
        # print("Return IMU data")
        return imuac, imugy, imumag

    return imuac, imugy, imumag


def imu_temp_load(fpath):
    min_flag = 0
    onehz_to_100hz_temp_list = []
    print(fpath)
    with open(fpath, 'rb') as f:
        barr = f.read()
        # print('total bytes', len(barr))
        # bytes_object = bytes.fromhex(hex_name)
        # ascii_name = bytes_object.decode("ASCII")

        print(len(barr))

        # print(barr)
        print(int(barr[:1].hex(), 16))
        print(barr[1:2])
        print(barr[2:3])
        print(barr[3:4])
        print(int(barr[999:1000].hex(), 16))

        for i in range(0, len(barr)):
            for j in range(0, 100):
                onehz_to_100hz_temp_list.append(int(barr[i:i + 1].hex(), 16))

        onehz_to_100hz_temp_list = [x + (-50) for x in onehz_to_100hz_temp_list]

    return onehz_to_100hz_temp_list, len(onehz_to_100hz_temp_list)


def drawing_xyz_accel(x, y, z, time):
    # print("what?", time)
    plot.plot(time, x, "-r", label="X")
    plot.plot(time, y, "-g", label="Y")
    plot.plot(time, z, "-b", label="Z")
    plot.title('Accel')

    # Give x axis label for the sine wave plot
    plot.xlabel('frame')

    # Give y axis label for the sine wave plot
    plot.ylabel('mg')
    plot.ylim(-50, 50)
    plot.grid(True, which='both')

    plot.axhline(y=0, color='k')
    plot.legend(frameon=False)
    plot.show()


def erase_cell_nand_flash(port):
    hex_erase_buf = bytes.fromhex(SYSCOMMAND_ERASE_NAND_FLASH)
    try:
        ser = serial.Serial(port[0], BAUDRATE)
        ser.write(hex_erase_buf)
        in_bin = ser.read(SYSCOMMAND_ERASE_NAND_FLASH_RESP_SIZE)
        print("Data erase done..!")
    except:
        print("Not open cell serial com port.")
        pass


if __name__ == '__main__':
    asd, len = imu_temp_load("./cell data/gyro_self_test/cell_imu_data_test_tt.txt")
    print(asd)

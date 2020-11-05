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
CELL_GPS_IMU_READ_CHUCK_SIZE = 2048
SYSCOMMAND_ERASE_NAND_FLASH = "AC C0 01 15 78"
SYSCOMMAND_ERASE_NAND_FLASH_RESP_SIZE = 7
IMU_ERR_STR = "IMUERR"
file_path = "./cell data"
frame_to_time_scaler = 1 / 6000


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
    # print(port)
    imu_data_chuck_validation_cnt = 0
    cell_read_error_serial = []
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
                        imu_data_reading_end_flag = 1
                        imu_error_str_delete_flag = 0
                    elif imu_data_chuck_validation_cnt > imu_page_size * 0.95:
                        imu_data_reading_end_flag = 0
                    else:
                        imu_data_reading_end_flag = 0
                        cell_read_error_serial.append(filename)
                if imu_error_str_delete_flag == 0:
                    f.write(data)
                    imu_error_str_delete_flag = 0
                imu_data_chuck_validation_cnt += 1
        return True
    except:
        # print("Not open cell serial com port.")
        return False
        pass


def imuread_firmwarev2(fpath):
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
        # print("Data erase done..!")
    except:
        # print("Not open cell serial com port.")
        pass


if __name__ == '__main__':
    # cell_com_port_name = get_cell_com_port(1155)
    # # print(cell_com_port_name)
    # read_and_save_imu_data(cell_com_port_name, file_path, "cell_imu_data_test", 4000)
    # erase_cell_nand_flash(cell_com_port_name)

    imuaccel_list = loadv2('./cell data/cell_imu_data_test.im', True)
    length = len(imuaccel_list[:, 0])
    time = np.arange(0, length, 1)
    # print(imuaccel_list)
    # print(imuaccel_list[:, 0])
    sum_xyz = imuaccel_list[:, 0] + imuaccel_list[:, 1] + imuaccel_list[:, 2]
    # print(sum_xyz)

    drawing_xyz_accel(imuaccel_list[:, 0], imuaccel_list[:, 1], imuaccel_list[:, 2], time)
    # drawing_xyz_accel(sum_xyz, sum_xyz, sum_xyz, time)

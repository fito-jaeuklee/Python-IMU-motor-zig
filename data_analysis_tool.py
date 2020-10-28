# accel / gyro
# 1. RMSE / MSE
# 2. Linear regression
# 3. correlation

# Magnetometer
# magnetic frequency depend on Motor RPM

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import one_cell_imu_data_extraction

fs = 100


def stft(sample):
    N = len(sample)
    amp = 2 * np.sqrt(2)
    print("amp", amp)
    noise_power = 0.01 * fs / 2
    time = np.arange(N) / float(fs)
    mod = 500 * np.cos(2 * np.pi * 0.25 * time)
    carrier = amp * np.sin(2 * np.pi * 3e3 * time + mod)
    noise = np.random.normal(scale=np.sqrt(noise_power),
                             size=time.shape)
    noise *= np.exp(-time / 5)
    x = carrier + noise

    f, t, Zxx = signal.stft(sample, 100)
    plt.subplot(211)
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=10, shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')



imuaccel_list, imugyro_list, imumag_list = one_cell_imu_data_extraction.loadv2("./cell data/cell_imu_data_test (1).im", True)
plt.subplot(212)
frame_length = np.arange(0, len(imumag_list[:, 0]), 1)
plt.plot(frame_length, imumag_list[:, 0], label="x")
plt.subplot(212)
stft(imumag_list[:, 0])
plt.show()

"""
This reads the newer sensors at a rate of 400 Hz using the
asyncio and pyserial asyncio modules instead of pyserial, queue, and threading modules
in Python 3.13 / Vizard 8
"""

import asyncio

import serial
import serial.tools.list_ports
import serial_asyncio

import numpy as np

import struct
import time
import datetime
from time import sleep
import sys
import collections
import os
import csv
import gzip
import ctypes

# SciPy
from scipy.interpolate import PchipInterpolator, CubicHermiteSpline, UnivariateSpline
from scipy.signal import correlate, correlation_lags, butter, filtfilt, find_peaks
from scipy.spatial.transform import Rotation
from scipy.linalg import block_diag
from sklearn.linear_model import LinearRegression

# import viz
# import viztask

####################################################       Define Basic Functions First   ############################

def quit_and_disconnect():
	# if (EndofPlay_Flag == False) and (Play_Flag == True) and (Reset_Flag == False) and (EndofReset_Flag == True):
		# imu_reset()
		# imu_disconnect()
	# viz.quit()
	sys.exit()


def get_ports():

    ports = serial.tools.list_ports.comports()
    return ports


def findArduino(portsFound):

    commPort = 'None'
    numConnection = len(portsFound)

    for i in range(0,numConnection):
        port = foundPorts[i]
        strPort = str(port)

        # look in Device Manager list to see the names of devices connected to USB ports
        # use the first word in the name of the device that you are looking for
        if 'USB' in strPort:
            splitPort = strPort.split(' ')
            commPort = (splitPort[0])

    return commPort

def micros():
    # return a timestamp in microseconds (us)
    tics = ctypes.c_int64() #use *signed* 64-bit variables; see the "QuadPart" variable here: https://msdn.microsoft.com/en-us/library/windows/desktop/aa383713(v=vs.85).aspx
    freq = ctypes.c_int64()

    #get ticks on the internal ~2MHz QPC clock
    ctypes.windll.Kernel32.QueryPerformanceCounter(ctypes.byref(tics))
    #get the actual freq. of the internal ~2MHz QPC clock
    ctypes.windll.Kernel32.QueryPerformanceFrequency(ctypes.byref(freq))

    t_us = tics.value*1e6/freq.value
    return t_us

def millis():
    # return a timestamp in milliseconds (ms)
    tics = ctypes.c_int64() #use *signed* 64-bit variables; see the "QuadPart" variable here: https://msdn.microsoft.com/en-us/library/windows/desktop/aa383713(v=vs.85).aspx
    freq = ctypes.c_int64()

    #get ticks on the internal ~2MHz QPC clock
    ctypes.windll.Kernel32.QueryPerformanceCounter(ctypes.byref(tics))
    #get the actual freq. of the internal ~2MHz QPC clock
    ctypes.windll.Kernel32.QueryPerformanceFrequency(ctypes.byref(freq))

    t_ms = tics.value*1e3/freq.value
    return t_ms

#-see here for example of constrain function: http://stackoverflow.com/questions/34837677/a-pythonic-way-to-write-a-constrain-function/34837691
def constrain(val, min_val, max_val):
    # "constrain a number to be >= min_val and <= max_val"
    if (val < min_val):
        val = min_val
    elif (val > max_val):
        val = max_val
    return val

#Other timing functions:
def delay(delay_ms):
    # "delay for delay_ms milliseconds (ms)"
    # constrain the commanded delay time to be within valid C type uint32_t limits
    delay_ms = constrain(delay_ms, 0, (1<<32)-1)
    t_start = millis()
    while ((millis() - t_start)%(1<<32) < delay_ms): #use modulus to force C uint32_t-like underflow behavior
        pass #do nothing
    return

def delayMicroseconds(delay_us):
    # "delay for delay_us microseconds (us)"
    # constrain the commanded delay time to be within valid C type uint32_t limits
    delay_us = constrain(delay_us, 0, (1<<32)-1)
    t_start = micros()
    #   %  Modulus	Divides left hand operand by right hand operand and returns remainder
    while ((micros() - t_start)%(1<<32) < delay_us): #use modulus to force C uint32_t-like underflow behavior
        pass #do nothing
    return

##########################      End of Basic Functions      ##################################



# Initialize Vizard
# viz.go()

# Global variables
global count
n_data_pts = 2400  # Number of data points to collect

idx = 0
count = 0
resync_count = 0
start_time = None

foundPorts = None
connectPort = None

payload_size = 30

# Arrays to store data
imu_meas_time = np.zeros(n_data_pts)
system_time = np.zeros(n_data_pts)
tx_time_stamp = np.zeros(n_data_pts)
accel_x = np.zeros(n_data_pts)
accel_y = np.zeros(n_data_pts)
accel_z = np.zeros(n_data_pts)
omega_x = np.zeros(n_data_pts)
omega_y = np.zeros(n_data_pts)
omega_z = np.zeros(n_data_pts)
accel_x2 = np.zeros(n_data_pts)
accel_y2 = np.zeros(n_data_pts)
accel_z2 = np.zeros(n_data_pts)
accel_hf_1 = np.zeros(n_data_pts)
accel_hf_2 = np.zeros(n_data_pts)
accel_hf_3 = np.zeros(n_data_pts)
status_var_value = np.zeros(n_data_pts)


def is_valid_frame(frame):
    # Check if the frame is valid (status_var = 4095).
    global payload_size
    try:
        status_var = struct.unpack(">H", frame[payload_size - 2:payload_size])[0]
        return status_var == 4095
    except struct.error:
        return False


class SerialReaderProtocol(asyncio.Protocol):
    def connection_made(self, transport):
        self.transport = transport
        self.buffer = b''
        print('Connection established.')
        # Send any initialization commands to the microcontroller if necessary


    def data_received(self, data):
        global count, start_time, resync_count, payload_size, idx

        self.buffer += data

        while len(self.buffer) >= payload_size and count < n_data_pts:
            frame = self.buffer[:payload_size]
            self.buffer = self.buffer[payload_size:]

            # If the data frame is not valid then the "continue" statement is executed and the while loop starts
            # over from the top instead of processing the data frame
            if not is_valid_frame(frame):
                resync_count += 1
                self.buffer = b""  # Clear buffer and wait for the next valid frame
                print(f"Resync triggered, count: {resync_count}")
                continue

            # Process the frame
            self.process_frame(frame)

            if count == 0:
                print('SWING NOW!!!')
                start_time = time.perf_counter()
            # else:
            #     print('data')

            count += 1

            if count >= n_data_pts:
                self.transport.close()
                asyncio.get_event_loop().stop()
                break

    def process_frame(self, frame):
        global imu_meas_time, imu_meas_time_hf, imu_collection_time, system_time, payload_size
        global tx_time_stamp, accel_x, accel_y, accel_z, accel_hf_1, accel_hf_2, accel_hf_3
        global omega_x, omega_y, omega_z, accel_x2, accel_y2, accel_z2, status_var_value, count, idx

        # Get system time
        frame_current_time = time.perf_counter()
        system_time[idx] = frame_current_time - start_time if start_time else 0.0

        # Unpack the data
        integers = []

        integers.append(struct.unpack(">L", frame[0:4])[0])  # 4-byte unsigned int

        for i in range(4, 22, 2):
            integers.append(struct.unpack(">h", frame[i:i+2])[0])  # 2-byte signed int

        for i in range(22, payload_size, 2):
            integers.append(struct.unpack(">H", frame[i:i+2])[0])  # 2-byte unsigned int

        idx = count


        # Map the data
        tx_time_stamp[idx] = integers[0]
        accel_x[idx] = -0.009576806641 * integers[1]
        accel_y[idx] = 0.009576806641 * integers[2]
        accel_z[idx] = -0.009576806641 * integers[3]
        omega_x[idx] = -0.002130528872 * integers[4]
        omega_y[idx] = 0.002130528872 * integers[5]
        omega_z[idx] = -0.002130528872 * integers[6]

        accel_x2[idx] = 0.4788403320 * integers[8]
        accel_y2[idx] = -0.4788403320 * integers[7]
        accel_z2[idx] = 0.4788403320 * integers[9]

        accel_hf_1[idx] = 0.4788403320 * integers[10]
        accel_hf_2[idx] = 0.4788403320 * integers[11]
        accel_hf_3[idx] = 0.4788403320 * integers[12]

        status_var_value[idx] = integers[13]

        if idx == 0:
            imu_meas_time[idx] = 0.0
        else:
            delta_tx = (tx_time_stamp[idx] - tx_time_stamp[0]) if tx_time_stamp[idx] >= tx_time_stamp[0] else \
                (tx_time_stamp[idx] + 4294967295 - tx_time_stamp[0])
            imu_meas_time[idx] = float(delta_tx / 1e6)

async def main():

    global foundPorts, connectPort

    # initialize the com port for data collection
    foundPorts = get_ports()
    connectPort = findArduino(foundPorts)
    print(foundPorts)
    print(connectPort)

    # Replace 'COM_PORT' with your actual serial port
    com_port = 'COM10'  # e.g., 'COM10' on Windows or '/dev/ttyUSB0' on Linux
    baud_rate = 921600

    loop = asyncio.get_running_loop()

    # Create serial connection
    transport, protocol = await serial_asyncio.create_serial_connection(
        loop, SerialReaderProtocol, connectPort, baudrate=baud_rate
    )

    """
    if connectPort != 'None':

        if hasattr(transport, "_serial"):
            print("Buffers will be flushed and reset.")
        else:
            print("Transport does not expose the _serial attribute.")
            quit_and_disconnect()

    else:
        print('Connection Issue!')
        quit_and_disconnect()


    # Flush and reset buffers
    serial_connection = transport._serial
    serial_connection.reset_input_buffer()
    serial_connection.reset_output_buffer()
    serial_connection.flush()
    """

    if connectPort == 'None':
        print('Connection Issue!')
        quit_and_disconnect()


    try:
        await asyncio.sleep(1e6)  # Run indefinitely until data collection is complete - https://superfastpython.com/asyncio-sleep/
    except asyncio.CancelledError:
        pass

    # After data collection, save data to CSV
    current_time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f'../reports/ICM45686_and_ADXL375_test_data_{current_time_str}.csv'

    np.savetxt(file_name, np.c_[
        tx_time_stamp[:count], imu_meas_time[:count], system_time[:count],
        status_var_value[:count],
        accel_x[:count], accel_y[:count], accel_z[:count],
        omega_x[:count], omega_y[:count], omega_z[:count],
        accel_x2[:count], accel_y2[:count], accel_z2[:count],
        accel_hf_1[:count], accel_hf_2[:count], accel_hf_3[:count]],
        delimiter=',',
        header='tx_time_stamp,imu_meas_time,system_time,status_var_value,accel_x,accel_y,accel_z,omega_x,omega_y,omega_z,accel_x2,accel_y2,accel_z2,accel_hf_1,accel_hf_2,accel_hf_3',
        comments='')

    imu_idx = np.arange(count)

    imu_map = {
        'accel_x':accel_x,'accel_y':accel_y,'accel_z':accel_z,
        'omega_x':omega_x,'omega_y':omega_y,'omega_z':omega_z,
        'accel_x2':accel_x2,'accel_y2':accel_y2,'accel_z2':accel_z2
    }

    print('Data collection is complete.')
# Run the asyncio event loop
# viztask.schedule(asyncio.run(main()))

asyncio.run(main())

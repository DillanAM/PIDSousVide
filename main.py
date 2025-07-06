import asyncio
import time as time
import csv
import numpy as np
import scipy
import math
from bleak import BleakClient
import matplotlib.pyplot as plt

address = "C2:71:1E:85:37:40"
UUID = "00000101-CAAB-3792-3D44-97AE51C1407A"


def map_range(x, in_max, out_max):
    return x * out_max / in_max


def cooling_fit(x, k):
    return 20.0 + (70.5 - 20.0)*np.exp(-k*x)


def plotter(x, y, length_in_seconds):

    plt.plot(x, y)
    plt.xlim([0, length_in_seconds])
    plt.ylim([20, max(y) + 10])
    plt.xlabel('Time in Seconds')
    plt.ylabel('Temperature in Celsius')
    plt.show()


def exporter(x, y, file_name):

    with open(str(file_name), 'w', newline='') as csvfile:
        for i in range(len(x)):
            csvfile.write(str(x[i]))
            csvfile.write('\n')
        csvfile.write('\n')
        for i in range(len(y)):
            csvfile.write(str(y[i]))
            csvfile.write('\n')


async def data_logger(thermo_object, length_in_seconds, address):

    temp_list = []
    time_list = []

    start_time = time.time()
    end_time = time.time()+length_in_seconds


    while time.time() < end_time:

        try:
            completion = round(map_range(time.time()-start_time, length_in_seconds, 100), 1)
            print(f'{completion}%')

            probe_status = await thermo_object.read_gatt_char(UUID)
            temp = temperature_read(probe_status)
            seconds = time.time()-start_time

            temp_list.append(round(temp[0].item(), 1))
            time_list.append(round(seconds, 1))

            await asyncio.sleep(1)

        except KeyboardInterrupt:
            break

        except:
            thermo_object = await reconnect_routine(address)

    return temp_list, time_list


def mode_read(probe_status_array):

    mode_byte = int.from_bytes(probe_status_array[21:22], byteorder='little')
    mode_bits = f'{mode_byte:02b}'

    mode = 'null'
    if mode_bits == '00':
        mode = 'Normal Mode'
    elif mode_bits == '01':
        mode = 'Instant Read Mode'
    elif mode_bits == '11':
        mode = 'Error'

    return mode


def temperature_read(probe_status_array):

    temperature_pack = int.from_bytes(probe_status_array[8:21], byteorder='little')
    temp_bits = f'{temperature_pack:0104b}'

    temps = np.zeros(8)
    for i in range(8):
        temps[-i-1] = int(temp_bits[13*(i):13*(i+1)], 2) * 0.05 - 20

    return temps


async def connect_routine(address):

    thermo = BleakClient(address, timeout=20.0)
    print('Connecting...')
    await thermo.connect()
    print('Connected!')

    return thermo


async def reconnect_routine(address):

    thermo = BleakClient(address, timeout=20.0)
    print('Reconnecting...')
    await thermo.connect()

    return thermo


async def main(address):

    thermo = await connect_routine(address)

    length_in_seconds = 7200
    temp, time = await data_logger(thermo, length_in_seconds, address)
    exporter(time, temp, 'Radiator Chiller Wax      ')
    plotter(time, temp, length_in_seconds)

asyncio.run(main(address))

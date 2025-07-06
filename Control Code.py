import asyncio
import time as time
import csv

import bleak.exc
import numpy as np
import scipy
import math
from bleak import BleakClient
import matplotlib.pyplot as plt

cooker_address = '94:A9:A8:19:77:5F'
thermoprobe_address = "C2:71:23:E2:CF:E0"


def exporter(time_s, core_temp, amb_temp, state, PID_gains, file_name):

    with open(str(file_name), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(time_s)
        writer.writerow(core_temp)
        writer.writerow(amb_temp)
        writer.writerow(state)
        writer.writerow(PID_gains)



class PIDController:
    def __init__(self, kp, ki, kd, target_temperature):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target_temperature = target_temperature
        self.integral = 0
        self.previous_error = 0

    def calculate(self, current_temperature, dt):
        # Calculate error
        error = self.target_temperature - current_temperature

        # Proportional term
        p = self.kp * error

        # Integral term
        self.integral += error * dt
        i = self.ki * self.integral

        # Derivative term
        d = self.kd * (error - self.previous_error) / dt
        self.previous_error = error

        # PID output
        return p + i + d


class thermoprobe:

    UUID = '00000101-CAAB-3792-3D44-97AE51C1407A'

    def __init__(self, MAC_Address):
        self.MAC_Address = MAC_Address
        self.BLE_Object = None

    async def connect_routine(self):
        if self.BLE_Object:
            await self.BLE_Object.disconnect()
        print('Connecting to Thermoprobe...')
        self.BLE_Object = BleakClient(self.MAC_Address, timeout=20.0)
        await self.BLE_Object.connect()
        print('Connected!')

    async def connection_status(self):
        if self.BLE_Object:
            return self.BLE_Object.is_connected
        return False

    async def disconnect_routine(self):
        if self.BLE_Object.is_connected and self.BLE_Object:
            await self.BLE_Object.disconnect()


    async def mode(self):

        probe_read = await self.BLE_Object.read_gatt_char(self.UUID)

        mode_byte = int.from_bytes(probe_read[21:22], byteorder='little')
        mode_bits = f'{mode_byte:02b}'

        if mode_bits == '00':
            mode = 'Normal Mode'
        elif mode_bits == '01':
            mode = 'Instant Read Mode'
        else:
            mode = 'Error'

        return mode

    async def temperature_read(self):

        probe_read = await self.BLE_Object.read_gatt_char(self.UUID)

        virtual_sensor_byte = int.from_bytes(probe_read[22:23], byteorder='little')
        virtual_sensor_bits = f'{virtual_sensor_byte:08b}'

        core_ID = int(virtual_sensor_bits[4:7], 2)
        surface_ID = int(virtual_sensor_bits[2:4], 2) + 3
        ambient_ID = int(virtual_sensor_bits[0:2], 2) + 4

        temperature_byte = int.from_bytes(probe_read[8:21], byteorder='little')
        temperature_bits = f'{temperature_byte:0104b}'
        temps = np.zeros(8)
        for i in range(8):
            temps[-i - 1] = int(temperature_bits[13 * (i):13 * (i + 1)], 2) * 0.05 - 20

        print(f'Identified Core:{core_ID+1:} at Temperature: {temps[core_ID]:.2f} C. Water Temperature: {temps[ambient_ID]:.2f} C')

        return [temps[core_ID].item(), temps[surface_ID].item(), temps[ambient_ID].item()]


class cooker:

    UUID = '0000ffe1-0000-1000-8000-00805f9b34fb'

    def __init__(self, MAC_Address):
        self.MAC_Address = MAC_Address
        self.BLE_Object = None

    async def connect_routine(self):
        if self.BLE_Object:
            await self.BLE_Object.disconnect()
        print('Connecting to Cooker...')
        self.BLE_Object = BleakClient(self.MAC_Address, timeout=20.0)
        await self.BLE_Object.connect()
        print('Connected!')

    async def connection_status(self):
        if self.BLE_Object:
            return self.BLE_Object.is_connected
        return False

    async def disconnect_routine(self):
        if self.BLE_Object.is_connected and self.BLE_Object:
            await self.BLE_Object.disconnect()

    async def manual_standby(self):
        await self.BLE_Object.write_gatt_char(self.UUID, bytes('0', 'utf-8'))

    async def manual_dwelling(self):
        await self.BLE_Object.write_gatt_char(self.UUID, bytes('3', 'utf-8'))

    async def manual_heating(self):

        while True:
            response = input('Turn element on or off: (ON, OFF, END)\n')
            if response == 'ON':
                await self.BLE_Object.write_gatt_char(self.UUID, bytes('1', 'utf-8'))  # 1 is IO Heating Mode
            if response == 'OFF':
                await self.BLE_Object.write_gatt_char(self.UUID, bytes('3', 'utf-8'))  # 3 is IO Dwelling Mode
            if response == 'END':
                break

    async def manual_cooling(self):

        while True:
            response = input('Turn cooler on or off: (ON, OFF, END)\n')
            if response == 'ON':
                await self.BLE_Object.write_gatt_char(self.UUID, bytes('2', 'utf-8'))  # 2 is IO Cooling Mode
            if response == 'OFF':
                await self.BLE_Object.write_gatt_char(self.UUID, bytes('3', 'utf-8'))  # 3 is IO Dwelling Mode
            if response == 'END':
                break

    async def normal_control(self, current_temperature, pid, cycle_time, initial_time):

        dt = time.time() - initial_time
        pid_output = pid.calculate(current_temperature[2], dt=dt)
        initial_time = time.time()
        state = 0

        # Define thresholds for heating and cooling
        if pid_output > 0.5:  # Turn heating on
            await self.BLE_Object.write_gatt_char(self.UUID, bytes('1', 'utf-8'))  # 1 is IO Heating Mode
            heating = True
            cooling = False
        else:  # Keep both off
            await self.BLE_Object.write_gatt_char(self.UUID, bytes('3', 'utf-8'))  # 3 is IO Dwelling Mode
            heating = False
            cooling = False

        # Print the states for monitoring
        print(f"Heating: {heating}, Cooling: {cooling}, PID Output: {pid_output:.2f}")

        if heating:
            state = 1
        if cooling:
            state = -1

        await asyncio.sleep(cycle_time)

        return initial_time, state

    async def automatic_control(self, current_temperature, pid, cycle_time, initial_time):

        dt = time.time() - initial_time
        pid_output = pid.calculate(current_temperature[0], dt=dt)
        initial_time = time.time()
        state = 0

        # Define thresholds for heating and cooling
        if pid_output > 5 and current_temperature[2] <= 80:  # Turn heating on
            await self.BLE_Object.write_gatt_char(self.UUID, bytes('1', 'utf-8'))  # 1 is IO Heating Mode
            heating = True
            cooling = False
        elif pid_output < -5:  # Turn cooling on
            await self.BLE_Object.write_gatt_char(self.UUID, bytes('2', 'utf-8'))  # 2 is IO Cooling Mode
            heating = False
            cooling = True
        else:  # Keep both off
            await self.BLE_Object.write_gatt_char(self.UUID, bytes('3', 'utf-8'))  # 3 is IO Dwelling Mode
            heating = False
            cooling = False

        # Print the states for monitoring
        print(f"Heating: {heating}, Cooling: {cooling}, PID Output: {pid_output:.2f}")

        if heating:
            state = 1
        if cooling:
            state = -1

        await asyncio.sleep(cycle_time)

        return initial_time, state


async def main(cooker_address, thermoprobe_address):

    sous_vide = cooker(cooker_address)
    thermometer = thermoprobe(thermoprobe_address)

    try:
        await sous_vide.connect_routine()
        await thermometer.connect_routine()

        while True:

            await sous_vide.manual_standby()

            mode = input('Mode: (Manual, Auto, Normal, END)\n')

            if mode == 'Normal':

                temp_target = float(input('Target Temperature in Celsius\n'))
                kp, ki, kd = 1, 0, 0
                pid = PIDController(kp=kp, ki=ki, kd=kd, target_temperature=temp_target)
                initial_time = time.time()
                start_time = initial_time

                filename = 'PID Test 7'
                core_temp_log = []
                amb_temp_log = []
                time_log = []
                state_log = []

                while True:

                    try:
                        current_temperature = await thermometer.temperature_read()

                        core_temp_log.append(current_temperature[0])
                        amb_temp_log.append(current_temperature[2])
                        time_log.append(time.time() - start_time)

                        initial_time, state = await sous_vide.normal_control(
                            current_temperature=current_temperature,
                            pid=pid, cycle_time=15.0, initial_time=initial_time)

                        state_log.append(state)

                        if current_temperature[0] >= temp_target:
                            break

                    except KeyboardInterrupt:
                        break

                    except:
                        await sous_vide.manual_dwelling()
                        print('AUTO ERROR')
                        while not await thermometer.connection_status():
                            try:
                                await thermometer.connect_routine()
                            except bleak.exc.BleakError:
                                print('Reconnect Failed. Trying Again...')
                                await asyncio.sleep(10)
                        await asyncio.sleep(5)
                        continue

                    finally:
                        exporter(time_log, core_temp_log, amb_temp_log, state_log, [kp, ki, kd],
                                 filename)

                print('Cooking Complete')
                mode = 'END'
                break

            if mode == 'Manual':
                while True:

                    await sous_vide.manual_dwelling()

                    manual_control_type = input('Type: Heating, Cooling , END?\n')

                    if manual_control_type == 'Heating':
                        await sous_vide.manual_heating()

                    if manual_control_type == 'Cooling':
                        await sous_vide.manual_cooling()

                    if manual_control_type == 'END':
                        break

            if mode == 'Auto':

                temp_target = float(input('Target Temperature in Celsius\n'))
                kp, ki, kd = 1.175, 0, 650
                pid = PIDController(kp=kp, ki=ki, kd=kd, target_temperature=temp_target)
                initial_time = time.time()
                start_time = initial_time
                completion_latch = False

                filename = 'PID Test Pork'
                core_temp_log = []
                amb_temp_log = []
                time_log = []
                state_log = []

                while True:

                    try:
                        current_temperature = await thermometer.temperature_read()

                        core_temp_log.append(current_temperature[0])
                        amb_temp_log.append(current_temperature[2])
                        time_log.append(time.time()-start_time)

                        initial_time, state = await sous_vide.automatic_control(current_temperature=current_temperature,
                                                                         pid=pid, cycle_time=30.0, initial_time=initial_time)

                        state_log.append(state)

                        if time.time()-start_time >= 5400:
                            break

                        if current_temperature[0] >= temp_target:
                            completion_latch = True

                        if completion_latch and current_temperature[0] <= temp_target:
                            break

                    except KeyboardInterrupt:
                        break

                    except:
                        await sous_vide.manual_dwelling()
                        print('AUTO ERROR')
                        while not await thermometer.connection_status():
                            try:
                                await thermometer.connect_routine()
                            except bleak.exc.BleakError:
                                print('Reconnect Failed. Trying Again...')
                                await asyncio.sleep(10)
                        await asyncio.sleep(5)
                        continue
                    finally:
                        exporter(time_log, core_temp_log, amb_temp_log, state_log, [kp, ki, kd],
                                 filename)

                print('Cooking Complete')
                mode = 'END'
                break

            if mode == 'END':
                break
    finally:
        await sous_vide.manual_standby()
        await sous_vide.disconnect_routine()
        await thermometer.disconnect_routine()


asyncio.run(main(cooker_address, thermoprobe_address))

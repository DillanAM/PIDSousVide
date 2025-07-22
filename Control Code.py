import asyncio, sys, time, numpy as np, qasync, json, os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
    QComboBox, QWidget, QGridLayout, QMessageBox, QFileDialog, QTabWidget, QVBoxLayout
)
from PyQt5.QtCore import Qt, QTimer
from bleak import BleakClient, BleakError
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import csv

GRAPH_INTERVAL = 1000           # ms between redraws
CONTROL_INTERVAL = 2           # s between PID decisions
COOKER_MAC = "94:A9:A8:19:77:5F"
PROBE_MAC = "C2:71:1E:F2:C6:20"
PID_SETTINGS_FILE = "pid_settings.json"

def exporter(time_s, core_temp, surface_temp, water_temp, set_temp, state, file_name):
    """Export step response data to CSV including surface temperatures."""
    with open(str(file_name), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Time (s)", *time_s])
        writer.writerow(["Core Temp (°C)", *core_temp])
        writer.writerow(["Surface Temp (°C)", *surface_temp])
        writer.writerow(["Water Temp (°C)", *water_temp])
        writer.writerow(["Set Temp (°C)", set_temp])
        writer.writerow(["State", *state])

class PIDController:
    def __init__(self, kp, ki, kd, target_temperature):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target_temperature = target_temperature
        self.integral = 0
        self.previous_error = 0

    def calculate(self, current_temperature, dt):
        error = self.target_temperature - current_temperature
        p = self.kp * error
        self.integral += error * dt
        i = self.ki * self.integral
        d = self.kd * (error - self.previous_error) / dt
        self.previous_error = error
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
        if self.BLE_Object and self.BLE_Object.is_connected:
            await self.BLE_Object.disconnect()

    async def temperature_read(self):
        while not await self.connection_status():
            print("Lost connection to probe. Reconnecting...")
            try:
                await self.connect_routine()
            except Exception as e:
                print(f"Reconnection attempt failed: {e}")
                await asyncio.sleep(2)

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
        if self.BLE_Object and self.BLE_Object.is_connected:
            await self.BLE_Object.disconnect()

    async def standby(self):
        await self.BLE_Object.write_gatt_char(self.UUID, b'0')

    async def dwell(self):
        await self.BLE_Object.write_gatt_char(self.UUID, b'3')

    async def heat(self):
        await self.BLE_Object.write_gatt_char(self.UUID, b'1')

    async def cool(self):
        await self.BLE_Object.write_gatt_char(self.UUID, b'2')

class MatplotCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots(figsize=(6,3), tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax.set_xlabel("Time [min]")
        self.ax.set_ylabel("T [°C]")
        self.lines = {
            "core":    self.ax.plot([], [], 'r-', label="Core")[0],
            "surface": self.ax.plot([], [], 'm-', label="Surface")[0],
            "water":   self.ax.plot([], [], 'b-', label="Water")[0],
            "set":     self.ax.plot([], [], 'k--', label="Set‑point")[0],
            "foptd":   self.ax.plot([], [], 'g--', label="FOPTD Fit")[0],
        }
        self.ax.legend()

    def plot_data(self, t, core, surface, water, setpoint):
        """Update graph with new measurements."""
        for k, y in [
            ("core", core),
            ("surface", surface),
            ("water", water),
            ("set", np.full_like(core, setpoint)),
        ]:
            self.lines[k].set_data(t / 60, y)
        self.ax.relim()
        self.ax.autoscale_view()
        self.draw_idle()

    def fit_plot(self, t, core, surface, water, setpoint, foptd, *params):
        """Plot the FOPTD fit alongside measured data."""
        for k, y in [
            ("core", core),
            ("surface", surface),
            ("water", water),
            ("set", np.full_like(core, setpoint)),
            ("foptd", foptd(t, *params)),
        ]:
            self.lines[k].set_data(t / 60, y)
        self.ax.relim()
        self.ax.autoscale_view()
        self.draw_idle()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sous‑Vide Control GUI")
        self.resize(880, 500)

        self.tabs = QTabWidget()
        self.control_tab = QWidget()
        self.tuning_tab = QWidget()

        self.tabs.addTab(self.control_tab, "Control")
        self.tabs.addTab(self.tuning_tab, "PID Tuning")

        self.setup_control_tab()
        self.setup_tuning_tab()

        self.setCentralWidget(self.tabs)

        # ---------- state ----------
        self.thermo = thermoprobe(PROBE_MAC)
        self.cooker = cooker(COOKER_MAC)
        self.task_loop = None
        self.t, self.core, self.surface, self.water = [], [], [], []
        self.running = False
        self.pid_core_surface = None
        self.pid_surface_water = None
        self.pid_water = None

        self.last_time = time.time()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.redraw)
        self.timer.start(CONTROL_INTERVAL*1000)

    def setup_control_tab(self):
        self.btn_probe = QPushButton("Connect Probe")
        self.btn_cooker = QPushButton("Connect Cooker")
        self.lbl_probe = QLabel("Probe: ⬤ Disconnected")
        self.lbl_cooker = QLabel("Cooker: ⬤ Disconnected")

        self.modeBox = QComboBox()
        self.modeBox.addItems(["Manual", "Normal", "PID Auto"])
        self.modeBox.currentTextChanged.connect(self.mode_changed)

        self.sp_set = QSpinBox(); self.sp_set.setRange(20, 95); self.sp_set.setValue(55)


        # Cascade PID spin boxes
        self.kpCoreSurfBox = QDoubleSpinBox(); self.kpCoreSurfBox.setRange(0, 100); self.kpCoreSurfBox.setDecimals(5)
        self.kiCoreSurfBox = QDoubleSpinBox(); self.kiCoreSurfBox.setRange(0, 100); self.kiCoreSurfBox.setDecimals(5)
        self.kdCoreSurfBox = QDoubleSpinBox(); self.kdCoreSurfBox.setRange(0, 2000); self.kdCoreSurfBox.setDecimals(5)

        self.kpSurfWaterBox = QDoubleSpinBox(); self.kpSurfWaterBox.setRange(0, 100); self.kpSurfWaterBox.setDecimals(5)
        self.kiSurfWaterBox = QDoubleSpinBox(); self.kiSurfWaterBox.setRange(0, 100); self.kiSurfWaterBox.setDecimals(5)
        self.kdSurfWaterBox = QDoubleSpinBox(); self.kdSurfWaterBox.setRange(0, 2000); self.kdSurfWaterBox.setDecimals(5)

        self.kpWaterBox = QDoubleSpinBox(); self.kpWaterBox.setRange(0, 100); self.kpWaterBox.setDecimals(5)
        self.kiWaterBox = QDoubleSpinBox(); self.kiWaterBox.setRange(0, 100); self.kiWaterBox.setDecimals(5)
        self.kdWaterBox = QDoubleSpinBox(); self.kdWaterBox.setRange(0, 2000); self.kdWaterBox.setDecimals(5)

        self.lbl_core = QLabel("Core: --.- °C")
        self.lbl_water = QLabel("Water: --.- °C")
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_heat = QPushButton("Heat")
        self.btn_cool = QPushButton("Cool")
        self.btn_dwell = QPushButton("Dwell")
        self.btn_standby = QPushButton("Standby")
        self.btn_export = QPushButton("Export Data")

        self.graph = MatplotCanvas(self)

        g = QGridLayout()
        g.addWidget(self.btn_probe, 0, 0);
        g.addWidget(self.lbl_probe, 0, 1)
        g.addWidget(self.btn_cooker, 1, 0);
        g.addWidget(self.lbl_cooker, 1, 1)
        g.addWidget(QLabel("Mode:"), 2, 0);
        g.addWidget(self.modeBox, 2, 1)
        g.addWidget(QLabel("Set‑point °C"), 3, 0);
        g.addWidget(self.sp_set, 3, 1)

        g.addWidget(QLabel("Core→Surface PID"), 4, 0, 1, 2)
        g.addWidget(QLabel("Kp"), 5, 0); g.addWidget(self.kpCoreSurfBox, 5, 1)
        g.addWidget(QLabel("Ki"), 6, 0); g.addWidget(self.kiCoreSurfBox, 6, 1)
        g.addWidget(QLabel("Kd"), 7, 0); g.addWidget(self.kdCoreSurfBox, 7, 1)

        g.addWidget(QLabel("Surface→Water PID"), 8, 0, 1, 2)
        g.addWidget(QLabel("Kp"), 9, 0); g.addWidget(self.kpSurfWaterBox, 9, 1)
        g.addWidget(QLabel("Ki"), 10, 0); g.addWidget(self.kiSurfWaterBox, 10, 1)
        g.addWidget(QLabel("Kd"), 11, 0); g.addWidget(self.kdSurfWaterBox, 11, 1)

        g.addWidget(QLabel("Water PID"), 12, 0, 1, 2)
        g.addWidget(QLabel("Kp"), 13, 0); g.addWidget(self.kpWaterBox, 13, 1)
        g.addWidget(QLabel("Ki"), 14, 0); g.addWidget(self.kiWaterBox, 14, 1)
        g.addWidget(QLabel("Kd"), 15, 0); g.addWidget(self.kdWaterBox, 15, 1)

        g.addWidget(self.lbl_core, 16, 0, 1, 2); g.addWidget(self.lbl_water, 17, 0, 1, 2)
        g.addWidget(self.btn_start, 18, 0); g.addWidget(self.btn_stop, 18, 1)
        g.addWidget(self.btn_heat, 19, 0); g.addWidget(self.btn_cool, 19, 1)
        g.addWidget(self.btn_dwell, 20, 0); g.addWidget(self.btn_standby, 20, 1)
        g.addWidget(self.btn_export, 21, 0, 1, 2)
        g.addWidget(self.graph, 0, 2, 22, 1)



        # Load PID gains from file if available
        self.load_pid_settings()

        # ---------- signals ----------
        self.btn_probe.clicked.connect(lambda: asyncio.create_task(self.handle_probe()))
        self.btn_cooker.clicked.connect(lambda: asyncio.create_task(self.handle_cooker()))
        self.btn_start.clicked.connect(lambda: asyncio.create_task(self.start_loop()))
        self.btn_stop.clicked.connect(lambda: asyncio.create_task(self.stop_loop()))
        self.btn_heat.clicked.connect(lambda: asyncio.create_task(self.cooker.heat()))
        self.btn_cool.clicked.connect(lambda: asyncio.create_task(self.cooker.cool()))
        self.btn_dwell.clicked.connect(lambda: asyncio.create_task(self.cooker.dwell()))
        self.btn_standby.clicked.connect(lambda: asyncio.create_task(self.cooker.standby()))
        self.btn_export.clicked.connect(lambda: self.export_data())

        container = QWidget()
        container.setLayout(g)
        self.control_tab.setLayout(QVBoxLayout())
        self.control_tab.layout().addWidget(container)

        self.mode_changed(self.modeBox.currentText())
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False)

    def setup_tuning_tab(self):
        layout = QVBoxLayout()
        self.loop_select = QComboBox()
        self.loop_select.addItems(["Core→Surface", "Surface→Water", "Water"])

        self.load_csv_btn = QPushButton("Load Step Response")
        self.result_label = QLabel("PID parameters will appear here.")
        self.plot_canvas = MatplotCanvas(self)

        self.load_csv_btn.clicked.connect(self.load_and_fit_csv)

        layout.addWidget(self.loop_select)
        layout.addWidget(self.load_csv_btn)
        layout.addWidget(self.result_label)
        layout.addWidget(self.plot_canvas)
        self.tuning_tab.setLayout(layout)

    def load_and_fit_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if not file_path:
            return

        try:
            df = pd.read_csv(file_path, index_col=False, encoding="cp1252", header=None)

            t = np.array([float(x) for x in df.iloc[0, 1:].dropna()])
            core = np.array([float(x) for x in df.iloc[1, 1:].dropna()])

            # Datasets exported with older versions may not include surface data
            label = str(df.iloc[2, 0]).lower()
            if "surface" in label:
                surface = np.array([float(x) for x in df.iloc[2, 1:].dropna()])
                water = np.array([float(x) for x in df.iloc[3, 1:].dropna()])
                set_point = float(df.iloc[4, 1])
            else:
                surface = np.array([])
                water = np.array([float(x) for x in df.iloc[2, 1:].dropna()])
                set_point = float(df.iloc[3, 1])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load CSV: {e}")
            return

        def foptd(t, K, tau, L):
            """First-order plus dead time model."""
            T0 = y[0]
            T = np.piecewise(
                t,
                [t < L, t >= L],
                [lambda t: T0, lambda t: T0 + K * (1 - np.exp(-(t - L) / tau))],
            )
            return T

        try:
            loop = self.loop_select.currentText()
            if loop == "Core→Surface" and surface.size:
                y = surface
            else:
                y = water

            params, _ = curve_fit(foptd, t, y, p0=[30, 300, 20])
            K, tau, L = abs(params)
            Kp = 1.2 * tau / (K * L)
            Ti = 2 * L
            Td = 0.5 * L
            Ki = Kp / Ti
            Kd = Kp * Td

            if loop == "Core→Surface":
                self.kpCoreSurfBox.setValue(Kp)
                self.kiCoreSurfBox.setValue(Ki)
                self.kdCoreSurfBox.setValue(Kd)
            elif loop == "Surface→Water":
                self.kpSurfWaterBox.setValue(Kp)
                self.kiSurfWaterBox.setValue(Ki)
                self.kdSurfWaterBox.setValue(Kd)
            elif loop == "Water":
                self.kpWaterBox.setValue(Kp)
                self.kiWaterBox.setValue(Ki)
                self.kdWaterBox.setValue(Kd)

            self.result_label.setText(
                f"FOPTD Fit ({loop}): K={K:.2f}, tau={tau:.2f}, L={L:.2f}\n"
                f"Ziegler-Nichols PID ({loop}): Kp={Kp:.3f}, Ki={Ki:.5f}, Kd={Kd:.3f}"
            )

            if surface.size:
                surf_data = surface
            else:
                # keep array of NaNs for plotting alignment
                surf_data = np.full_like(core, np.nan)

            self.plot_canvas.fit_plot(t, core, surf_data, water, set_point, foptd, *params)
        except Exception as e:
            QMessageBox.critical(self, "Fitting Error", f"Could not fit model: {e}")


    def mode_changed(self, text):
        is_manual = (text == "Manual")
        is_auto = (text == 'PID Auto')
        self.btn_heat.setEnabled(is_manual)
        self.btn_cool.setEnabled(is_manual)
        self.btn_dwell.setEnabled(is_manual)
        self.btn_standby.setEnabled(is_manual)
        self.btn_start.setEnabled(not is_manual)
        self.btn_stop.setEnabled(not is_manual)
        self.kpCoreSurfBox.setEnabled(is_auto)
        self.kiCoreSurfBox.setEnabled(is_auto)
        self.kdCoreSurfBox.setEnabled(is_auto)
        self.kpSurfWaterBox.setEnabled(is_auto)
        self.kiSurfWaterBox.setEnabled(is_auto)
        self.kdSurfWaterBox.setEnabled(is_auto)
        self.kpWaterBox.setEnabled(is_auto)
        self.kiWaterBox.setEnabled(is_auto)
        self.kdWaterBox.setEnabled(is_auto)

    def load_pid_settings(self):
        if os.path.exists(PID_SETTINGS_FILE):
            try:
                with open(PID_SETTINGS_FILE, 'r') as f:
                    data = json.load(f)
                    self.kpCoreSurfBox.setValue(data.get('kpCoreSurf', 0.0))
                    self.kiCoreSurfBox.setValue(data.get('kiCoreSurf', 0.0))
                    self.kdCoreSurfBox.setValue(data.get('kdCoreSurf', 0.0))
                    self.kpSurfWaterBox.setValue(data.get('kpSurfWater', 0.0))
                    self.kiSurfWaterBox.setValue(data.get('kiSurfWater', 0.0))
                    self.kdSurfWaterBox.setValue(data.get('kdSurfWater', 0.0))
                    self.kpWaterBox.setValue(data.get('kpWater', 0.0))
                    self.kiWaterBox.setValue(data.get('kiWater', 0.0))
                    self.kdWaterBox.setValue(data.get('kdWater', 0.0))
            except Exception as e:
                print(f"Failed to load PID settings: {e}")

    def save_pid_settings(self):
        data = {
            'kpCoreSurf': self.kpCoreSurfBox.value(),
            'kiCoreSurf': self.kiCoreSurfBox.value(),
            'kdCoreSurf': self.kdCoreSurfBox.value(),
            'kpSurfWater': self.kpSurfWaterBox.value(),
            'kiSurfWater': self.kiSurfWaterBox.value(),
            'kdSurfWater': self.kdSurfWaterBox.value(),
            'kpWater': self.kpWaterBox.value(),
            'kiWater': self.kiWaterBox.value(),
            'kdWater': self.kdWaterBox.value()
        }
        try:
            with open(PID_SETTINGS_FILE, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Failed to save PID settings: {e}")

    async def handle_probe(self):
        try:
            self.lbl_probe.setText("Probe: ⬤ Connecting...")
            self.btn_probe.setEnabled(False)
            if await self.thermo.connection_status():
                await self.thermo.disconnect_routine()
                self.lbl_probe.setText("Probe: ⬤ Disconnected")
                self.btn_probe.setText("Connect Probe")
            else:
                await self.thermo.connect_routine()
                self.lbl_probe.setText("Probe: ⬤ Connected")
                self.btn_probe.setText("Disconnect Probe")
        except BleakError as e:
            QMessageBox.warning(self,"Probe Error",str(e))
            self.lbl_probe.setText("Probe: ⬤ Disconnected")
            self.btn_probe.setText("Connect Probe")
        finally:
            self.btn_probe.setEnabled(True)

    async def handle_cooker(self):
        try:
            self.lbl_cooker.setText("Cooker: ⬤ Connecting...")
            self.btn_cooker.setEnabled(False)
            if await self.cooker.connection_status():
                await self.cooker.disconnect_routine()
                self.lbl_cooker.setText("Cooker: ⬤ Disconnected")
                self.btn_cooker.setText("Connect Cooker")
            else:
                await self.cooker.connect_routine()
                self.lbl_cooker.setText("Cooker: ⬤ Connected")
                self.btn_cooker.setText("Disconnect Cooker")
        except BleakError as e:
            QMessageBox.warning(self,"Cooker Error",str(e))
            self.lbl_cooker.setText("Cooker: ⬤ Disconnected")
            self.btn_cooker.setText("Connect Cooker")
        finally:
            self.btn_cooker.setEnabled(True)

    async def start_loop(self):
        if self.running:
            QMessageBox.information(self,"Running","Loop already running")
            return
        if not (await self.thermo.connection_status() and await self.cooker.connection_status()):
            QMessageBox.warning(self,"Not connected","Connect probe and cooker first")
            return

        setpoint = self.sp_set.value()

        self.t.clear()
        self.core.clear()
        self.surface = []
        self.water.clear()

        # Initialize cascade PID controllers from control tab spinboxes
        self.pid_core_surface = PIDController(
            self.kpCoreSurfBox.value(),
            self.kiCoreSurfBox.value(),
            self.kdCoreSurfBox.value(),
            setpoint
        )
        self.pid_surface_water = PIDController(
            self.kpSurfWaterBox.value(),
            self.kiSurfWaterBox.value(),
            self.kdSurfWaterBox.value(),
            setpoint
        )
        self.pid_water = PIDController(
            self.kpWaterBox.value(),
            self.kiWaterBox.value(),
            self.kdWaterBox.value(),
            setpoint
        )

        mode = self.modeBox.currentText()
        self.running = True
        self.last_time = time.time()
        self.t0 = time.time()
        self.task_loop = asyncio.create_task(self.control_loop(mode))

    async def stop_loop(self):
        self.running = False
        if self.task_loop:
            await self.task_loop
            self.task_loop = None
        await self.cooker.standby()

    async def control_loop(self, mode):
        try:
            t0 = time.time()
            while self.running:
                now = time.time()
                dt = now - self.last_time
                self.last_time = now
                try:
                    core, surface, water = await self.thermo.temperature_read()
                    self.lbl_core.setText(f"Core: {core:.1f} °C")
                    self.lbl_water.setText(f"Water: {water:.1f} °C")
                    self.t.append(now - t0)
                    self.core.append(core); self.surface.append(surface); self.water.append(water)

                    setpoint = self.sp_set.value()
                    self.redraw()

                    if mode == "Manual":
                        return

                    if mode == "Normal":
                        error = setpoint - water
                        if error > 0:
                            await self.cooker.heat()
                        elif error < -2:
                            await self.cooker.cool()
                        else:
                            await self.cooker.dwell()

                    elif mode == "PID Auto":
                        # Cascade PID control: core→surface→water
                        self.pid_core_surface.target_temperature = setpoint
                        surface_sp = self.pid_core_surface.calculate(core, dt)

                        self.pid_surface_water.target_temperature = surface_sp
                        water_sp = self.pid_surface_water.calculate(surface, dt)

                        self.pid_water.target_temperature = water_sp
                        output = self.pid_water.calculate(water, dt)

                        print(
                            f"[CASCADE] core_sp={setpoint:.1f}, surf_sp={surface_sp:.1f}, water_sp={water_sp:.1f}, out={output:.2f}"
                        )

                        if output > 1:
                            await self.cooker.heat()
                        elif output < -1:
                            await self.cooker.cool()
                        else:
                            await self.cooker.dwell()


                except Exception as e:
                    print(f"[ERROR] Control loop exception: {e}")
        finally:
            await self.cooker.standby()

    def redraw(self):
        if self.t:
            self.graph.plot_data(
                np.array(self.t),
                np.array(self.core),
                np.array(self.surface),
                np.array(self.water),
                self.sp_set.value(),
            )

    def export_data(self):
        if not self.t:
            QMessageBox.information(self, "No Data", "There is no data to export.")
            return

        filename, _ = QFileDialog.getSaveFileName(self, "Save CSV", "sous_vide_data.csv", "CSV Files (*.csv)")
        if filename:
            exporter(
                self.t,
                self.core,
                self.surface,
                self.water,
                self.sp_set.value(),
                ["running"] * len(self.t),
                filename,
            )

    def closeEvent(self, event):
        # Save PID settings on exit
        self.save_pid_settings()
        event.accept()

def main():
    app = QApplication(sys.argv)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    win = MainWindow()
    win.show()
    with loop:
        loop.run_forever()

if __name__ == "__main__":
    main()

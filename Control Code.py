import asyncio, sys, time, numpy as np, qasync, json, os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
    QComboBox, QWidget, QGridLayout, QMessageBox, QFileDialog, QTabWidget, QVBoxLayout
)
from PyQt5.QtCore import Qt, QTimer
from bleak import BleakClient, BleakError
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
import csv
import optuna
import scipy.stats as st
import logging

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("optuna").setLevel(logging.WARNING)
logging.getLogger("optuna").propagate = False

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

class simulateCoolingSchedule():
    def __init__(self, setpoint, t0):
        self.time = None
        self.Tc = None
        self.Ts = None
        self.Tw = None
        self.t0 = t0
        self.target_core = setpoint


    def calculateParameters(self, time, Tc, Ts, Tw):

        self.time = np.asarray(time, dtype=float)
        self.Tc = np.asarray(Tc, dtype=float)
        self.Ts = np.asarray(Ts, dtype=float)
        self.Tw = np.asarray(Tw, dtype=float)

        # ---------------------------------------------------------------
        # 2)  Numerical derivatives  (central difference)
        # ---------------------------------------------------------------
        dTc_dt = np.gradient(self.Tc, self.time)
        dTs_dt = np.gradient(self.Ts, self.time)

        # ---------------------------------------------------------------
        # 3)  Linear fit for  a  in   dTc/dt = a·(Ts-Tc)
        # ---------------------------------------------------------------
        X_a = self.Ts - self.Tc
        Y_a = dTc_dt
        mask_a = np.abs(X_a) > 1e-3  # avoid divide-by-0
        Xa, Ya = X_a[mask_a], Y_a[mask_a]

        self.a_hat = (Xa @ Ya) / (Xa @ Xa)  # slope through origin
        res_a = Ya - self.a_hat * Xa
        n_a = len(Xa)
        sigma2a = (res_a @ res_a) / (n_a - 1)
        var_a = sigma2a / (Xa @ Xa)
        se_a = np.sqrt(var_a)
        self.ci_a = self.a_hat + st.t.ppf([0.025, 0.975], df=n_a - 1) * se_a

        # ---------------------------------------------------------------
        # 4)  Linear fit for  b  in   dTs/dt + a(Ts-Tc) = b·(Tw-Ts)
        # ---------------------------------------------------------------
        Y_b = dTs_dt + self.a_hat * (self.Ts - self.Tc)
        X_b = self.Tw - self.Ts
        mask_b = np.abs(X_b) > 1e-3
        Xb, Yb = X_b[mask_b], Y_b[mask_b]

        self.b_hat = (Xb @ Yb) / (Xb @ Xb)
        res_b = Yb - self.b_hat * Xb
        n_b = len(Xb)
        sigma2b = (res_b @ res_b) / (n_b - 1)
        var_b = sigma2b / (Xb @ Xb)
        se_b = np.sqrt(var_b)
        self.ci_b = self.b_hat + st.t.ppf([0.025, 0.975], df=n_b - 1) * se_b

    def calculateEffectiveWaterMass(self, time, Tw):

        self.time = np.asarray(time, dtype=float)
        self.Tw = np.asarray(Tw, dtype=float)

        dTw_dt = np.gradient(self.Tw, self.time)
        # ---------------------------------------------------------------
        # 4)  Linear fit for  m  in   dTw/dt = 1/M·(Ph - Kloss*(Tw-Ta))/Cp
        # ---------------------------------------------------------------
        Ph, Kloss, Ta, Cp = [1500, 16.94, 25.0, 4180]
        X_m = (Ph - Kloss * (self.Tw[:800] - Ta)) / Cp
        Y_m = dTw_dt[:800]
        mask_m = np.abs(X_m) > 1e-3
        Xm, Ym = X_m[mask_m], Y_m[mask_m]

        self.m_hat = (Xm @ Xm) / (Xm @ Ym)
        res_m = Ym - self.m_hat * Xm
        n_m = len(Xm)
        sigma2m = (res_m @ res_m) / (n_m - 1)
        var_m = sigma2m / (Xm @ Xm)
        se_m = np.sqrt(var_m)
        self.ci_m = self.m_hat + st.t.ppf([0.025, 0.975], df=n_m - 1) * se_m


    async def predictCoolingTime(self):

        duration_s = 60 * 60 - self.t0  # 60 minutes
        dt = 1.0
        self.steps = int(duration_s / dt) + 1

        def simulate_pork(hte: int,
                          Tw0: float = self.Tw[0],
                          Ts0: float = self.Ts[0],
                          Tc0: float = self.Tc[0],
                          *,
                          a: float = self.a_hat,  # conduction rate (surface -> core), 1/s
                          b: float = self.b_hat,  # convection rate (water -> surface), 1/s
                          m_water: float = self.m_hat,
                          cp_water: float = 4186,
                          P_heater: float = 1500,
                          Tw_max: float = 82.5,
                          Ta: float = 25.0,
                          Target_Core: float = self.target_core,
                          k_loss: float = 14.0,
                          ):


            t = np.arange(0, self.steps) * dt + self.t0

            N = int(duration_s / dt) + 1
            heater = np.zeros(N)
            heater[:hte] = 1.0

            cooler = np.zeros(N)
            cooler[hte:] = 1.0

            WAX_ENERGY_FULL = 264000
            wax_left = WAX_ENERGY_FULL

            def get_val(src, time_s, lo=0.0, hi=1.0):
                if callable(src):
                    val = float(src(time_s))
                else:
                    idx = min(int(time_s / dt), len(src) - 1)
                    val = float(src[idx])
                return max(lo, min(hi, val))

            def cooling_power_from_T(Tw):
                """Return cooling power (W, negative) for current water temperature."""
                Tw_med = [64.725, 65.15, 65.61666667, 66.35, 66.8, 67.2, 67.7, 68.3, 69.6, 70.3, 71.15, 72.25, 73.4,
                          74.35, 76., 78., 79., 80.65, 82.21, 82.7]
                Pc_med = [-34.6927029, -124.97092194, -346.38955333, -696.22550052, -927.63652278, -1302.36990235,
                          -1681.12659496, -1849.84280792, -1999.11749958, -2144.74028961, -2278.62155835, -2353.98138997,
                          -2423.82046095, -2491.63877127, -2544.83632095, -2588.31310997, -2640.96913835, -2679.70440608,
                          -2814.58564536, -2918.36965847]
                if Tw < 64.725:
                    return 0  # below melt range, basically no cooling
                if Tw > 82.5:
                    Tw = 82.5
                Pcool_T = PchipInterpolator(Tw_med, Pc_med, extrapolate=False)
                return Pcool_T(Tw)

            Tw = np.empty(self.steps)
            Ts = np.empty(self.steps)
            Tc = np.empty(self.steps)
            u_heat = np.zeros(self.steps)
            u_cool = np.zeros(self.steps)

            Tw[0], Ts[0], Tc[0] = Tw0, Ts0, Tc0
            tau_s = 240  # seconds, tweak
            Tw_f = Tw[0]
            Cw = m_water * cp_water
            chiller_on = False

            for k in range(1, self.steps):
                time_s = t[k]

                # Heater duty
                u = get_val(heater, time_s, 0.0, 1.0)
                u_heat[k] = u

                # Cooler on/off
                cool_cmd = get_val(cooler, time_s, 0.0, 1.0) > 0.5
                u_cool[k] = cool_cmd
                if cool_cmd and not chiller_on:
                    chiller_on = True
                    t_on_cool = 0.0
                elif not cool_cmd:
                    chiller_on = False
                    t_on_cool = 0.0

                if chiller_on:
                    Pc = cooling_power_from_T(Tw[k - 1])  # ≤ 0
                    # limit by remaining wax energy
                    Pc = max(Pc, -wax_left / dt)
                    wax_left += Pc * dt  # Pc negative → decreases wax_left
                else:
                    Pc = 0.0

                # Water dynamics
                P_net = P_heater * u + Pc - k_loss * (Tw[k - 1] - Ta)
                Tw[k] = Tw[k - 1] + (P_net / Cw) * dt
                if Tw[k] > Tw_max:
                    Tw[k] = Tw_max

                # Pork nodes
                Tw_f += (dt / tau_s) * (Tw[k - 1] - Tw_f)
                dTs = b * (Tw_f - Ts[k - 1]) - a * (Ts[k - 1] - Tc[k - 1])
                dTc = a * (Ts[k - 1] - Tc[k - 1])
                Ts[k] = Ts[k - 1] + dTs * dt
                Tc[k] = Tc[k - 1] + dTc * dt

            target_error = abs(max(Tc) - Target_Core)
            time_idx = np.where(Tc >= Target_Core)
            if len(time_idx[0]) == 0:
                target_time = float(-99999)
            else:
                target_time = (t[time_idx[0][0]]) / 60

            return {"t": t, "Tw": Tw, "Ts": Ts, "Tc": Tc, "u_heat": u_heat, "u_cool": u_cool, 'cool start': t[hte],
                    "target_error": target_error, "target_time": target_time}

        def objective(trial):
            hte = trial.suggest_int('hte', 0, self.steps-1)

            res = simulate_pork(hte)

            time = res['target_time']

            return time

        study = optuna.create_study(directions=['maximize'])
        study.optimize(objective, n_trials=100)

        hte = study.best_trials[0].params['hte']

        res = simulate_pork(hte)

        return res

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
            "core":    self.ax.plot([], [], 'r-', label="Core", linewidth=3.0)[0],
            "surface": self.ax.plot([], [], 'm-', label="Surface", linewidth=3.0)[0],
            "water":   self.ax.plot([], [], 'b-', label="Water", linewidth=3.0)[0],
            "set":     self.ax.plot([], [], 'k--', label="Set‑point")[0],
            "core sim":   self.ax.plot([], [], 'r--', label="Core sim")[0],
            "surface sim": self.ax.plot([], [], 'm--', label="Surface sim")[0],
            "water sim": self.ax.plot([], [], 'b--', label="Water sim")[0],
        }
        self.ax.legend()

    def plot_data(self, t, core, surface, water, setpoint, time_sim, core_sim, surface_sim, water_sim):
        """Update graph with new measurements."""
        for k, y in [
            ("core", core),
            ("surface", surface),
            ("water", water),
            ]:
            self.lines[k].set_data(t / 60, y)
        for k, y in [
            ("core sim", core_sim),
            ("surface sim", surface_sim),
            ("water sim", water_sim)
             ]:
            self.lines[k].set_data(time_sim / 60, y)
        self.ax.axhline(setpoint, color='k', linestyle='--')
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

        self.tabs.addTab(self.control_tab, "Control")

        self.setup_control_tab()

        self.setCentralWidget(self.tabs)

        # ---------- state ----------
        self.thermo = thermoprobe(PROBE_MAC)
        self.cooker = cooker(COOKER_MAC)
        self.task_loop = None
        self.t, self.core, self.surface, self.water, self.time_sim, self.core_sim, self.surface_sim, self.water_sim = [], [], [], [], [], [], [], []
        self.running = False

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
        self.modeBox.addItems(["Manual", "Normal", "Automatic"])
        self.modeBox.currentTextChanged.connect(self.mode_changed)

        self.sp_set = QSpinBox(); self.sp_set.setRange(20, 82); self.sp_set.setValue(55)

        self.lbl_core = QLabel("Core: --.- °C")
        self.lbl_surface = QLabel("Surface: --.- °C")
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

        g.addWidget(self.lbl_core, 4, 0, 1, 2); g.addWidget(self.lbl_surface, 5, 0, 1, 2); g.addWidget(self.lbl_water, 6, 0, 1, 2)
        g.addWidget(self.btn_start, 7, 0); g.addWidget(self.btn_stop, 7, 1)
        g.addWidget(self.btn_heat, 8, 0); g.addWidget(self.btn_cool, 8, 1)
        g.addWidget(self.btn_dwell, 9, 0); g.addWidget(self.btn_standby, 9, 1)
        g.addWidget(self.btn_export, 10, 0, 1, 2)
        g.addWidget(self.graph, 0, 2, 11, 1)

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

    def mode_changed(self, text):
        is_manual = (text == "Manual")
        is_auto = (text == 'Automatic')
        self.btn_heat.setEnabled(is_manual)
        self.btn_cool.setEnabled(is_manual)
        self.btn_dwell.setEnabled(is_manual)
        self.btn_standby.setEnabled(is_manual)
        self.btn_start.setEnabled(not is_manual)
        self.btn_stop.setEnabled(not is_manual)

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
        self.surface.clear()
        self.water.clear()
        self.time_sim.clear()
        self.core_sim.clear()
        self.surface_sim.clear()
        self.water_sim.clear()

        self.simulator = simulateCoolingSchedule(setpoint, 0.0)
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
            mass_latch = False
            heating_latch = True
            cooling_latch = False
            coolingStart = None

            while self.running:
                now = time.time()
                dt = now - self.last_time
                self.last_time = now
                try:
                    core, surface, water = await self.thermo.temperature_read()
                    self.lbl_core.setText(f"Core: {core:.1f} °C")
                    self.lbl_surface.setText(f"Surface: {surface:.1f} °C")
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

                    elif mode == "Automatic":

                        if coolingStart and (now - t0) > coolingStart:
                            heating_latch = False
                            cooling_latch = True

                        if (now - t0) >= 60*8 and not mass_latch and heating_latch:

                            self.simulator.calculateParameters(self.t, self.core, self.surface, self.water)
                            print(f'Conduction Coefficient: {self.simulator.a_hat:.2e} || Convection Coefficient: {self.simulator.b_hat:.2e}')
                            self.simulator.calculateEffectiveWaterMass(self.t, self.water)
                            mass_latch = True
                            print(f'Effective Water Mass: {self.simulator.m_hat:.2f}')
                            res = await self.simulator.predictCoolingTime()
                            coolingStart = res['cool start']
                            self.time_sim, self.core_sim, self.surface_sim, self.water_sim = [res['t'], res['Tc'],
                                                                                              res['Ts'], res['Tw']]
                            print(f'Cooling Start Time: {coolingStart/60:.2f} min')

                            self.simulator.t0 = now - t0

                        if ((now - t0)%180) - (((now-dt) - t0)%180) < 0 and mass_latch and heating_latch:
                            print('Updating Prediction...')
                            start_idx = np.where(np.asarray(self.t) >= self.simulator.t0)
                            self.simulator.calculateParameters(self.t[start_idx[0][0]:], self.core[start_idx[0][0]:], self.surface[start_idx[0][0]:], self.water[start_idx[0][0]:])
                            print(f'Conduction Coefficient: {self.simulator.a_hat:.2e} || Convection Coefficient: {self.simulator.b_hat:.2e}')
                            res = await self.simulator.predictCoolingTime()
                            coolingStart = res['cool start'] - self.simulator.t0
                            self.time_sim, self.core_sim, self.surface_sim, self.water_sim = [res['t'], res['Tc'],
                                                                                              res['Ts'], res['Tw']]
                            print(f'Cooling Start Time: {coolingStart/60:.2f} min')

                            self.simulator.t0 = now - t0

                        if heating_latch and water < 82.5:
                            await self.cooker.heat()
                        elif cooling_latch:
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
                np.array(self.time_sim),
                np.array(self.core_sim),
                np.array(self.surface_sim),
                np.array(self.water_sim)
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

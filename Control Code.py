import asyncio, sys, time, numpy as np, qasync
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
    QComboBox, QWidget, QGridLayout, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer
from bleak import (BleakClient, BleakError)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
import csv

GRAPH_INTERVAL = 1000           # ms between redraws
CONTROL_INTERVAL = 2           # s between PID decisions
COOKER_MAC = "94:A9:A8:19:77:5F"
PROBE_MAC = "C2:71:1E:F2:C6:20"


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
        self.ax.set_xlabel("Time [s]")
        self.ax.set_ylabel("T [°C]")
        self.lines = {
            "core":  self.ax.plot([], [], 'r-', label="Core")[0],
            "water": self.ax.plot([], [], 'b-', label="Water")[0],
            "set":   self.ax.plot([], [], 'k--',label="Set‑point")[0],
        }
        self.ax.legend()

    def plot_data(self, t, core, water, setpoint):
        for k, y in [("core",core),("water",water),("set",np.full_like(core,setpoint))]:
            self.lines[k].set_data(t, y)
        self.ax.relim(); self.ax.autoscale_view()
        self.draw_idle()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sous‑Vide Control GUI")
        self.resize(880,480)

        # ---------- widgets ----------
        self.btn_probe   = QPushButton("Connect Probe")
        self.btn_cooker  = QPushButton("Connect Cooker")
        self.lbl_probe   = QLabel("Probe: ⬤ Disconnected")
        self.lbl_cooker  = QLabel("Cooker: ⬤ Disconnected")

        self.modeBox     = QComboBox()
        self.modeBox.addItems(["Manual", "Normal", "PID Auto"])
        self.modeBox.currentTextChanged.connect(self.mode_changed)

        self.sp_set      = QSpinBox(); self.sp_set.setRange(20, 95); self.sp_set.setValue(55)
        self.kpBox       = QDoubleSpinBox(); self.kpBox.setRange(0,100); self.kpBox.setValue(1.0)
        self.kiBox       = QDoubleSpinBox(); self.kiBox.setRange(0,100); self.kiBox.setValue(0.0)
        self.kdBox       = QDoubleSpinBox(); self.kdBox.setRange(0,2000); self.kdBox.setValue(0.0)

        self.lbl_core    = QLabel("Core: --.- °C")
        self.lbl_water   = QLabel("Water: --.- °C")
        self.btn_start   = QPushButton("Start")
        self.btn_stop    = QPushButton("Stop")
        self.btn_heat    = QPushButton("Heat")
        self.btn_cool    = QPushButton("Cool")
        self.btn_dwell   = QPushButton("Dwell")
        self.btn_standby = QPushButton("Standby")

        self.graph       = MatplotCanvas(self)

        # ---------- layout ----------
        g = QGridLayout()
        g.addWidget(self.btn_probe, 0,0);  g.addWidget(self.lbl_probe, 0,1)
        g.addWidget(self.btn_cooker,1,0);  g.addWidget(self.lbl_cooker,1,1)
        g.addWidget(QLabel("Mode:"),       2,0); g.addWidget(self.modeBox,2,1)
        g.addWidget(QLabel("Set‑point °C"),3,0); g.addWidget(self.sp_set,3,1)
        g.addWidget(QLabel("Kp"),4,0); g.addWidget(self.kpBox,4,1)
        g.addWidget(QLabel("Ki"),5,0); g.addWidget(self.kiBox,5,1)
        g.addWidget(QLabel("Kd"),6,0); g.addWidget(self.kdBox,6,1)
        g.addWidget(self.lbl_core,7,0,1,2);  g.addWidget(self.lbl_water,8,0,1,2)
        g.addWidget(self.btn_start,9,0);     g.addWidget(self.btn_stop,9,1)
        g.addWidget(self.btn_heat,10,0);     g.addWidget(self.btn_cool,10,1)
        g.addWidget(self.btn_dwell,11,0);    g.addWidget(self.btn_standby,11,1)
        g.addWidget(self.graph,0,2,12,1)

        central = QWidget(); central.setLayout(g)
        self.setCentralWidget(central)

        # ---------- state ----------
        self.thermo  = thermoprobe(PROBE_MAC)
        self.cooker  = cooker(COOKER_MAC)
        self.task_loop = None
        self.t, self.core, self.water = [],[],[]

        # ---------- signals ----------
        self.btn_probe.clicked.connect(lambda: asyncio.create_task(self.handle_probe()))
        self.btn_cooker.clicked.connect(lambda: asyncio.create_task(self.handle_cooker()))
        self.btn_start.clicked.connect(lambda: asyncio.create_task(self.start_loop()))
        self.btn_stop.clicked.connect(lambda: asyncio.create_task(self.stop_loop()))
        self.btn_heat.clicked.connect(lambda: asyncio.create_task(self.cooker.heat()))
        self.btn_cool.clicked.connect(lambda: asyncio.create_task(self.cooker.cool()))
        self.btn_dwell.clicked.connect(lambda: asyncio.create_task(self.cooker.dwell()))
        self.btn_standby.clicked.connect(lambda: asyncio.create_task(self.cooker.standby()))

        # auto‑refresh graph timer
        self.timer = QTimer(self); self.timer.timeout.connect(self.redraw)
        self.timer.start(GRAPH_INTERVAL)
        self.mode_changed("Manual")

    def mode_changed(self, text):
        enabled = (text == "Manual")
        self.btn_heat.setEnabled(enabled)
        self.btn_cool.setEnabled(enabled)
        self.btn_dwell.setEnabled(enabled)
        self.btn_standby.setEnabled(enabled)

    async def handle_probe(self):
        try:
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

    async def handle_cooker(self):
        try:
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

    async def start_loop(self):
        if self.task_loop:
            QMessageBox.information(self,"Running","Loop already running")
            return
        if not (await self.thermo.connection_status() and await self.cooker.connection_status()):
            QMessageBox.warning(self,"Not connected","Connect probe and cooker first")
            return

        self.t.clear(); self.core.clear(); self.water.clear()
        mode = self.modeBox.currentText()
        self.running = True
        self.t0 = time.time()
        self.pid = PIDController(self.kpBox.value(), self.kiBox.value(),
                                 self.kdBox.value(), self.sp_set.value())
        self.task_loop = asyncio.create_task(self.loop(mode))

    async def stop_loop(self):
        self.running = False
        if self.task_loop:
            await self.task_loop
            self.task_loop = None
        await self.cooker.standby()

    async def loop(self, mode):
        try:
            last = time.time()
            while self.running:
                try:
                    core,surf,water = await self.thermo.temperature_read()
                except Exception as e:
                    self.statusBar().showMessage(f"Probe read error: {e}")
                    await asyncio.sleep(2); continue

                now = time.time()
                self.t.append(now-self.t0)
                self.core.append(core)
                self.water.append(water)
                self.lbl_core.setText(f"Core: {core:.1f} °C")
                self.lbl_water.setText(f"Water: {water:.1f} °C")
                self.redraw()

                if mode=="Manual":
                    pass
                else:
                    dt = now - last; last = now
                    out = self.pid.calculate(water if mode=="Normal" else core, dt)
                    if mode=="Normal":
                        if out>0.5:  await self.cooker.heat()
                        else:        await self.cooker.dwell()
                    elif mode=="PID Auto":
                        if out>5:         await self.cooker.heat()
                        elif out<-5:      await self.cooker.cool()
                        else:             await self.cooker.dwell()
                await asyncio.sleep(CONTROL_INTERVAL)
        finally:
            pass

    def redraw(self):
        if self.t:
            self.graph.plot_data(np.array(self.t),
                                 np.array(self.core),
                                 np.array(self.water),
                                 self.sp_set.value())


def main():
    app = QApplication(sys.argv)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    win = MainWindow(); win.show()
    with loop:
        loop.run_forever()


if __name__ == "__main__":
    main()

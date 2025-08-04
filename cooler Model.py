# -----------------------------------------------
# 1) Set paths & constants
# -----------------------------------------------
CSV_FILE   = "Potato IMC Test 1.csv"   # <- full path if needed
M_WATER_KG = 5.5                      # bath mass (kg)
CP_WATER   = 4186.0                    # J kg‑1 K‑1

# -----------------------------------------------
# 2) Load data  (time in s, water‑T in °C)
#    ─ first row = time, 4th row = water T
#    adjust indices if your CSV differs
# -----------------------------------------------
import pandas as pd, numpy as np, matplotlib.pyplot as plt
raw = pd.read_csv(CSV_FILE, header=None, encoding="ISO‑8859‑1")
time_s = raw.iloc[0,1:].astype(float).values
Tw_C    = raw.iloc[3,1:].astype(float).values

import optuna
heating_end = 700

heating_end_index = np.where(time_s < heating_end)

time_h   = time_s[:heating_end_index[0][-1]]
Tw_h     = Tw_C[:heating_end_index[0][-1]]

def objective(trial):
    Kloss = trial.suggest_float('Kloss', 0, 50.0)

    dTw_dt = np.gradient(Tw_h, time_h)
    Ph, Ta, Cp = [1500, 25.0, 4180]
    X_m = (Ph - Kloss*(Tw_h - Ta))/(Cp*M_WATER_KG)
    Y_m = dTw_dt
    mask_m = np.abs(X_m) > 1e-3
    Xm, Ym = X_m[mask_m], Y_m[mask_m]

    RSS = sum(pow(Ym - Xm, 2))
    TSS = sum(pow(Ym - np.mean(Ym), 2))

    R2 = RSS/TSS
    return 1-R2

study = optuna.create_study(directions=['maximize'])
study.optimize(objective, n_trials=1000)
K_LOSS = study.best_trials[0].params['Kloss']
print(K_LOSS)

x = range(heating_end)
y = np.zeros(len(x))
y[0] = 56.5
for i in range(1, len(x)):
    y[i] = y[i-1] + (1500 - K_LOSS*(y[i-1] - 25))/(4180*M_WATER_KG)
plt.plot(time_s, Tw_C)
plt.plot(x, y)
plt.show()


# -----------------------------------------------
# 3) Identify the cool‑down segment
#    Here: everything after the global maximum of Tw
# -----------------------------------------------

cooling_start = 1150

cooling_start_index = np.where(time_s > cooling_start)

time_c   = time_s[cooling_start_index[0][0]:]
Tw_c     = Tw_C[cooling_start_index[0][0]:]


# Throw away the first 20 s of cool segment to skip the discontinuity


# -----------------------------------------------
# 4) Compute cooling power [W] = m cp dT/dt
# -----------------------------------------------
def temp2energy(temp):
    energy = []
    for i in temp:
        energy.append((i+273.15)*4180*M_WATER_KG)
    return energy
energy = temp2energy(Tw_c)

params = np.polyfit(time_c, energy, 30)

# -----------------------------------------------
# 5) Build monotone P‑chip interpolator P(Tw)
# -----------------------------------------------
wattage_params = np.polyder(params, m=1)
cooling_wattage = []
ambient_loss = np.gradient(M_WATER_KG*K_LOSS*(Tw_c-25.0), time_c)
for i in range(len(Tw_c)):
    cooling_wattage.append(np.polyval(wattage_params, time_c[i]) - ambient_loss[i])

wattage_afo_temp_params = np.polyfit(Tw_c, cooling_wattage, 15)
wattage_afo_temp = np.polyval(wattage_afo_temp_params, Tw_c)
print(wattage_afo_temp_params)

# -----------------------------------------------
# 6) Plot & sanity‑check
# -----------------------------------------------
plt.figure(figsize=(6,4))
Tw_grid = np.linspace(Tw_c.max(), Tw_c.min(), 200)
plt.plot(Tw_c, cooling_wattage, "r--", lw=2, label="data")
plt.plot(Tw_c, wattage_afo_temp, "k-", lw=2, label="poly fit")
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("Water temperature (°C)")
plt.ylabel("Cooling power (kW, negative = remove heat)")
plt.title("Wax chiller power vs. water temperature")
plt.legend(); plt.tight_layout(); plt.show()

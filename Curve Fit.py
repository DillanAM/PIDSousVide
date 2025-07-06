import numpy
import matplotlib.pyplot as plt
import csv

def importer(filename):
    with open(str(filename), 'r') as file:
        reader = csv.reader(file)
        data = []
        x = []
        y = []
        i = True
        for row in reader:
            data.append(row)

        for i in range(int((len(data)-1)/2)):
            x.append(float(data[i][0]))

        for i in range(int(((len(data)-1)/2)+1), int(len(data))):
            y.append(float(data[i][0]))

    return x, y


def temp2energy(temp):
    energy = []
    for i in temp:
        energy.append((i+273.15)*4180*17.4)
    return energy


def R2(x, y, param):

    y_pred = numpy.polyval(param, x)

    sst = numpy.sum((y - numpy.mean(y)) ** 2)
    sse = numpy.sum((y - y_pred) ** 2)
    return 1 - (sse/sst)

filename = 'Radiator Chiller Wax'

sec, temp = importer(filename)

energy = temp2energy(temp)

params = numpy.polyfit(sec, energy, 30)
energy_pred = numpy.polyval(params, sec)
R2 = R2(sec, energy, params)

print(R2)

time = range(1, 7200)
wattage_params = numpy.polyder(params, m=1)
wattage = numpy.polyval(wattage_params, time)
cooling_rating = numpy.mean(wattage[0:899])

print(cooling_rating)

plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
plt.plot(sec, energy, 'r-', sec, energy_pred, 'b--')
plt.title('Thermal Energy over Time', fontweight='bold')
plt.legend(['Data', 'Curve Fit R2: %5.3f' % R2])
plt.xlabel('Time (s)', fontweight='bold')
plt.ylabel('Thermal Energy (J)', fontweight='bold')
plt.subplot(1, 2, 2)
plt.plot(time, wattage)
plt.axvline(x=900, color='r', linestyle='dashed')
plt.text(2000, -3000, 'Cooling Rating: %5.0f W' % cooling_rating, bbox=dict(facecolor='white', alpha=0.5))
plt.title('Heat-flow over Time', fontweight='bold')
plt.xlabel('Time (s)', fontweight='bold')
plt.ylabel('Cooling Power (W)', fontweight='bold')
plt.suptitle('Power Analysis of %s' % filename, fontweight='bold')
plt.show()



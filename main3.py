import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import PolynomialModel, ExponentialModel
from sklearn.metrics import r2_score

# Fig 1

root = ET.parse('HY202103_D08_(0,2)_LION1_DCM_LMZC.xml').getroot()

v = []
for child in root.find('.//IVMeasurement'):
    v.append(list(map(float, child.text.split(','))))

v[1] = list(map(abs, v[1]))


x2 = np.asarray(v[0][5:13])
y2 = np.asarray(v[1][5:13])

m0 = PolynomialModel()
pars = m0.guess(y2, x=x2)
o = m0.fit(y2, pars, x=x2)

plt.subplot(235)

plt.plot(x2, o.best_fit, '-', label='best fit 2')
print('Residual value IV: ', r2_score(y2, o.best_fit))

x = np.asarray(v[0][0:8])
y = np.asarray(v[1][0:8])

m = PolynomialModel()
pars = m.guess(y, x=x)
o2 = m.fit(y, pars, x=x)

plt.plot(x, o2.best_fit, '-', label='best fit 2')
print('Residual value IV: ', r2_score(y, o2.best_fit))


plt.plot(v[0], v[1], 'o')
plt.yscale('log')
plt.legend()



# r hoch 2 berechnen max min and waveleangth , deviding it in two parts the IV , values nicht mehr in absolut legend

plt.subplot(234)
plt.plot(v[0], v[1], 'k-o')
plt.title("IV-Analysis", size=20)
plt.xlabel('Voltage [V]', size=15)
plt.ylabel('Current [A]', size=15)
plt.yscale('log')


v = []
for waveLengthSweep in root.findall('.//WavelengthSweep'):
    waveValues = []
    for child in waveLengthSweep:
        waveValues.append(list(map(float, child.text.split(','))))
    waveValues.append(waveLengthSweep.attrib['DCBias'])
    v.append(waveValues)


plt.subplot(231)

plots = []
for i in range(len(v) - 1):
    line, = plt.plot(v[i][0], v[i][1], label="DCBias=\"" + v[i][2] + "\"")
    plots.append(line)

line, = plt.plot(v[6][0], v[6][1], color='black', label="Fit ref polynomial O3")

plt.gca().add_artist(plt.legend(handles=[line], loc='upper right'))
plt.legend(handles=plots, ncol=2, loc="lower center")
plt.title("Transmission spectra - as measured", size=20)
plt.xlabel('Wavelength [nm]', size=15)
plt.ylabel('Measured transmission [dB]', size=15)


plt.subplot(232)

line, = plt.plot(v[6][0], v[6][1], color='black', label="Fit ref polynomial O3")
plt.gca().add_artist(plt.legend(handles=[line], loc='upper right'))


model = np.poly1d(np.polyfit(v[6][0], v[6][1], 2))
polyline = np.linspace(1530, 1580, 6065)
l2, = plt.plot(polyline, model(polyline), '--', color='red', label="degree 2")

model2 = np.poly1d(np.polyfit(v[6][0], v[6][1], 3))
polyline = np.linspace(1530, 1580, 6065)
l3, = plt.plot(polyline, model2(polyline), '--', color='green', label="degree 3")

model3 = np.poly1d(np.polyfit(v[6][0], v[6][1], 4))
polyline = np.linspace(1530, 1580, 6065)
l4, = plt.plot(polyline, model3(polyline), '--', color='blue', label="degree 4")

model4 = np.poly1d(np.polyfit(v[6][0], v[6][1], 5))
polyline = np.linspace(1530, 1580, 6065)
l5, = plt.plot(polyline, model4(polyline), '--', color='orange', label="degree 5")

plt.legend(handles=[l2, l3, l4, l5], ncol=2, loc="lower center")

print('Residual value for degree 2: ', r2_score(v[6][1], model(v[6][0])))
print('Residual value for degree 3: ', r2_score(v[6][1], model2(v[6][0])))
print('Residual value for degree 4: ', r2_score(v[6][1], model3(v[6][0])))
print('Residual value for degree 5: ', r2_score(v[6][1], model4(v[6][0])))


plt.subplot(233)

plots = []
for i in range(len(v) - 1):
    # subtracting values
    if len(v[i][1]) == len(v[6][1]):
        array1 = np.array(v[i][1])
        array2 = np.array(v[6][1])
        subtracted_array = np.subtract(array1, array2)
        subtracted = list(subtracted_array)
    else:
        n = len(v[i][1]) - len(v[6][1]) if len(v[i][1]) > len(v[6][1]) else len(v[6][1]) - len(v[i][1])
        v[i][0] = v[i][0][:-n]
        array1 = np.array(v[i][1][:-n])
        array2 = np.array(v[6][1])
        subtracted_array = np.subtract(array1, array2)
        subtracted = list(subtracted_array)

    # plot values
    line, = plt.plot(v[i][0], subtracted, label="DCBias=\"" + v[i][2] + "\"")
    plots.append(line)

line, = plt.plot(v[6][0], v[6][1], color='black', label="Fit ref polynomial O3")
plt.gca().add_artist(plt.legend(handles=[line], loc='upper right'))
plt.legend(handles=plots, ncol=2, loc="lower center")
plt.title("Transmission spectra - as measured")
plt.xlabel('Wavelength [nm]')
plt.ylabel('Measured transmission [dB]')
plt.show()
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import lmfit
from scipy.optimize import curve_fit


# Fig 1, IV analysis raw data
from scipy.signal import argrelextrema

root = ET.parse('HY202103_D08_(0,2)_LION1_DCM_LMZC.xml').getroot()

v = []
for child in root.find('.//IVMeasurement'):
    v.append(list(map(float, child.text.split(','))))

v[1] = list(map(abs, v[1]))

q = 1.6e-19
kT = 1.3806488e-23
I_S = -2.0
n = 1

V_D = np.linspace(-3.0, 2.0, 100)
I_D = I_S * (np.exp(q*V_D/n*kT)-1)

mod = lmfit.models.ExponentialModel()
pars = mod.guess(I_D, x = V_D)
out = mod.fit(I_D, pars, x = V_D)

print(out.fit_report())

plt.subplot(231)
plt.plot(v[0], v[1], 'k-o')
plt.title("IV-Analysis", size=20)
plt.xlabel('Voltage [V]', size=15)
plt.ylabel('Current [A]', size=15)
plt.grid(True)
plt.minorticks_off()
plt.yscale('logit')


# Fig 3, Tramsimission spectrum data (all)

v = []
for waveLengthSweep in root.findall('.//WavelengthSweep'):
    waveValues = []
    for child in waveLengthSweep:
        waveValues.append(list(map(float, child.text.split(','))))
    waveValues.append(waveLengthSweep.attrib['DCBias'])
    v.append(waveValues)

plt.subplot(233)

plots = []
for i in range(len(v) - 1):
    line, = plt.plot(v[i][0], v[i][1], label="DCBias=\"" + v[i][2] + "\"")
    plots.append(line)


line, = plt.plot(v[6][0], v[6][1], color='black', label="Fit ref polynomial O3")


model = np.poly1d(np.polyfit(v[6][0], v[6][1], 2))
polyline = np.linspace(1530, 1580, 10000)
plt.plot(polyline, model(polyline), '--', color='red')

plt.gca().add_artist(plt.legend(handles=[line], loc='upper right'))
plt.legend(handles=plots, ncol=2, loc="lower center")
plt.title("Transmission spectra - as measured", size=20)
plt.xlabel('Wavelength [nm]', size=15)
plt.ylabel('Measured transmission [dB]', size=15)




# Fig 2, Fit ref polynomial

plt.subplot(232)
line, = plt.plot(v[6][0], v[6][1], color='black', label="Fit ref polynomial O3")


model = np.poly1d(np.polyfit(v[6][0], v[6][1], 2))
polyline = np.linspace(1530, 1580, 10000)
plt.plot(polyline, model(polyline), '--', color='red')


# R2 square
wavePredict = []
waveActual = []

for i in range(len(v[6][0])):
    nowPredict = model(v[6][0][i])
    wavePredict.append(nowPredict)
    waveActual.append(v[6][1][i])

corr_matrix = np.corrcoef(waveActual, wavePredict)
corr = corr_matrix[0, 1]
R_sq = corr ** 2

print("R square : ", R_sq)
print(model)
print("Max : ", max(v[6][1]))
print("Min : ", min(v[6][1]))



plt.gca().add_artist(plt.legend(handles=[line], loc='upper right'))

plt.title("Transmission spectra - as measured", size=20)
plt.xlabel('Wavelength [nm]', size=15)
plt.ylabel('Measured transmission [dB]', size=15)




# Fig 4, raw data of transmission spectrum
plt.subplot(234)
plots = []

for i in range(len(v) - 1):
    waveLift = []
    waveBase = []

    for j in range(len(v[i][0]) - 1):
        nowWaveLift = v[i][1][j] - wavePredict[j]
        waveLift.append (nowWaveLift)
        waveBase.append (v[i][0][j])
    line, = plt.plot(waveBase, waveLift, label="DCBias="" + v[i][2] + """)
plots.append(line)


plt.title("Transmission spectra - as measured", size=20)
plt.xlabel('Wavelength [nm]', size=15)
plt.ylabel('Measured transmission [dB]', size=15)

plt.show()



import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# Fig 1
from scipy.signal import argrelextrema

root = ET.parse('HY202103_D08_(0,2)_LION1_DCM_LMZC.xml').getroot()

v = []
for child in root.find('.//IVMeasurement'):
    v.append(list(map(float, child.text.split(','))))

v[1] = list(map(abs, v[1]))

plt.subplot(121)
plt.plot(v[0], v[1], 'k-o')
plt.title("IV-Analysis", size=20)
plt.xlabel('Voltage [V]', size=15)
plt.ylabel('Current [A]', size=15)
plt.yscale('log')

# Fig 2

v = []
for waveLengthSweep in root.findall('.//WavelengthSweep'):
    waveValues = []
    for child in waveLengthSweep:
        waveValues.append(list(map(float, child.text.split(','))))
    waveValues.append(waveLengthSweep.attrib['DCBias'])
    v.append(waveValues)

plt.subplot(122)

plots = []
for i in range(len(v) - 1):
    line, = plt.plot(v[i][0], v[i][1], label="DCBias=\"" + v[i][2] + "\"")
    plots.append(line)


#line, = plt.plot(v[6][0], v[6][1], color='black', label="Fit ref polynomial O3")


model = np.poly1d(np.polyfit(v[6][0], v[6][1], 2))
polyline = np.linspace(1530, 1580, 10000)
plt.plot(polyline, model(polyline), '--', color='red')

#plt.gca().add_artist(plt.legend(handles=[line], loc='upper right'))
plt.legend(handles=plots, ncol=2, loc="lower center")
plt.title("Transmission spectra - as measured", size=20)
plt.xlabel('Wavelength [nm]', size=15)
plt.ylabel('Measured transmission [dB]', size=15)
plt.tight_layout()
plt.show()
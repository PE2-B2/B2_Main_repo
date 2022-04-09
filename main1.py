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
plt.title("IV-Analysis")
plt.xlabel('Voltage [V]')
plt.ylabel('Current [A]')
plt.yscale('log')

# Fig 2

v = []
for waveLengthSweep in root.findall('.//WavelengthSweep'):
    waveValues = []
    for child in waveLengthSweep:
        waveValues.append(list(map(float, child.text.split(','))))
    waveValues.append(waveLengthSweep.attrib['DCBias'])
    v.append(waveValues)

# getting local minima
extremeValues = []
for i in range(len(v) - 1):
    points = []
    x_Values = []
    y_Values = []
    for index in argrelextrema(np.array(v[i][1]), np.less_equal, order=250):
        for value in index:
            x_Values.append(v[i][0][value])
            y_Values.append(v[i][1][value] - v[6][1][value])
    points.append(x_Values)
    points.append(y_Values)
    extremeValues.append(points)

plt.subplot(122)

# plot local minima
for i in range(len(extremeValues) - 1):
    plt.scatter(extremeValues[i][0], extremeValues[i][1], s=100)

# plot wavelength after subtracting ref from y-values
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
plt.tight_layout()
plt.show()
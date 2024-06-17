import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import welch
from dat_hea_reader import *
from eeg_avg import *
import scipy.signal as sig
import os


df = pd.read_csv('TP_PSB\data_avg_N1\promedios_homo.csv')

spl_100 = df['100']
print(spl_100)
fs = 48000

""" # Calcular el periodograma
periodograma = np.abs(np.fft.fft(st1))**2 / len(st1)

# Calcular la frecuencia correspondiente a cada componente del periodograma
w=fftfreq(len(periodograma))*N

# Graficar el periodograma
plt.figure(figsize=(10, 5))
plt.plot(w[:n // 2], periodograma[:n // 2])
plt.title('Periodograma')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Densidad Espectral de Potencia')
plt.grid(True)
plt.show() """


#welch

nper = int(len(spl_100)//25)
f, Pxx_den = welch(spl_100, fs, noverlap=nper//2  , nperseg=nper)
plt.figure(figsize = (9.7,4))
plt.plot(f, Pxx_den)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [uV^2/Hz]')
plt.title('Welch')
plt.grid()
plt.show()
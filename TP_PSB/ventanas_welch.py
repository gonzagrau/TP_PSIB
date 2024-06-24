import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import welch

# cargo la data
path = r'data_avg_N1\promedios_homo.csv'

df = pd.read_csv(path)
spl_100 = df['100']
fs = 48000
N = len(spl_100)

# Periodograma
periodograma = np.abs(np.fft.fft(spl_100))**2 / N
# Frecs del periodograma
w = np.fft.fftfreq(N, 1/fs)

# valores de nper
nper_values = [5, 10, 25]

# Set up subplots
fig, axs = plt.subplots(1, len(nper_values) + 1, figsize=(20, 5))

# Plot periodograma
axs[0].set_xlim(0, 2000)
axs[0].plot(w[:N // 2], periodograma[:N // 2])
axs[0].set_title('Periodograma')
axs[0].set_xlabel('Frecuencia (Hz)')
axs[0].set_ylabel('Densidad Espectral de Potencia')
axs[0].grid(True)

# Loop over sobre valores y Welch
for i, nper_factor in enumerate(nper_values):
    nper = int(len(spl_100) // nper_factor)
    f, Pxx_den = welch(spl_100, fs, noverlap=nper // 2, nperseg=nper)
    
    axs[i + 1].set_xlim(0, 2000)
    axs[i + 1].plot(f, Pxx_den)
    axs[i + 1].set_title(f'Welch con nper = {nper_factor}')
    axs[i + 1].set_xlabel('Frecuencia (Hz)')
    axs[i + 1].set_ylabel('Densidad Espectral de Potencia')
    axs[i + 1].grid(True)

# Show plot
plt.tight_layout()
plt.show()
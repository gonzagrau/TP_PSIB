import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import welch
from dat_hea_reader import *
from eeg_avg import *
import scipy.signal as sig
import os


df = pd.read_csv('data_avg_N1/promedios_homo.csv')
spl_100 = df['100']
fs = 48000
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
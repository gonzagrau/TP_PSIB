import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dat_hea_reader import *
from eeg_avg import *
import scipy.signal as sig
import os


df = pd.read_csv('data.csv')

print(df.to_string()) 

#welch
nper = int(len(x[canal-1])//25)
f, Pxx_den = scipy.signal.welch(x[canal-1], fs, noverlap=nper/2  , nperseg=nper)
plt.figure(figsize = (9.7,4))
plt.plot(f, Pxx_den)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [uV^2/Hz]')
plt.title('Welch')
plt.grid()
plt.show()
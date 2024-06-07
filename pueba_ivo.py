import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dat_hea_reader import *
from eeg_avg import *
import scipy.signal as sig
import os

lista_paths=[]
directory = os.fsencode('base_de_datos_Prueba')

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".dat"):
        # Join the bytes path of the directory with the string filename
        filepath = os.path.join(directory, os.fsencode(filename))
        # Decode the filepath for printing
        name = os.fsdecode(filepath)
        lista_paths.append(name.strip('.dat'))

cant_files = len(lista_paths)

mat_trials_mean = np.zeros((cant_files,2002))
mat_trials_amp = np.zeros((cant_files,2002))
mat_trials_var = np.zeros((cant_files,2002))
mat_trials_both = np.zeros((cant_files,2002))


for i in range(0,cant_files):

    for file in lista_paths:
        try:
            fs, trials, comments = read_trials(file)

            tr_len = comments['Trial Length (samples)']
            t = np.linspace(0, tr_len/fs, tr_len)*1000 # convert to ms

            trials_mean = average_EEG(trials, mode='homogenous') 
            mat_trials_mean[i] = trials_mean

            trials_amp = average_EEG(trials, mode='amp')
            mat_trials_amp[i] = trials_amp

            trials_var = average_EEG(trials, mode='var')
            mat_trials_var[i] = trials_var

            trials_both = average_EEG(trials, mode='both')
            mat_trials_both[i] = trials_both


        except ValueError:
            print(f"error con el archivo {file}")
            break
    else:
        print('todo ok')

df = pd.DataFrame({'t':t, 'mean':trials_mean, 'amp':trials_amp, 'var':trials_var, 'both':trials_both})
df.to_csv("N1_evoked_raw_100_F1_R1.csv")
df.head(10)

df = pd.DataFrame({'t':t, 'mean':trials_mean, 'amp':trials_amp, 'var':trials_var, 'both':trials_both})
'1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' '13' 14 15 16 17 18 19

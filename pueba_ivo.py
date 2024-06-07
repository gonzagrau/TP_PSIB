import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dat_hea_reader import *
from eeg_avg import *
import scipy.signal as sig

#lista de las aplitudes que van de 10 a 100
list = []
for i in range(10,105,5):
    list.append(i)
amplitudes = np.array(list)  

lista_paths=[]
for i in amplitudes:
    lista_paths.append(f"C:\\Users\\Usuario\\Desktop\\evoked-auditory-responses-in-normals-1.0.0\\raw\\N1\\N1_evoked_raw_{i}_F1_R1")
   

#path_realizacion = r'datos_ejemplo_señales\N1_evoked_raw_100_F1_R1'
fs, trials, comments = read_trials(r'datos_ejemplo_señales\N1_evoked_raw_100_F1_R1')


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

for file in lista_paths:
    try:
        fs, trials, comments = read_trials(file)
    except ValueError:
        print(f"error con el archivo {file}")
        break
else:
    print('todo ok')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import welch
from dat_hea_reader import *
from eeg_avg import *
import scipy.signal as sig
import os
from waveVamp import exponential_regression


#genero una funcion que dado un path a un csv me devuelve la suma del welch para cada header
def suma_welch_peri(path,
                    fs: int = 48000,
                    plot: bool = False):
    #cargo la data
    df = pd.read_csv(path)
    headers = df.columns.values.tolist()
    n = len(headers)

    #genero una matriz para almacenar la data

    PSD_sum_mat_welch = np.zeros((n,2))
    PSD_sum_mat_Periodograma = np.zeros((n,2))

    for  i in range(n):
        #  Welch's 
        spl = df[headers[i]]
        N = len(spl)
        nper = int(len(spl) // 5)
        ## se decide hacer 5 ventanas e manera empirica ya que se realizaron varios tests y es el que mas suavisa al periodograma sin tener errores por promediacion 
        ##el overlap es el clasico del 50% ya que se considera que es el que da mejores resultados
        f, Pxx_den = welch(spl, fs, noverlap=nper//2, nperseg=nper)

        #se realiza la integral
        suma_PSD = sum(Pxx_den)
        #se almacenan los datos
        PSD_sum_mat_welch[i,0] = int(headers[i])
        PSD_sum_mat_welch[i,1] = 1/suma_PSD
    

        # Periodograma
        periodograma = np.abs(np.fft.fft(spl))**2 /N
        # Frecs del periodograma
        w = np.fft.fftfreq(N, 1/fs)
         #se realiza la integral
        suma_PSD_Peri = sum(periodograma)
        #se almacenan los datos
        PSD_sum_mat_Periodograma[i,0] = int(headers[i])
        PSD_sum_mat_Periodograma[i,1] = 1/suma_PSD_Peri
        
    
    #normalizo por el valor mas alto
    max_per_col_welch = np.max(PSD_sum_mat_welch,axis=0)
    max_welch = max_per_col_welch[1]

    max_per_col_peri = np.max(PSD_sum_mat_Periodograma,axis = 0)
    max_peri = max_per_col_peri[1]

    for i in range(n):  
        PSD_sum_mat_welch[i,1] = PSD_sum_mat_welch[i,1]/max_welch
        PSD_sum_mat_Periodograma[i,1] = PSD_sum_mat_Periodograma[i,1]/max_peri
    
    

    return PSD_sum_mat_welch,PSD_sum_mat_Periodograma


def graf_intensidad_vs_welch_suma(mat: np.array, metodo: str,modo: str = 'welch'):
    
    # extraigo las intencidades y las sumas
    intensity = mat[:, 0]
    
    suma = (mat[:, 1])
    

    # Plot
    plt.figure(figsize=(8, 6))  # Optional: Adjust the figure size
    if modo == 'welch':
        plt.plot(intensity, suma, marker='o', linestyle='-', color='b', label='sumas de welch vs Intensidad')
    else:
        plt.plot(intensity, suma, marker='o', linestyle='-', color='r', label='sumas del Periodograma vs Intensidad')
    _, LG_exp = exponential_regression(intensity, suma)
    plt.plot(intensity, LG_exp, 'g--', label='Regresión exponencial')

    # Add labels and title
    plt.xlabel('Intensidad')
    if modo == 'welch':
        plt.ylabel('sumas de Welch')
    else:
        plt.ylabel('sumas de Periodograma')

    plt.title(f"Promediado {metodo}")

    # Add grid
    plt.grid(True)

    # Add legend
    plt.legend()

    # Show plot
    plt.show()

def main():
    #fs de las realizaciones
    fs = 48000
    lista_paths=[]
    directory = os.fsencode('data_avg_N1')
    for file in os.listdir(directory):
            filename = os.fsdecode(file)        
            filepath = os.path.join(directory, os.fsencode(filename))
            name = os.fsdecode(filepath)
            lista_paths.append(name)
            
     # calculo la suma del welch
    PSD_sum_mat_amp= suma_welch_peri(lista_paths[0])
    PSD_sum_mat_both= suma_welch_peri(lista_paths[1])
    PSD_sum_mat_homo= suma_welch_peri(lista_paths[2])
    PSD_sum_mat_var= suma_welch_peri(lista_paths[3])

    #almaceno las matrices 
    matrices_suma = {"amp":PSD_sum_mat_amp,"both":PSD_sum_mat_both,"homo":PSD_sum_mat_homo,"var":PSD_sum_mat_var}
    for key, value in matrices_suma.items():
        graf_intensidad_vs_welch_suma(value[0], key,modo = 'welch')
        graf_intensidad_vs_welch_suma(value[1], key,modo = 'periodogrma') 



if __name__ == '__main__':
    main()




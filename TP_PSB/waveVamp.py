import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import welch
from dat_hea_reader import *
from eeg_avg import *
import scipy.signal as sig
import os
from typing import List
import re

def find_index_peak(t_arr: np.ndarray, 
                   signal: np.ndarray, 
                   t_min: int, 
                   t_max: int,
                   take_abs: bool=False,
                   plot: bool=True,
                   type_str: str='') -> Tuple[int, float]:
    """
    Finds the largest amplitude of a signal within t_min < t < t_max

    Args:
        t_arr (np.ndarray): time array
        signal (np.ndarray): signal array
        t_min (int): initial time to analyze
        t_max (int): final time to analyze
        take_abs (bool): indicates whether or not to analyze the 
        plot (bool, optional):. Defaults to True.
        type_str (str, optional): Indicates how the data was averaged

    Returns:
        tuple[int, float]: idx_peak (time index of found peak), amplitude (max - min in segment)
    """
    original_signal = np.copy(signal)
    if take_abs:
        signal = np.abs(signal - signal.mean()) # make zero-mean and take abs
    
    fs = int(len(t_arr)/t_arr[len(t_arr) - 1]) # not necessarily in hertz! 
    idx_min = int(fs*t_min)
    idx_max = int(fs*t_max)
    idx_peak = np.argmax(signal[idx_min: idx_max]) + idx_min
    amp_peak = original_signal[idx_peak]
    
    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t_arr, original_signal, label='Signal')
        ax.plot(t_arr[idx_min: idx_max], original_signal[idx_min: idx_max], label='Segment')
        ax.plot(t_arr[idx_peak], original_signal[idx_peak], 'ro', label='peak')
        ax.set_xlabel('t [ms]')
        ax.set_ylabel('v [nv]')
        ax.set_title(f'Onda 5 a SPL=100 [dB] con promediado {type_str}')
        plt.legend()
        plt.show()

    return idx_peak, amp_peak


def trace_peak(t_arr: np.ndarray,
               signal_df: pd.DataFrame,
               init_idx: int, 
               left_interval: float=0.5,
               right_interval: float=1.0,
               plot: bool=False) -> Tuple[List[int], List[int]]:
    """
    Finds relative maxima on a sequence of signals by tracking
    an initial peak as it moves in latency
    Args:
        t_arr (np.ndarray): time array
        signal_mat (np.ndarray): 2D array where each row is a time series
        init_idx (int): initial time seed to look ar
        left_interval (float): Start of time interval relative (negatively) to an index
        right_interval (float): Start of time interval relative (positively) to an index
        plot (bool, optional):. Defaults to True.
    Returns:
        List[int]: List of indeces where the peak happens
    """
    fs = int(len(t_arr) / t_arr[len(t_arr) - 1])  # not necessarily in hertz! 
    idx_min = int(fs*left_interval)
    idx_max = int(fs*right_interval)

    idx_peak_list = []
    amp_peak_list = []
    prev_idx = init_idx
    for column in signal_df.columns:
        signal = signal_df[column]
        new_idx = np.argmax(signal[prev_idx - idx_min : prev_idx + idx_max]) + prev_idx - idx_min

        if plot:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(t_arr, signal, label='Signal')
            ax.plot(t_arr[prev_idx - idx_min : prev_idx + idx_max], 
                    signal[prev_idx - idx_min : prev_idx + idx_max],
                    label='Segment')
            ax.plot(t_arr[new_idx], signal[new_idx], 'ro', label='peak')
            ax.set_xlabel('t [ms]')
            ax.set_ylabel('v [nv]')
            ax.set_title(f"SPL={column} [dB]")
            plt.legend()
            plt.show()

        idx_peak_list.append(new_idx)
        amp_peak_list.append(signal[new_idx])
        prev_idx = new_idx

    return idx_peak_list, amp_peak_list


def exponential_regression(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Uses least squares to fit an exponential of the form
                   y = A*e^(b*x)

    Args:
        X (np.ndarray): independent variable, 1D
        Y (np.ndarray): dependent variable, also 1D 

    Returns:
        np.ndarray: A, B, Y_est
    """
    # exponential regression
    mat_LS = np.column_stack((X, np.ones_like(X)))
    theta = np.linalg.lstsq(mat_LS, np.log(Y), rcond=None)[0].squeeze()
    print(theta)
    reg_LG = np.exp(theta[0]*X + theta[1])
    return theta, reg_LG


def waveV_LG_estimation(df: pd.DataFrame,
                        t: np.ndarray,
                        output_dir: str,
                        prom_type: str='') -> None:
    """
    Estimates loudness growth on a subject by analizyng the decaying latency
    of the wave V peak in a series of ERP measured at different SPL levels

    Args:
        df (pd.DataFrame): Each column is a series of time samples of an ERP signal 
        for a given SPL, where the SPL level is itself the column name
        t: time samples array, in miliseconds
        outputdir: directory to save the figures
        prom_type (str): Indicates how the data was averaged

    returns: nothing, but it plots
    """
     # Find first peak
    first_avg = df['100']
    idx_peak_wave5, peak_amp = find_index_peak(t, first_avg, 5, 10, 
                                                take_abs=False, plot=True, type_str=prom_type)

    # get remaining signals and trace the peak location
    check_columns = df.columns[1:]
    mat_to_trace  = df[check_columns]

    # trace
    peak_indeces, peak_amps = trace_peak(t, mat_to_trace, idx_peak_wave5, 
                                            left_interval=0.5, right_interval=2.5, plot=True)

    # merge with the data from the original signal, and sort
    columns_arr = np.array([int(col) for col in df.columns])
    sort_idx = columns_arr.argsort()
    columns_arr = columns_arr[sort_idx]

    # define your estimate
    peak_indeces = np.array([idx_peak_wave5] + peak_indeces)
    peak_indeces = peak_indeces[sort_idx]
    est_LG = 1/peak_indeces
    est_LG /= est_LG.max()

    # regression
    _, reg_LG = exponential_regression(columns_arr, est_LG)

    # plot
    fig, ax = plt.subplots()
    ax.plot(columns_arr, est_LG, label='LG')
    ax.plot(columns_arr, reg_LG, label='Reg. exponencial')
    ax.set_xlabel('SPL [dB]')
    ax.set_ylabel('LG [uu. aa.]')
    ax.set_xticks(columns_arr)
    ax.set_title(f"Percepción estimada con promediado {prom_type}")
    plt.legend()
    plt.show()

    fig.savefig(rf"{output_dir}/est_LG_{prom_type}.png")
    plt.close()


def main():
    path_realizacion = r'datos_ejemplo_señales\N1_evoked_raw_100_F1_R1'
    fs, _, comments = read_trials(path_realizacion)
    tr_len = comments['Trial Length (samples)']
    t = np.linspace(0, tr_len/fs, tr_len)*1000

    prom_pattern = re.compile(r'(promedios_)(\w*)')
    prom_type_dict = {'amp': 'ponderado por amplitud',
                      'both': 'ponderado por amplitud y varianza',
                      'var': 'ponderado por varianza',
                      'homo': 'homogéneo'}

    directory = os.fsencode('data_avg_N1')
    output_img_dir = 'estimaciones_LG_waveV'
    for file in os.listdir(directory):
        # read file with averaged data
        filename = os.fsdecode(file)        
        filepath = os.path.join(directory, os.fsencode(filename))
        name = os.fsdecode(filepath)
        df = pd.read_csv(name, header=0)
        prom_key = re.search(prom_pattern, name).group(2)
        prom_type = prom_type_dict[prom_key]

        waveV_LG_estimation(df, t, output_img_dir, prom_type)

if __name__ == '__main__':
    main()
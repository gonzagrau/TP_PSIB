import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import welch
from dat_hea_reader import *
from eeg_avg import *
import scipy.signal as sig
import os
from typing import List

def find_index_peak(t_arr: np.ndarray, 
                   signal: np.ndarray, 
                   t_min: int, 
                   t_max: int,
                   take_abs: bool=False,
                   plot: bool=True) -> tuple[int, float]:
    """
    Finds the largest amplitude of a signal within t_min < t < t_max

    Args:
        t_arr (np.ndarray): time array
        signal (np.ndarray): signal array
        t_min (int): initial time to analyze
        t_max (int): final time to analyze
        take_abs (bool): indicates whether or not to analyze the 
        plot (bool, optional):. Defaults to True.

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
    
    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t_arr, original_signal, label='Signal')
        ax.plot(t_arr[idx_min: idx_max], original_signal[idx_min: idx_max], label='Segment')
        ax.plot(t_arr[idx_peak], original_signal[idx_peak], 'ro', label='peak')
        ax.set_xlabel('t [ms]')
        ax.set_ylabel('v [nv]')
        plt.legend()
        plt.show()

    return idx_peak


def trace_peak(t: np.ndarray, 
               signal_mat: np.ndarray, 
               init_idx: int, 
               width: int,
               plot: bool) -> List[int]:
    """
    Finds relative maxima on a sequence of signals by tracking
    an initial peak as it moves in latency
    Args:
        t_arr (np.ndarray): time array
        signal_mat (np.ndarray): 2D array where each row is a time series
        init_idx (int): initial time seed to look ar
        plot (bool, optional):. Defaults to True.
    Returns:
        List[int]: List of indeces where the peak happens
    """
    idx_peak_list = []
    prev_idx = init_idx
    for i in range(signal_mat.shape[1]):
        signal = signal_mat[:, i]
        print(f"{i=}, {prev_idx=}")
        new_idx = np.argmax(signal[prev_idx - width//2 : prev_idx + width//2]) + prev_idx - width//2

        if plot:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(t, signal, label='Signal')
            ax.plot(t[new_idx - width//2 : new_idx + width//2], 
                    signal[new_idx - width//2 : new_idx + width//2],
                    label='Segment')
            ax.plot(t[new_idx], signal[new_idx], 'ro', label='peak')
            ax.set_xlabel('t [ms]')
            ax.set_ylabel('v [nv]')
            plt.legend()
            plt.show()

        idx_peak_list.append(new_idx)
        prev_idx = new_idx

    return idx_peak_list


def main():
    path_realizacion = r'datos_ejemplo_señales\N1_evoked_raw_100_F1_R1'
    fs, trials, comments = read_trials(path_realizacion)
    tr_len = comments['Trial Length (samples)']
    t = np.linspace(0, tr_len/fs, tr_len)*1000

    dataframes = []
    directory = os.fsencode('data_avg_N1')
    for file in os.listdir(directory):
            filename = os.fsdecode(file)        
            filepath = os.path.join(directory, os.fsencode(filename))
            name = os.fsdecode(filepath)
            df = pd.read_csv(name)
            dataframes.append(df)

    amp_df = dataframes[0]
    first_avg = amp_df['100']

    # Find first peak
    idx_peak_wave5 = find_index_peak(t, first_avg, 5, 10, take_abs=False, plot=True)
    print(f"{idx_peak_wave5=}")
    
    # get remaining signals and trace
    check_columns = amp_df.columns[1:]
    mat_to_trace  = df[check_columns].to_numpy()

    # trace
    peak_indeces = trace_peak(t, mat_to_trace, idx_peak_wave5, width=10, plot=True)
    print(peak_indeces)


    


if __name__ == '__main__':
    main()
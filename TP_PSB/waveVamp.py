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
                   plot: bool=True) -> Tuple[int, float]:
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
    amp_peak = original_signal[idx_peak]
    
    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t_arr, original_signal, label='Signal')
        ax.plot(t_arr[idx_min: idx_max], original_signal[idx_min: idx_max], label='Segment')
        ax.plot(t_arr[idx_peak], original_signal[idx_peak], 'ro', label='peak')
        ax.set_xlabel('t [ms]')
        ax.set_ylabel('v [nv]')
        plt.legend()
        plt.show()

    return idx_peak, amp_peak


def trace_peak(t_arr: np.ndarray,
               signal_mat: np.ndarray,
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
    for i in range(signal_mat.shape[1]):
        signal = signal_mat[:, i]
        new_idx = np.argmax(signal[prev_idx - idx_min : prev_idx + idx_max]) + prev_idx - idx_min

        if plot:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(t_arr, signal, label='Signal')
            ax.plot(t_arr[new_idx - idx_min : new_idx + idx_max], 
                    signal[new_idx - idx_min : new_idx + idx_max],
                    label='Segment')
            ax.plot(t_arr[new_idx], signal[new_idx], 'ro', label='peak')
            ax.set_xlabel('t [ms]')
            ax.set_ylabel('v [nv]')
            ax.set_title(f"column {i}")
            plt.legend()
            plt.show()

        idx_peak_list.append(new_idx)
        amp_peak_list.append(signal[new_idx])
        prev_idx = new_idx

    return idx_peak_list, amp_peak_list


def main():
    path_realizacion = r'datos_ejemplo_señales\N1_evoked_raw_100_F1_R1'
    fs, trials, comments = read_trials(path_realizacion)
    tr_len = comments['Trial Length (samples)']
    t = np.linspace(0, tr_len/fs, tr_len)*1000

    directory = os.fsencode('data_avg_N1')
    for file in os.listdir(directory):
            filename = os.fsdecode(file)        
            filepath = os.path.join(directory, os.fsencode(filename))
            name = os.fsdecode(filepath)
            df = pd.read_csv(name, header=0)

            first_avg = df['100']

            # Find first peak
            idx_peak_wave5, peak_amp = find_index_peak(t, first_avg, 5, 10, take_abs=False, plot=True)
            print(f"{idx_peak_wave5=}")

            # get remaining signals and trace
            check_columns = df.columns[1:]
            mat_to_trace  = df[check_columns].to_numpy()

            # trace
            peak_indeces, peak_amps = trace_peak(t, mat_to_trace, idx_peak_wave5, 
                                                 left_interval=1, right_interval=2, plot=True)

            # merga data and plot
            peak_amps = np.array([peak_amp] + peak_amps)
            peak_amps /= peak_amps.max()
            plt.plot(df.columns, peak_amps)
            plt.title(f"Loudness growth using {name}")
            plt.show()


if __name__ == '__main__':
    main()
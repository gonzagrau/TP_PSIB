import wfdb
from typing import Tuple
import re
import numpy as np


def parse_comments(comments: list[str]) -> dict:
    """
    Reads the 'comments' section of the trial .hea file into a dictionary
    """
    INT_KEYS = ['SPL', 'Stim Freq (kHz)', 'Trial Length (samples)']
    comments = comments[0]
    pattern = r'<([^>]+)>: ([^<]+)'
    matches = re.findall(pattern, comments)
    parsed_dict = {key.strip(): value.strip() for key, value in matches}
    
    for key in INT_KEYS:
        try:
            parsed_dict[key] = int(parsed_dict[key])
        except KeyError:
            continue
        except ValueError:
            raise ValueError(f"Invalid data type for {key} : {parsed_dict[key]}")
        
    return parsed_dict


def read_trials(filepath: str) -> Tuple[int, np.ndarray, dict]:
    """
    Reads .dat and .hea files, and the

    Args:
        filepath (str): filepath WITHOUT the extension

    Returns:
        Tuple[int, np.ndarray, dict]: sample frequency, trials matrix, comments dict
    """

    record = wfdb.rdrecord(filepath)
    fs = record.fs
    ABR_raw = record.p_signal[:, 0]
    comments = parse_comments(record.comments)
    tr_len = comments['Trial Length (samples)']
    sig_len = record.sig_len
    
    #funcion previa que no checkea outliers, BORRAR
    #trials = [ABR_raw[i: i+tr_len] for i in range(0, sig_len - tr_len, tr_len)]

#create an empty list
    trials = []
    for i in range(0, sig_len - tr_len, tr_len):
        #separate a trial 
        trial = ABR_raw[i:i+tr_len]
        #check if the trail has some outlier data, if it does the trial is discarted
        if any(abs(point) > 50000 for point in trial):
            continue
        else:
            trials.append(trial)
            
    if not trials:
            raise ValueError("Todos los trials estan fuera del limite, ende Trials esta vacia")
    
    trials = np.array(trials)

    return fs, trials, comments
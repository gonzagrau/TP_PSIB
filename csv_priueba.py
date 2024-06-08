import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dat_hea_reader import *
from eeg_avg import *
import scipy.signal as sig
import os


# Create a NumPy array
a = np.array([[1.0000, 6.999, 4.88], [2.77, 4.55, 8.4], [3.7, 9.8, 1.8]])

# Define row headers
row_headers = np.array([["Row1"], ["Row2"], ["Row3"]])

# Concatenate row headers with data
data_with_headers = np.hstack((row_headers, a.astype(str)))

# Save to CSV
np.savetxt("preueba_with_row_headers.csv", data_with_headers, delimiter=";", fmt='%s', header="RowHeader;Column1;Column2;Column3", comments='')

print("Array with row headers saved to preueba_with_row_headers.csv")
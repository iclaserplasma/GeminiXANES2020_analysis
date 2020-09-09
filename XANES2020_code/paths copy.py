# Fill in the following line with the path to the data folder
# (that is, the folder containing MIRAGE and Calibrations as subfolders)
# then uncomment it and copy this file to paths.py

# DATA_FOLDER = r'E:\Streeter2019'
import os
DATA_FOLDER = r'/Users/streeter/BoxSync/Experiments/GeminiXANES2020/MIRAGE'

# path for calibrations assumed as relative to DATA_FOLDER ../ProcessedCalibrations
CAL_DATA = os.path.join(os.path.split(DATA_FOLDER[:-1])[0],'ProcessedCalibrations')
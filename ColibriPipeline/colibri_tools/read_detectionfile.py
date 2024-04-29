"""
Filename:   read_detectionfile.py
Author(s):  Peter Quigley
Contact:    pquigley@uwo.ca
Created:    
Updated:    
    
Description:


Usage:

"""

# Module Imports
import os,sys
import re
import argparse
import subprocess
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime, timedelta, timezone
from getStarHour import getStarHour
from astropy.time import Time
from astropy.coordinates import Angle,EarthLocation,SkyCoord
from astropy.io.fits import Header
from astropy import units
from astropy.coordinates import AltAz
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.collections import PolyCollection
from matplotlib.dates import DateFormatter
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter, MaxNLocator
from PIL import Image

# Custom Imports


#-------------------------------global vars-----------------------------------#

# Path variables
BASE_PATH = Path('D:/')
DATA_PATH = BASE_PATH / 'ColibriData'
IMGE_PATH = BASE_PATH / 'ColibriImages'
ARCHIVE_PATH = BASE_PATH / 'ColibriArchive'
WEATHER_PATH = BASE_PATH / 'Logs' / 'Weather'

# Timestamp format
OBSDATE_FORMAT = '%Y%m%d'
MINDIR_FORMAT  = '%Y%m%d_%H.%M.%S.%f'
TIMESTAMP_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'
BARE_FORMAT = '%Y-%m-%d_%H%M%S_%f'
CLOCK_FORMAT   = '%H:%M:%S'
ACPLOG_STRP    = '%a %b %d %H:%M:%S %Z %Y'

# Regex patterns
MINDIR_REGEX = re.compile(r'\d{8}_\d{2}\.\d{2}\.\d{2}\.\d{3}')
NPY_PATTERN = r'(\d{8}_\d{2}.\d{2}.\d{2}.\d{3})'
DET_PATTERN = r'det_(\d{4}-\d{2}-\d{2}_\d{8}_\d{6})'


# Site longitude/latitude
SITE_LAT  = 43.1933116667
SITE_LON = -81.3160233333
SITE_HGT = 224
SITE_LOC  = EarthLocation(lat=SITE_LAT,
                         lon=SITE_LON,
                         height=SITE_HGT)

# Weather Files
WEATHER_CSV = WEATHER_PATH / 'weather.csv'
WEATHER_HEADERS = ['unix_time', 'temp', 'humidity', 'wind_speed', 'wind_direction',
                   'rain_value', 'sky_temp', 'ground_temp', 'alert', 'polaris_mag']


#----------------------------------class--------------------------------------#


#--------------------------------functions------------------------------------#

def analyzeDetectionMeta(filepath):
    """
    A general function to read the metadata from a detection file.

    Parameters
    ----------
    filepath : pathlib.Path
        Path to the detection file.

    Returns
    -------
    timestamp : str
        Timestamp of the detection.
    RA : str
        Right ascension of the detection.
    DEC : str
        Declination of the detection.
    telescope : str
        Telescope used to make the detection.
    significance : float
        Significance of the detection.
    
    """

    # Open file to read timestamp, RA/DEC, telescope, and significance
    with open(filepath, 'r') as det:
        lines = det.readlines()
        for i,line in enumerate(lines):
            
            # TODO: convert this to regex matching
            if i == 6: # RA/Dec
                try:
                    RA  = line.split(" ")[-2].strip("\n")
                    DEC = line.split(" ")[-1].strip("\n")
                except:
                    print(f"WARNING: RA/Dec were not found in {filepath}!")
                    RA,DEC = "",""
            elif i == 7:
                try:
                    timestamp = line.split(" ")[-1].strip("\n")
                except:
                    print(f"WARNING: Timestamp was not found in {filepath}!")
                    timestamp = ""
            elif i == 8:
                try:
                    telescope = line.split(" ")[-1].strip("\n")
                except:
                    print(f"WARNING: Telescope name was not found in {filepath}!")
                    telescope = ""
            elif i == 10:
                try:
                    sigma = line.split(':')[1].strip("\n")
                except:
                    print(f"WARNING: Significance was not found in {filepath}!")
                    sigma = ""
            elif i > 10:
                break

    # Return the relevant information about this detection file
    print(f"Read matched detection: {telescope} {timestamp}")
    return telescope, timestamp, RA, DEC, sigma


def readDetTimestamp(det_file):
    """
    Read minute dir of origin from det file

        Parameters:
            det_file (str): Path to det file.

        Returns:
            timestamp (str): Timestamp of det file.
    
    """

    # Read the 5th line of the det file
    with open(det_file, 'r') as f:
        for i in range(5):
            event_file = f.readline()

    minute_dir = re.search(NPY_PATTERN, event_file).group(1)
    return minute_dir


def analyzeDetectionData(filepath):
    """
    Analyze the data of a detection file.
    
    Parameters
    ----------
    filepath : pathlib.Path
        Path to the detection file.
        
    Returns
    -------
    lightcurve_raw : np.ndarray
        Array of the raw lightcurve data.
    
    TODO: update to use pandas
    """

    # Read in the detection file lightcurve and strip filenames
    lightcurve_data = np.loadtxt(str(filepath),dtype='object')
    lightcurve_raw  = lightcurve_data[:,[1,2]].astype(float)

    # Return float array with columns
    # [0] = seconds; [1] = flux; [2] = conv_flux
    return lightcurve_raw

#----------------------------------main---------------------------------------#

if __name__ == '__main__':
    pass
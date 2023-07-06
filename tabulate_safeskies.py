"""
Filename:   tabulate_safeskies.py
Author(s):  Peter Quigley
Contact:    pquigley@uwo.ca
Created:    July 5, 2023
Updated:    July 5, 2023
    
Usage: python tabulate_safeskies.py
"""

# Module Imports
import os,sys
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.coordinates import EarthLocation

# Custom Imports
import colibri_tools as ct


#-------------------------------global vars-----------------------------------#

# Path variables
BASE_PATH = Path('D:/')
DATA_PATH = BASE_PATH / 'ColibriData'
IMGE_PATH = BASE_PATH / 'ColibriImages'
ARCHIVE_PATH = BASE_PATH / 'ColibriArchive'
WEATHER_PATH = BASE_PATH / 'Logs' / 'Weather'

# Weather Files
WEATHER_CSV = WEATHER_PATH / 'weather.csv'
WEATHER_HEADERS = ['unix_time', 'temp', 'humidity', 'wind_speed', 'wind_direction',
                   'rain_value', 'sky_temp', 'ground_temp', 'alert', 'polaris_mag']


# Timestamp format
OBSDATE_FORMAT = '%Y%m%d'
MINDIR_FORMAT  = '%Y%m%d_%H.%M.%S.%f'
TIMESTAMP_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'
BARE_FORMAT = '%Y-%m-%d_%H%M%S_%f'

# Site longitude/latitude
SITE_LAT  = 43.1933116667
SITE_LON = -81.3160233333
SITE_HGT = 224
SITE_LOC  = EarthLocation(lat=SITE_LAT,
                         lon=SITE_LON,
                         height=SITE_HGT)


# Verbose print statement
verboseprint = lambda *a, **k: None


#--------------------------------functions------------------------------------#

def readWeatherCSV(csv_file):
    """
    
    """


    if not csv_file.exists():
        print('WARNING: Weather CSV file does not exist.')
        return pd.DataFrame([], columns=WEATHER_HEADERS)
    
    # Read in CSV file
    weather_df = pd.read_csv(csv_file, names=WEATHER_HEADERS)

    return weather_df

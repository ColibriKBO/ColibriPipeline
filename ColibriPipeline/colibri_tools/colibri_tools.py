"""
Filename:   colibri_tools.py
Author(s):  Peter Quigley
Contact:    pquigley@uwo.ca
Created:    July 5, 2023
Updated:    July 5, 2023
    
Usage: import colibri_tools as ct
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

class ErrorTracker(object):
    """
    Indicates if any errors or warnings occured during the running of the program.
    """

    def __init__(self):

        self.errors = []


    def addError(self, error_msg):

        self.errors.append(error_msg)
        print(error_msg)


#--------------------------------functions------------------------------------#

#############################
## Formatting Dates
#############################

def hyphonateDate(obsdate):
    """
    Change the date format from YYYYMMDD to YYYY-MM-DD.

    Parameters
    ----------
    obsdate : str
        Date in YYYYMMDD format.

    Returns
    -------
    obsdate : str
        Date in YYYY-MM-DD format.

    """

    # Convert the date to a datetime object
    obsdate = datetime.strptime(obsdate, OBSDATE_FORMAT)

    # Convert the date to a hyphonated string
    obsdate = obsdate.strftime('%Y-%m-%d')

    return obsdate


def slashDate(obsdate):
    """
    Change the date format from YYYYMMDD to YYYY/MM/DD.

    Parameters
    ----------
    obsdate : str
        Date in YYYYMMDD format.

    Returns
    -------
    obsdate : str
        Date in YYYY/MM/DD format.

    """

    # Convert the date to a datetime object
    obsdate = datetime.strptime(obsdate, OBSDATE_FORMAT)

    # Convert the date to a hyphonated string
    obsdate = obsdate.strftime('%Y/%m/%d')

    return obsdate


#############################
## File/Directory Comprehension
#############################

def getMinDirTimes(data_dir_path):
    """
    Get the timestamps of all minute directories in the obsdate directory.

    Parameters
    ----------
    data_dir_path : pathlib.Path
        Path to the obsdate directory.

    Returns
    -------
    directory_times : list
        List of datetime objects corresponding to the minute directories.

    """

    # Get all minute directories in the given data directory
    minute_dirs = [folder.name+'000' for folder in data_dir_path.iterdir() \
                   if ((folder.is_dir()) and (folder.name != 'Dark'))]
    minute_dirs.sort()

    # Get timestamps from the directory names
    directory_times = []
    for timestamp in minute_dirs:
        dir_datetime = datetime.strptime(timestamp, MINDIR_FORMAT)
        directory_times.append(dir_datetime)
        
    return directory_times


def getMedstackTimes(archive_dir_path):
    """
    Get the timestamps of all medstacked images in the obsdate archive.

    Parameters
    ----------
    archive_dir_path : pathlib.Path
        Path to the obsdate directory.

    Returns
    -------
    directory_times : list
        List of datetime objects corresponding to the stacked minute.
        
    """

    # Get timestamps from the medstack names
    directory_times = []
    for filename in archive_dir_path.glob('*medstacked.fits'):
        # Get the timestamp from the filename using regex
        timestamp = MINDIR_REGEX.match(filename.name).group() + '000'

        # Convert the timestamp to a datetime object and add to list
        file_datetime = datetime.strptime(timestamp, MINDIR_FORMAT)
        directory_times.append(file_datetime)
        
    return directory_times


def analyzeDetectionMeta(filepath):
    """
    Analyze the meta/header data of a detection file.

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


def readDetectionSigma(filepath):
    """
   Read sigma from det.txt file

   Parameters
   ----------
   filepath : path type
       Txt file of the detections.
   Returns
   -------
   sigma : float
       Event significance.

   """
    with open(filepath, 'r') as f:
        
        #loop through each line of the file
        for i, line in enumerate(f):
            
            if i==10:
                try:
                    sigma = float(line.split(':')[1])
                except ValueError: # occurs in the case of ARTIFICIAL
                    sigma = 0.

                break


    return (sigma)


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


#############################
## Time Functions
#############################

def twilightTimesJD(julian_date, site=(SITE_LAT,SITE_LON)):
    '''
    M. Mazur's code to calculate sunrise and sunset time for specified JD time
    See: https://en.wikipedia.org/wiki/Sunrise_equation

    Parameters
    ----------
    julian_date : float
        Julian Date (2460055.24633).
    site : TYPE, optional
        Observatory location. The default is [43.0,-81.0].

    Returns
    -------
    Jrise : float
        Sunrise time.
    Jset : float
        Sunset time.

    '''

    n = np.floor(julian_date - 2451545.0 + 0.0008)
    Jstar = n - (site[1]/360.0)
    M = (357.5291 + 0.98560028 * Jstar) % 360.0
    C = 1.9148*np.sin(np.radians(M)) + 0.02*np.sin(2*np.radians(M)) + 0.0003*np.sin(3*np.radians(M))
    lam = (M + C + 180.0 + 102.9372) % 360.0
    Jtransit = 2451545.0 + Jstar + 0.0053*np.sin(np.radians(M)) - 0.0069*np.sin(2*np.radians(lam))
    sindec = np.sin(np.radians(lam)) * np.sin(np.radians(23.44))
    cosHA = (np.sin(np.radians(-12.0)) - (np.sin(np.radians(site[0]))*sindec)) / (np.cos(np.radians(site[0]))*np.cos(np.arcsin(sindec)))
    Jrise = Jtransit - (np.degrees(np.arccos(cosHA)))/360.0
    Jset = Jtransit + (np.degrees(np.arccos(cosHA)))/360.0

    return Jrise, Jset


#############################
## Coordinate Functions
#############################

def calculateAlt(ra, dec, time):
    
    # Convert RA and Dec to SkyCoord object
    coord = SkyCoord(ra, dec, unit=(units.deg, units.deg))
    
    # Calculate the altitude at the given time
    alt = coord.transform_to(AltAz(obstime=time, location=SITE_LOC)).alt
    
    return alt.degree


def calculateAirmass(alt):
    
    # Calculate the airmass
    airmass = 1/np.cos(np.radians(90-alt))
    
    return airmass


#############################
## Weather Functions
#############################

def readWeatherLog(weather_csv_file):
    """
    
    """


    if not weather_csv_file.exists():
        print('WARNING: Weather CSV file does not exist.')
        return pd.DataFrame([], columns=WEATHER_HEADERS)
    
    # Read in CSV file
    weather_df = pd.read_csv(weather_csv_file, names=WEATHER_HEADERS)

    return weather_df


#############################
## Astrometry.Net Functions
#############################

def getSolution(image_file, save_file, order):
    '''send request to solve image from astrometry.net
    input: path to the image file to submit, filepath to save the WCS solution header to, order of soln
    returns: WCS solution header'''
    from astroquery.astrometry_net import AstrometryNet
    #astrometry.net API
    ast = AstrometryNet()
    
    #key for astrometry.net account
    ast.api_key = 'vbeenheneoixdbpb'    #key for Rachel Brown's account (040822)
    wcs_header = ast.solve_from_image(image_file, crpix_center = True, tweak_order = order, force_image_upload=True)

    #save solution to file
    if not save_file.exists():
            wcs_header.tofile(save_file)
            
    return wcs_header

def getLocalSolution(image_file, save_file, order):
    """
    Astrometry.net must be installed locally to use this function. It installs under WSL.
    To use the local solution, you'll need to modify call to the function somewhat.
    This function will write the new fits file w/ plate solution to a file with the name save_file in the
    tmp directory on the d: drive.
    The function will return wcs_header. Alternatively, you could comment out those lines and read it from
    the pipeline.
    """
    try:
        # -D to specify write directory, -o to specify output base name, -N new-fits-filename
        print(image_file)
        # print(save_file.split(".")[0])
        print(save_file.split(".fits")[0])

        cwd = os.getcwd()
        os.chdir('d:\\')

        #p = subprocess.run('wsl time solve-field --no-plots -D /mnt/d/tmp -O -o ' + save_file.split(".")[0] + ' -N ' + save_file + ' -t ' + str(order) + ' --scale-units arcsecperpix --scale-low 2.2 --scale-high 2.6 ' + image_file)
        p = subprocess.run('wsl time solve-field --no-plots -D /mnt/d/tmp -O -o ' + save_file.split(".fits")[0] + ' -N ' + save_file + ' -t ' + str(order) + ' --scale-units arcsecperpix --scale-low 2.2 --scale-high 2.6 ' + image_file)
        
        os.chdir(cwd)
        print(os.getcwd())

        #wcs_header = Header.fromtextfile('d:\\tmp\\' + save_file.split('.')[0] + '.wcs')

        wcs_header = Header.fromfile('d:\\tmp\\' + save_file.split(".fits")[0] + '.wcs')

    except:
        pass

    return wcs_header
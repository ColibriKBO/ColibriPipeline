# -*- coding: utf-8 -*-
"""
Filename:   timeline.py
Author(s):  Roman Akhmetshyn, Peter Quigley
Contact:    pquigley@uwo.ca
Created:    Tue Nov 22 14:39:39 2022
Updated:    Mon May  1 14:35:58 2023
    
Description:
Create a series of diagnostic plots from ACP logs to help summarize the
nightly operations of the Colibri project. Runs only on Greenbird.
"""

# Module Imports
import os,sys
import re
import csv
import math
import shutil
import argparse
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import seaborn as sns
import numba as nb
from datetime import datetime, timedelta, date
from getStarHour import getStarHour
from astropy.time import Time
from astropy.coordinates import Angle,EarthLocation,SkyCoord
from astropy import units
from astropy.coordinates import AltAz
from scipy import interpolate
from pathlib import Path
from astropy import time
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.collections import PolyCollection
from matplotlib.dates import DateFormatter
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator

# Custom Script Imports
import colibri_image_reader as cir
import colibri_photometry as cp


#-------------------------------global vars-----------------------------------#

# Site longitude/latitude
SITE_LAT  = 43.1933116667
SITE_LON = -81.3160233333
SITE_HGT = 224
SITE_LOC  = EarthLocation(lat=SITE_LAT,
                         lon=SITE_LON,
                         height=SITE_HGT)


# Noteable log patterns
LOG_PATTERNS = ['starts', 'Weather unsafe!', 'Dome closed!', 'Field Name:', 'Sunrise JD', 'Sunset JD']
OBSDATE_FORMAT = '%Y%m%d'
CLOCK_FORMAT   = '%H:%M:%S'
ACPLOG_STRP    = '%a %b %d %H:%M:%S %Z %Y'
MINUTEDIR_STRP = '%Y%m%d_%H.%M.%S.%f'
TIMESTAMP_STRP = '%Y-%m-%dT%H:%M:%S.%f'

# Directory structure
BASE_PATH  = Path('/', 'D:')
CLOUD_PATH = BASE_PATH / 'Logs' / 'Weather' / 'Weather'

# Plotting constants
TELESCOPE_ID = {"REDBIRD" : 1, "GREENBIRD" : 2, "BLUEBIRD" : 3}
COLOURMAPPING = {"REDBIRD" : "r", "GREENBIRD" : "#66ff66", "BLUEBIRD" : "b"}


#----------------------------------class--------------------------------------#

class Telescope:

    def __init__(self, name, colour, base_path, obs_date):
        
        # Telescope identifiers
        self.name   = name
        self.colour = colour

        # Error logging
        self.errors = []

        # Path variables
        self.base_path = Path(base_path)
        self.log_path  = self.base_path.joinpath("Logs", "ACP", f"{obs_date}-ACP.log")
        self.data_path = self.base_path.joinpath("ColibriData", obs_date)
        sensitivity_dir = self.base_path.joinpath("ColibriArchive", 
                                                  "{}_diagnostics".format(obs_date), 
                                                  "Sensitivity")
        
        # Check if log and data directories exist
        self.log_exists  = True if self.log_path.exists() else False
        self.data_exists = True if self.data_path.exists() else False

        # Get important lines from the appropriate log
        if self.log_exists:
            self.log_lines = getImportantLines(self.log_path, LOG_PATTERNS)
        else:
            self.addError("ERROR: {} has no valid ACP log for {}!".format(name, obs_date))
            self.log_lines = []

        # Get observation minute directory times
        if self.data_exists:
            self.data_times = getDataTimes(self.data_path)
        else:
            self.addError("ERROR: {} has no data for {}!".format(name, obs_date))
            self.data_times = []

        # Get sensitivity data
        if sensitivity_dir.exists():
            try:
                self.sensitivity_path = [folder for folder in sensitivity_dir.iterdir() if folder.is_dir()][0]
            except:
                self.addError("ERROR: {} has no sensitivity data for {}!".format(name, obs_date))
                self.sensitivity_path = None
        else:
            self.addError("ERROR: {} has no sensitivity data for {}!".format(name, obs_date))
            self.sensitivity_path = None


    def addError(self, error_msg):

        self.errors.append(error_msg)
        print(error_msg)


    def analyzeSensitivity(self):

        # Check if sensitivity data exists
        if self.sensitivity_path is None:
            print("WARNING: {} has no sensitivity data. Skipping...\n".format(self.name))
            return
        
        # Load sensitivity data in pandas dataframe
        column_names = [ 'X', 'Y', 'ra', 'dec', 'GMAG',
                        'Gaia_RA' ,'Gaia_dec', 'Gaia_B-R',
                        'med' ,'std', 'SNR']
        star_table = pd.read_csv(self.sensitivity_path.glob("starTable*.txt")[0],
                                 names=column_names, sep=' ',
                                 header=0, index_col=0)
        
        # Read in data from sample star light curves
        # TODO: Guard against directory with no star txt files
        star_txt_file = (self.sensitivity_path.joinpath('high_4sig_lightcurves').glob('*.txt'))[0]
        with star_txt_file.open() as file:
            for i,line in enumerate(file):
                if i == 6:
                    sample_star_time = line.split(' ')[-1].strip('\n')
                    sample_star_time = Time(sample_star_time, format='isot', scale='utc')
                    break
            else:
                print("ERROR: Could not find sample star time in {}!".format(star_txt_file))
                return
            
        # Get field airmass
        field_alt = calculateAlt(star_table['ra'][len(red_table['ra'] // 2)],
                                 star_table['dec'][len(red_table['dec'] // 2)],
                                 sample_star_time)
        field_airmass = calculateAirmass(field_alt)

        print("SUCCESS: Sensitivity data loaded for {}!".format(self.name))
        return star_table, field_airmass
    
    ##TODO: determine if the directory date agrees with the timestamp date


#--------------------------------functions------------------------------------#

#############################
## Handling Times
#############################

def convUTCtoLST(timestamp, value_only=True):
    
    # Strip leading/trailing whitespace
    given_time = timestamp.strip()
    
    # Create a datetime object which can then be used to calculate the
    # equivalent sidereal time
    datetime_t = datetime.strptime(given_time, ACPLOG_STRP)
    astro_time = Time(datetime_t)
    
    # Calculate sidereal time
    sidereal_time = astro_time.sidereal_time('mean', SITE_LON)
    
    
    # If value_only, give the magnitude of the sidereal time only
    if value_only:
        return sidereal_time.value
    else:
        return sidereal_time
    
    
def setLSTtoUTC(log_lines):
    
    global UTC_REF,LST_REF
    
    # Find the log line with the current LST. Use that as a conversion
    # factor between LST and UTC.
    curr_LST_regex = "INFO: Current LST"
    for line in log_lines:
        if curr_LST_regex in line:
            # Split line into UTC and LST times
            UTC_and_LST = line.split(curr_LST_regex)
            
            # Convert to appropriate types and assign to globals
            UTC_REF = datetime.strptime(UTC_and_LST[0], ACPLOG_STRP)
            LST_REF = float(UTC_and_LST[1])
            
            return True
    else:
        return False


def convLSTtoUTC(LST):
    
    # Check if reference times are set
    if ('UTC_REF' not in globals()) or ('LST_REF' not in globals()):
        print("ERROR: LST and UTC references not set!")
        return
    
    # Calculate the difference between the LST reference and given LST and
    # find the difference to the reference UTC.
    LST_diff = LST - LST_REF
    UTC = UTC_REF + timedelta(hours=(LST_diff/1.0027379))
    
    return UTC

def convJDtoUTC(JD):
    
    # Convert to datetime object
    datetime_JD = Time(JD, format='jd', scale='utc')
    
    # Convert to UTC
    UTC = datetime_JD.to_datetime()
    
    return UTC


def hyphonateDate(obsdate):

    # Convert the date to a datetime object
    obsdate = datetime.strptime(obsdate, OBSDATE_FORMAT)

    # Convert the date to a hyphonated string
    obsdate = obsdate.strftime('%Y-%m-%d')

    return obsdate


#############################
## Analyze Logs
#############################

def getImportantLines(log_path, log_patterns):
    
    
    # Break if the log_path is invalid
    if not os.path.isfile(log_path):
        print("ERROR: Log file does not exist!")
        return []
        
    # Read the log line-by-line and save lines matching log_pattern substrings
    matched_line = []
    with open(log_path, "r", encoding="utf-16") as log:

        for line in log.readlines():
            if any(pattern in line for pattern in log_patterns):
                print("Pattern matched!")
                matched_line.append(line.strip("\n"))
                
    return matched_line


def getSunsetSunrise(log_lines):
    
    # Find the calculated JD times for sunset and sunrise
    sunset_found,sunrise_found = False,False
    for line in log_lines:
        if 'INFO: Sunset JD: ' in line:
            # Extract sunset JD from line
            split_line = line.split('INFO: Sunset JD: ')
            sunset = Time(float(split_line[-1]), format='jd', scale='utc')
            sunset_found = True
        elif 'INFO: Sunrise JD: ' in line:
            # Extract sunrise JD from line
            split_line = line.split('INFO: Sunrise JD: ')
            sunrise = Time(float(split_line[-1]), format='jd', scale='utc')
            sunrise_found = True

    # Determine which times could be obtained
    if sunset_found and sunrise_found:
        # Case where both sunset and sunrise were found
        print(f"Sunset occurred at {sunset.to_datetime()} ({sunset}).")
        print(f"Sunrise occurred at {sunrise.to_datetime()} ({sunrise}).")

        return sunset,sunrise
    elif sunset_found:
        # Case where only sunset was found
        print(f"Sunset occurred at {sunset.to_datetime()} ({sunset}).")
        print(f"WARNING: Sunrise could not obtained!")

        return sunset,None
    elif sunrise_found:
        # Case where only sunrise was found
        print(f"WARNING: Sunset could not be obtained!")
        print(f"Sunrise occurred at {sunrise.to_datetime()} ({sunrise}).")

        return None,sunrise
    else:
        # Case where neither sunset nor sunrise were found
        print(f"WARNING: Sunset and sunrise could not be obtained!")

        return None,None


def summarizeEvents(log_lines):

    # Gets relevant actual events from the log
    weather_aborts = getWeatherUnsafe(log_lines)
    dome_closures  = getDomeClosure(log_lines)

    # Create marker list correlated with event times
    event_times = []
    markers     = []
    for event in weather_aborts:
        event_times.append(event)
        markers.append('x')
    for event in dome_closures:
        event_times.append(event)
        markers.append('D')

    # Return event_times and markers lists
    return event_times,markers


def getObservingPlan_LST(log_lines):
    
    # Get shortlist lines and extract LST start time
    plan_info = []
    for line in log_lines:
        if 'starts' in line:
            # Split the line into individual words
            shortlist_seg = line.split(' ')
            
            # Get from the line segments the field number, start LST time, and
            # the number of stars.
            field_id  = shortlist_seg[7]
            field_num = int(field_id.strip('field'))
            start_LST = float(shortlist_seg[9])
            num_stars = int(shortlist_seg[16])
            
            # Convert LST to UTC
            start_UTC = convLSTtoUTC(start_LST)

            # Save the tuple to the list
            plan_info.append((field_num, start_UTC, num_stars))
    
    # Eliminate duplicates in list
    return list(set(plan_info))


def getObservingPlan_JD(log_lines, return_utc=True):

    # Get shortlist lines and extract JD start time
    plan_info = []
    for line in log_lines:
        if 'starts' in line:
            # Split the line into individual words
            shortlist_seg = line.split(' ')
            
            # Get from the line segments the field number, start LST time, and
            # the number of stars.
            field_id  = shortlist_seg[7]
            field_num = int(field_id.strip('field'))
            start_JD  = float(shortlist_seg[9])
            num_stars = int(shortlist_seg[16])
            
            # Convert JD to UTC if requested
            if return_utc:
                start_UTC = convJDtoUTC(start_JD)
                plan_info.append((start_UTC, field_num, num_stars))
            else:
                plan_info.append((start_JD, field_num, num_stars))

    # Eliminate duplicates in list
    return list(set(plan_info))


def getFieldsObserved(log_lines):
    
    # Find which fields were actually observed and record their start time
    fields_observed = []
    field_regex = "INFO: Field Name: "
    for line in log_lines:
        if field_regex in line:
            # Identify start time of the field and get the field number
            split_line = line.split(field_regex)
            field_time = datetime.strptime(split_line[0].strip(), ACPLOG_STRP)
            field_num  = int(split_line[1].strip('field'))
            
            # Save the tuple to the list
            fields_observed.append((field_time, field_num))
    
    return fields_observed


def getWeatherUnsafe(log_lines):
    
    # Find weather aborts and save the times
    weather_abort_time = []
    weather_regex = "INFO: Weather unsafe!"
    for line in log_lines:
        if weather_regex in line:
            # Get time from weather abort line
            line_time  = line.split(weather_regex)[0]
            abort_time = datetime.strptime(line_time.strip(), ACPLOG_STRP)
            
            # Save time to list
            weather_abort_time.append(abort_time)
            
    return weather_abort_time


def getDomeClosure(log_lines):
    
    # Find weather aborts and save the times
    dome_closure_time = []
    dome_regex = 'ALERT: Dome closed!'
    for line in log_lines:
        if dome_regex in line:
            # Get time from weather abort line
            line_time  = line.split(dome_regex)[0]
            closure_time = datetime.strptime(line_time.strip(), ACPLOG_STRP)
            
            # Save time to list
            dome_closure_time.append(closure_time)
            
    return dome_closure_time


#############################
## Analyze Data Directory
#############################

def getDataTimes(data_dir_path):
    
    # Get all minute directories in the given data directory
    minute_dirs = [folder.name+'000' for folder in data_dir_path.iterdir() \
                   if ((folder.is_dir()) and (folder.name != 'Bias'))]
    minute_dirs.sort()

    # Get timestamps from the directory names
    directory_times = []
    for timestamp in minute_dirs:
        dir_datetime = datetime.strptime(timestamp, MINUTEDIR_STRP)
        directory_times.append(dir_datetime)
        
    return directory_times


#############################
## Coordinate Functions
#############################

def calculateAlt(ra, dec, time):
    
    # Convert RA and Dec to SkyCoord object
    coord = SkyCoord(ra, dec, unit=(units.hourangle, units.deg))
    
    # Calculate the altitude at the given time
    alt = coord.transform_to(AltAz(obstime=time, location=SITE_LOC)).alt
    
    return alt


def calculateAirmass(alt):
    
    # Calculate the airmass
    airmass = 1/np.cos(np.radians(90-alt))
    
    return airmass


#############################
## Weather
#############################

def getCloudData(*obs_dates):

    # Find cloud log files
    weather_data = []
    for weather_date in obs_dates:
        # Path names
        filename = f"weather.log.{hyphonateDate(weather_date)}"
        filepath = CLOUD_PATH / filename

        # Check if this file exists or is "today" and filelines to list
        if filepath.exists():
            with open(filepath,'r') as file:
                for line in file.readlines():
                    weather_data.append(line.split(','))
        elif weather_date == datetime.now().strftime(OBSDATE_FORMAT):
            with open(CLOUD_PATH / "weather.log", 'r') as file:
                for line in file.readlines():
                    weather_data.append(line.split(','))
        else:
            print(f"WARNING: No log for {weather_date}!")

    # Check that at least one weather log exists
    if weather_data is []:
        print("ERROR: No weather logs found!")
        return
    
    # Convert data list to numpy array for convenience
    weather_array = np.array(weather_data,dtype=float)
    print("Successfully found and collected cloud/transparency data.")

    return weather_array


#############################
## Plotting Functions
#############################

def plotTimeBlockVertices(times, height):

    # Create vertices for the time blocks
    addMinute = timedelta(minutes=1)
    vertices  = []
    for time in times:
        v = [(mdates.date2num(time), height - 0.4),
             (mdates.date2num(time), height + 0.4),
             (mdates.date2num(time + addMinute), height + 0.4),
             (mdates.date2num(time + addMinute), height - 0.4),
             (mdates.date2num(time), height - 0.4)]
        
        vertices.append(v)

    return vertices


def plotObservations(red=[], green=[], blue=[]):

    # Create figure and axes
    timeline_fig, (ax1, ax2) = plt.subplots(2, 1)
    loc = mdates.HourLocator(interval=1)
    xfmt = DateFormatter('%H')

    ## Plot actual observations ##

    # Get red/green/blue time blocks
    red_vertices   = plotTimeBlockVertices(red, 1)
    green_vertices = plotTimeBlockVertices(green, 2)
    blue_vertices  = plotTimeBlockVertices(blue, 3)

    # Plot the time blocks
    ##TODO: append these blocks together?
    red_bars   = PolyCollection(red_vertices, facecolors=COLOURMAPPING['red'])
    green_bars = PolyCollection(green_vertices, facecolors=COLOURMAPPING['green'])
    blue_bars  = PolyCollection(blue_vertices, facecolors=COLOURMAPPING['blue'])

    # Set transparency of the blocks
    red_bars.set_alpha(0.7)
    green_bars.set_alpha(0.7)
    blue_bars.set_alpha(0.7)

    # Plot time blocks to the axes
    ax1.add_collection(red_bars)
    ax1.add_collection(green_bars)
    ax1.add_collection(blue_bars)

    # Set the axes limits and labels
    ax1.autoscale()
    ax1.zorder=1
    ax1.patch.set_alpha(0.01)
    ax1.xaxis.set_major_locator(loc)
    ax1.xaxis.set_major_formatter(xfmt)
    ax1.set_xlim([mdates.date2num(sunset), mdates.date2num(sunrise)])
    ax1.set_yticks([1,2,3])
    ax1.set_ylim(top=3+.4)
    ax1.set_yticklabels(['Green','Red','Blue'])
    ax1.xaxis.set_tick_params(labelsize=9)
    ax1.xaxis.grid(True)
    ax1.xaxis.tick_top()

    ## Plot cloud/transparency data ##

    # Read in data from the cloud log
    weatherlog_name = "weather.log.{}".format(obs_date)
    cloud_log_path  = CLOUD_PATH / weatherlog_name
    cloud_data = pd.read_csv()


#############################
## Old Functions
#############################

def ReadFiledList(log_list):
    """
     

    Parameters
    ----------
    log_list : list of str.
        List of Log lines.
    pattern : str
        Line with a specific string.

    Returns
    -------
    Line that matched patter and time in Log of that line.

    """
    
       
    fields=[]
    dates=[]
    for line in log_list:
        if 'Current LST' in line:
            LST=float(line.split(': ')[2])
            UTC=float(line.split(': ')[0].split(' ')[3].split(':')[0]) + float(line.split(': ')[0].split(' ')[3].split(':')[1])/60
            
            time_diff=UTC-LST
            
            
        elif 'starts' in line:
            fields.append((line).split(': ')[1].split(' ')[0])
            #times=(float((line).split(': ')[1].split(' ')[2]))+time_diff
            times=(float((line).split(': ')[1].split(' ')[2]))
            JD = Time(times, format='jd', scale='utc')
            JD.format = 'datetime'
#            if times>24:
#                times=times-24
#            if times<0:
#                times=times+24
#            dates.append(str(obs_date)[:-2]+line.split(" ")[2]+' '+str(times).split('.')[0]+':'+str(math.floor(float('0.'+str(times).split('.')[1])*60))+':00')

            dates.append(JD)           
            

    return fields, dates


def getAirmass(time, RA, DEC):
    '''get airmass of the field at the given time
    input: time [isot format string], field coordinates [RA, Dec]
    returns: airmass, altitude [degrees], and azimuth [degrees]'''
    
    #get local sidereal time
    LST = Time(time, format='isot', scale='utc').sidereal_time('mean', longitude = SITE_LON)

    #get hour angle of field
    HA = LST.deg - RA
    
    #convert angles to radians for sin/cos funcs
    dec = np.radians(DEC)
    siteLat = np.radians(SITE_LAT)
    HA = np.radians(HA)
    
    #get altitude (elevation) of field
    alt = np.arcsin(np.sin(dec)*np.sin(siteLat) + np.cos(dec)*np.cos(siteLat)*np.cos(HA))
    
    #get azimuth of field
    A = np.degrees(np.arccos((np.sin(dec) - np.sin(alt)*np.sin(siteLat))/(np.cos(alt)*np.cos(siteLat))))
    
    if np.sin(HA) < 0:
        az = A
    else:
        az = 360. - A
    
    alt = np.degrees(alt)
    
    #get zenith angle
    ZA = 90 - alt
    
    #get airmass
    airmass = 1./np.cos(np.radians(ZA))
    
    return airmass, alt, az


def readSigma(filepath):
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
    with filepath.open() as f:
        
        #loop through each line of the file
        for i, line in enumerate(f):
            

            
            if i==10:
                sigma = float(line.split(':')[1])
 
    return (sigma)


def twilightTimes(julian_date, site=(SITE_LAT,SITE_LON)):
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


def ReadLogLine(log_list, pattern, Break=True):
    """
    

    Parameters
    ----------
    log_list : list of str.
        List of Log lines.
    pattern : str
        Line with a specific string.

    Returns
    -------
    Line that matched pattern and time in Log of that line.

    """
    
    if Break==False:
       
        messages=[]
        times=[]
        for line in log_list:
            
            for pat in pattern:
                if (pat in line and 'LST' not in line):
                    messages.append((line))
                    # if int(line.split(" ")[3].split(':')[0])>20:
                    #     times.append(str(obs_date)+' '+line.split(" ")[3])
                    # else:
                    #     times.append(str(tomorrowT)+' '+line.split(" ")[3])
                    times.append(str(obs_date)[:-2]+line.split(" ")[2]+' '+line.split(" ")[3])
        #times yyyy-mm-dd hh:mm:ss

        return messages, times
    
    else:
        
        for line in log_list:
            if pattern in line:
                message=line
                
                Time=line.split(" ")[3]
                break
    
        
        return message, Time


def ReadLog(file):
    """
    

    Parameters
    ----------
    file : path-like obj.
        path for Log file

    Returns
    -------
    Log file as list of strings

    """
    
    f= open(file, 'r', encoding='UTF-16')

    log_list=f.readlines()

    
    return(log_list)


def getPrevDate(path):
    """
    Get date of previous results that are present in the log folder

    Parameters
    ----------
    path : path type
        Path of the observation logging.

    Returns
    -------
    time of previous observations results

    """
    try:
        timeline_file=[f for f in path.iterdir() if '.csv' in f.name][0]
        
    except:
        return -1
    return(timeline_file.name.split('_')[0])

    
def ToFM():
    """
    get first hour of observations

    Returns
    -------
    int
        hour number.

    """
    #reading first minute of the night on Red, if no data then switch to Green or Blue
    try:
        
        first_min=[f for f in green_datapath.iterdir() if ('Bias' not in f.name and '.txt' not in f.name)][0]
        
    except:
        try:
            first_min=[f for f in red_datapath.iterdir() if ('Bias' not in f.name and '.txt' not in f.name)][0]
        except:
            first_min=[f for f in blue_datapath.iterdir() if ('Bias' not in f.name and '.txt' not in f.name)][0]

    return int(first_min.name.split('_')[1].split('.')[0])


#------------------------------------main-------------------------------------#


if __name__ == '__main__':
    
    
###########################
## Argument Parser & Setup
###########################

    ## Argparser ##

    # Generate argument parser
    arg_parser = argparse.ArgumentParser(description="Generate diagnostic plots from the ACP logs of a night.",
                                         formatter_class=argparse.RawTextHelpFormatter)
    
    # Available argument functionality
    arg_parser.add_argument('date', help='Observation date (as YYYYMMDD) to be analyzed.')


    # Extract date from command line arguments
    cml_args = arg_parser.parse_args()
    obs_date = cml_args.date
    obs_date_dashed = hyphonateDate(obs_date)
    
    # Check if this date has a valid format
    regex_match = re.fullmatch("[0-9]{8}", obs_date)
    if regex_match is None:
        print("ERROR: Invalid date given.")
        sys.exit()


    ## Setup telescope classes ##

    # Initialize telescopes
    Red = Telescope("RED", "r", "R:", obs_date)
    Green = Telescope("GREEN", "#66ff66", "D:", obs_date)
    Blue  = Telescope("BLUE", "b", "B:", obs_date)
    
    # Check that at least one log file exists
    if not (Red.log_exists | Green.log_exists | Blue.log_exists):
        print("ERROR: No logs exist for this date!")
        sys.exit()

    # Create a directory for diagnostic files
    diagnostic_dir = BASE_PATH / 'Logs' / 'Operations' / obs_date
    if diagnostic_dir.exists():
        for old_file in diagnostic_dir.iterdir():
            old_file.unlink()
    else:
        diagnostic_dir.mkdir()


    ## Initialize observation plots and timeline ##

    # Create figure and axes
    timeline_fig, (ax1, ax3) = plt.subplots(2, 1)
    loc = mdates.HourLocator(interval=1)
    xfmt = DateFormatter('%H')


###########################
## Sunset/Sunrise Operations
###########################

    # Get sunset and sunrise times (as JD) from logs
    for machine in (Red,Green,Blue):
        # Try to get sunset and sunrise times from each machine
        sunsetJD,sunriseJD = getSunsetSunrise(machine.log_lines)

        # If one or both times are missing, try again with next machine
        if (sunsetJD is not None) and (sunriseJD is not None):
            break
    else:
        # TODO: If sunrise and sunset times could not be obtained from
        # TODO: logs, use Mike's function or minute dir timestamps.
        print("ERROR: Could not obtain sunrise and sunset times!")
        sys.exit()

    # Convert sunset/sunrise times to 'fits' format from JD
    #sunset.format  = 'fits'
    #sunrise.format = 'fits'

    # Convert sunset/sunrise times to datetime objects
    sunset  = sunsetJD.to_datetime()
    sunrise = sunriseJD.to_datetime()


###########################
## Timeblocks Plot
###########################

    # Fill timeblocks plot for each telescope
    for i,machine in enumerate((Red,Green,Blue)):

        # Get time blocks from observation data
        vertices = plotTimeBlockVertices(machine.data_times, i)

        # Plot the time blocks
        ##TODO: append these blocks together?
        time_blocks = PolyCollection(vertices, facecolors=machine.colour)

        # Set transparency of the blocks
        time_blocks.set_alpha(0.7)

        # Plot time blocks to the axes
        ax1.add_collection(time_blocks)


    # Set the axes limits and labels of timeblocks plot
    ax1.autoscale()
    ax1.zorder=1
    ax1.patch.set_alpha(0.01)
    ax1.xaxis.set_major_locator(loc)
    ax1.xaxis.set_major_formatter(xfmt)
    ax1.set_xlim([mdates.date2num(sunset), mdates.date2num(sunrise)])
    ax1.set_yticks([1,2,3])
    ax1.set_ylim(top=3+.4)
    ax1.set_yticklabels(['Red','Green','Blue'])
    ax1.xaxis.set_tick_params(labelsize=9)
    ax1.xaxis.grid(True)
    ax1.xaxis.tick_top()


###########################
## Cloud/Transparency Plot
###########################

    print("\n## Cloud/Transparency ##")

    # Get list of unique dates this observation covered
    sunset_date  = sunset.strftime(OBSDATE_FORMAT)
    sunrise_date = sunrise.strftime(OBSDATE_FORMAT)
    if sunset_date == sunrise_date:
        print("Run occurred over one date.")
        cloud_data = getCloudData(sunset_date)
    else:
        print("Run occurred over two dates.")
        cloud_data = getCloudData(sunset_date,sunrise_date)

    # If cloud data has been passed, plot the transparency plot
    # TODO: Fix this so if transparency data could not be found on Green, it will look at other telescopes
    if cloud_data is not None:
        print("Plotting transparency data...")
        # Isolate data between sunset and sunrise
        cloud_data = cloud_data[np.where(cloud_data[:,0] > sunset.timestamp()) and np.where(cloud_data[:,0] < sunrise.timestamp())]

        # Underlay timeblock data with transparency heatmap
        ax = inset_axes(ax1, width="100%", height="100%",loc=3, bbox_to_anchor=(-0.014,-0.06,1,1), bbox_transform=ax1.transAxes)
        ax = sns.heatmap(cloud_data[:,[0,9]],cmap='Blues_r',vmax=5,cbar=False,zorder=2)
        ax.axes.invert_yaxis()

        # Overlay transparency heatmap with transparency linegraph 
        ax2 = plt.twinx()
        sns.lineplot(x=cloud_data[:,0],y=cloud_data[:,9]+15.55,color='k',ax=ax2, zorder=5)

        # Set the axes limits and labels of the transparency plot
        ax2.yaxis.set_ticks(np.arange(0, 5, 1))
        ax2.set_ylim([0,4])
        ax.set_yticklabels([])
        ax2.set_ylabel('mag')
        ax.set_xticklabels([])
        ax2.set_xticklabels([])
        ax2.set_xticks([])
    

###########################
## Event Timeline
###########################

    print("\n## Event Timeline ##")

    # Plot event timeline for each telescope
    for i,machine in enumerate((Red,Green,Blue)):        
        
        # If no log exists, skip this telescope
        if not machine.log_exists:
            print(f"WARNING: Since no log data exists for {machine.name}, skipping event timeline...")
            pass


        ## Scrape information from the log ##

        print(f"Scraping information from {machine.name} log...")

        # Get observing plan from log
        plan_times,plan_markers = [],[]
        for field in getObservingPlan_JD(machine.log_lines):
            plan_times.append(field[0])
            plan_markers.append(fr'${field[1]}$')

        # Get observed fields from log
        field_times,field_markers = [],[]
        for field in getFieldsObserved(machine.log_lines):
            field_times.append(field[0])
            field_markers.append(fr'${field[1]}$')

        # Get physical events
        event_times,event_markers = summarizeEvents(machine.log_lines)


        ## Plot each of the lines for this telescope ##

        print(f"Plotting {machine.name}'s timelines...")

        # Plot observing plan
        for j in range(len(plan_times)):
            ax3.plot(plan_times[j], i*2, color=machine.colour,
                     marker=plan_markers[j], markerfacecolor='k', markeredgecolor='k')
        ax3.axhline(y = i*2, color=machine.colour, linestyle='-')
        ax3.text(mdates.date2num(sunset), 0+i*2, 'planned',fontsize=8, ha='right', va='center') #TODO: fix

        # Plot observed fields
        for j in range(len(field_times)):
            ax3.plot(field_times[j], i*2+0.4, color=machine.colour,
                     marker=field_markers[j], markerfacecolor='k', markeredgecolor='k')
        ax3.axhline(y = i*2+0.4, color=machine.colour, linestyle='-')
        ax3.text(mdates.date2num(sunset), 0+i*2+0.4, 'observed',fontsize=8, ha='right', va='center') #TODO: fix

        # Plot physical events
        for j in range(len(field_times)):
            ax3.plot(event_times[j], i*2+0.8, color=machine.colour,
                     marker=event_markers[j], markerfacecolor='k', markeredgecolor='k')
        ax3.axhline(y = i*2+0.8, color=machine.colour, linestyle='-')
        ax3.text(mdates.date2num(sunset), 0+i*2+0.8, 'events',fontsize=8, ha='right', va='center') #TODO: fix


        # Set the axes limits and labels of the timelines
        ax3.set_xlim([mdates.date2num(sunset), mdates.date2num(sunrise)])#limit plot to sunrise and sunset
        ax3.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax3.yaxis.set_visible(False)
        ax3.spines[:].set_visible(False)
        ax3.xaxis.tick_top()
        ax3.set_xticks([])
        ax3.margins(y=0.1)

        # TODO: Add legend -> currently not formatted correctly
        #legend = "X - bad weather \n ♦ - dome close"
        #plt.text(0.02, 0.5, legend, fontsize=14, transform=timeline_fig.transFigure)

        # Save figure
        timeline_fig.subplots_adjust(hspace=0)
        #plt.title(f"{sunset.strftime(OBSDATE_FORMAT)}:\n{sunset.strftime(CLOCK_FORMAT)} - {sunrise.strftime(CLOCK_FORMAT)}")
        plt.title(f"{obs_date_dashed}")
        timeline_fig.savefig(str(diagnostic_dir / "event.svg"),dpi=800,bbox_inches='tight')
        plt.close()
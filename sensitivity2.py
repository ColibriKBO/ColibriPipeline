"""
Filename:   sensitivity2.py
Author(s):  Peter Quigley
Contact:    pquigle@uwo.ca
Created:    
Updated:    
    
Description:


Usage:

"""

# Module Imports
import sys, os, time
import argparse
import sep
import pathlib
import astropy
import astropy.stats
import numba as nb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from datetime import datetime, date
from astropy.io import fits
from astropy import wcs
from astropy.time import Time 

# Custom Imports
import getRAdec
import VizieR_query
import astrometrynet_funcs
import snplots
import lightcurve_maker
import lightcurve_looker
import read_npy
from colibri_tools import hyphonateDate


#-------------------------------global vars-----------------------------------#

# Processing Parameters
polynom_order = '4th'                           #order of astrometry.net plate solution polynomial
ap_r = 3                                        #radius of aperture for photometry
gain = 'high'                                   #which gain to take from rcd files ('low' or 'high')
global telescope
this_telescope = os.environ['COMPUTERNAME']       #telescope identifier

# Telescope Code Dictionary
BASE_DIR = pathlib.Path('D:')
TELESCOPE_BASE_DIR = {'REDBIRD':pathlib.Path('R:'),
                      'GREENBIRD':pathlib.Path('G:'),
                      'BLUEBIRD':pathlib.Path('B:')}

# Datetime Formats
MINUTEDIR_STRP = '%Y%m%d_%H.%M.%S.%f'
ASTROPY_STRP = '%Y-%m-%dT%H:%M:%S.%f'
OBSTIME_STRP = '%H.%M.%S'

# Catalog Parameters
CATALOG = 'I/345/gaia2'  #reference catalog to use
SEARCH_RADIUS = 2.0      #search radius for reference catalog query (degrees)
MATCH_RADIUS  = 1/3600   #matching radius for reference catalog query (degrees)

#--------------------------------functions------------------------------------#

###########################
## I/O Functions
###########################

def primarysummaryReader(summary_path):
    """
    Read primary_summary.txt file and return a dataframe
    
    Args:
        summary_path (pathlib.Path): path to primary_summary.txt file
    
    Returns:
        df (dataframe): dataframe containing summary information
    
    """

    # Load primary_summary.txt as a pandas dataframe
    try:
        star_hours = pd.read_csv(summary_path, header=None, 
                                    names=['timestamp','stars','detec'],
                                    comment='#', index_col=0,
                                    parse_dates=['timestamp'], date_parser=lambda x: datetime.strptime(x+'000', MINUTEDIR_STRP))
        return star_hours
    
    # If primary pipeline failed, return None (to be ignored)
    except:
        print(f"ERROR: Could not read primary summary on {summary_path.anchor}!")
        return None
    

def formatTimestamp(timestamp):
    """
    Convert timestamp to a string in the format used by astropy.time.Time.

    Args:
        timestamp (datetime): minute timestamp to format

    Returns:
        timestamp_str (str): formatted timestamp

    """

    # Convert timestamp to datetime
    timestamp_str = datetime.strptime(timestamp, MINUTEDIR_STRP)

    # Convert datetime to string
    timestamp_str = timestamp_str.strftime(ASTROPY_STRP)

    return timestamp_str


###########################
## Inter-Telescope Functions
###########################

def checkMinuteMatches(target, minute_list, tolerance=60):
    """
    Find a match for a given minute in a list of minutes within a specified tolerance.

    Args:
        target (datetime): target minute to match
        minute_list (list): list of minutes to search
        tolerance (int): tolerance in seconds for matching

    Returns:
        match (datetime): matched minute
    
    """

    # Iterate through minute list, add valid minutes to list
    matched_minutes = []
    for minute in minute_list:
        # Check if minute is within tolerance
        if abs((target - minute).total_seconds()) < tolerance:
            matched_minutes.append(minute)

    # If only one match is found, return it
    if len(matched_minutes) == 1:
        return matched_minutes[0]
    
    # If multiple matches are found, return the earliest one
    elif len(matched_minutes) > 1:
        return min(matched_minutes)
    
    # If no match is found, return None
    else:
        return None


###########################
## Astrometry Functions
###########################

def getAirmass(time, RA, dec):
    '''get airmass of the field at the given time
    input: time [isot format string], field coordinates [RA, Dec]
    returns: airmass, altitude [degrees], and azimuth [degrees]'''
    
    #latitude and longitude of Elginfield
    siteLat = 43.1933116667
    siteLong = -81.3160233333
    
    #get J2000 day
    
    #get local sidereal time
    LST = Time(time, format='isot', scale='utc').sidereal_time('mean', longitude = siteLong)

    #get hour angle of field
    HA = LST.deg - RA
    
    #convert angles to radians for sin/cos funcs
    dec = np.radians(dec)
    siteLat = np.radians(siteLat)
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

def match_RADec(data, gdata, SR):
    '''matches list of found stars with Gaia catalog by RA/dec to get magnitudes
    input: pandas Dataframe of detected star data {x, y, RA, dec}, dataframe of Gaia data {ra, dec, magnitudes....}, search radius [deg]
    returns: original star data frame with gaia magnitude columns added where a match was found'''
    
    #from Mike Mazur's 'MagLimitChecker.py' script--------------------
    
    match_counter = 0
    
    for i in range(len(data.index)):            #loop through each detected star

        RA = data.loc[i,'RA']                  #RA coord for matching
        DEC = data.loc[i,'Dec']                #dec coord for matching
        SNR = data.loc[i,'snr']                #magnitude of detected star
         
        RA_SR = SR/np.cos(np.radians(DEC))     #adjustment for RA

        df = gdata[(gdata['RA_ICRS'] <= (RA+RA_SR))]      #make new data frame with rows from Gaia df that are within upper RA limit
        df = df[(df['RA_ICRS']  >= (RA-RA_SR))]            #only include rows from this new df that are within lower RA limit
        df = df[(df['DE_ICRS'] <= (DEC+SR))]          #only include rows from this new df that are within upper dec limit
        df = df[(df['DE_ICRS'] >= (DEC-SR))]          #only include rows from this new df that are withing lower dec limit
         
         #RAB - uncomment lines below to match based on smallest distance
        #df['diff'] = np.sqrt(((df.RA_ICRS - RA)**2*np.cos(np.radians(df.DE_ICRS))) + (df.DE_ICRS - DEC)**2)
        #df.sort_values(by=['diff'], ascending=True, inplace = True)
       
        #RAB - uncomment line below to match based on brightest magnitude
        df.sort_values(by=["Gmag"], ascending=True, inplace=True) #sort matches by brightness (brightest at top)
        
        #reset index after sorting
        df.reset_index(drop=True, inplace=True)

        #if matches are found, add corresponsing magnitude columns from brightest Gaia match to original star dataframe
        if len(df.index)>=1:   #RAB - added >= sign
             data.loc[i,'GMAG'] = df.loc[0]['Gmag']
             data.loc[i,'Gaia_RA'] = df.loc[0]['RA_ICRS']
             data.loc[i,'Gaia_dec'] = df.loc[0]['DE_ICRS']
             data.loc[i, 'Gaia_B-R'] = df.loc[0]['BP-RP']

             
       #end of Mike's section -------------------------------------------
             match_counter +=1 
    
    print('Number of Gaia matches: ', match_counter)
    
    return data


def match_XY(mags, snr, SR):
    '''matches two data frames based on X, Y 
    input: pandas dataframe of star data with Gaia magnitudes {x, y, RA, dec, 3 magnitudes} 
    dataframe of star data with SNR {x, y, med, std, snr}, search radius [px]
    returns: original star data frame with SNR data added where a match was found'''
    
    match_counter = 0
    
    #modified from Mike Mazur's code in 'MagLimitChecker.py'------------------
    SR = 0.1 # Search distance in pixels
    for i in range(len(mags.index)):                    #loop through each detected star
         X = mags.loc[i,'X']                            #X coord from star magnitude table
         Y = mags.loc[i,'Y']                            #Y coord from star magnitude table

         df = snr[(snr.X < (X+SR))]                     #make new data frame with rows from SNR df that are within upper X limit
         df = df[(df.X > (X-SR))]                       #only include rows from this new df that are within lower X limit
         df = df[(df.Y < (Y+SR))]                       #only include rows from this new df that are within lower Y limit
         df = df[(df.Y > (Y-SR))]                       #only include rows from this new df that are within lower Y limit
         df.sort_values(by=["SNR"], ascending=True, inplace=True)       #sort matches by SNR
         df.reset_index(drop=True, inplace=True)

         #if matches are found, add corresponsing med, std, SNR columns from match to original star magnitude dataframe
         if len(df.index)>=1:
             mags.loc[i,'med'] = df.loc[0]['med']
             mags.loc[i,'std'] = df.loc[0]['std']
             mags.loc[i,'SNR'] = df.loc[0]['SNR']
             
       #end of Mike's section -------------------------------------------
             match_counter +=1 
    
    print('Number of SNR matches: ', match_counter)
    return mags


#----------------------------------main---------------------------------------#

if __name__ == '__main__':

###########################
## Argument Parser & Setup
###########################

    ## Argparser ##

    # Generate argument parser
    arg_parser = argparse.ArgumentParser(description=""" Analyze sensitivity of Colibri system for a given minute sycnhronized across all telescopes.
        """,
        formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('-d', '--date', help='Observation date (YYYYMMDD) of data to be processed.', required=True)
    arg_parser.add_argument('-m', '--minute', help='hh.mm.ss to process.')
    arg_parser.add_argument('-l', '--lightcurve', help='Star detection threshold.', default=True)

    # Extract date from command line arguments
    cml_args = arg_parser.parse_args()
    obs_date = cml_args.date
    obs_date_dashed = hyphonateDate(obs_date)

    # Substitute path into appropriate telescope
    TELESCOPE_BASE_DIR[this_telescope] = BASE_DIR

    # Set up pathing
    archive_dir = TELESCOPE_BASE_DIR[this_telescope] / 'ColibriArchive' / obs_date_dashed



###########################
## Minute Selection
###########################

    # If minute is specified, check that npy file exists and process it
    if cml_args.minute:

        minute_to_process = cml_args.minute
        minute_npy = sorted(archive_dir.glob(f'{minute_to_process}*.npy'))
        
        # Load minute star data if it exists
        if len(minute_npy) == 1:
            # Load minute data
            star_data = np.load(minute_npy[0])

        # If an unexpected number of npy files is found, raise an error
        elif len(minute_npy) == 0:
            raise FileNotFoundError(f'No npy file found for minute {minute_to_process}')
        else:
            raise ValueError(f'Multiple npy files found for minute {minute_to_process}')

    # Decide on unified minute to process if none is specified
    else:

        # Read primary summary for each telescope
        star_hours = {}
        for telescope,base_dir in TELESCOPE_BASE_DIR.items():
            star_df = primarysummaryReader(base_dir / 'ColibriArchive' / obs_date_dashed / 'primary_summary.txt')
            if star_hours[telescope] is None:
                print(f'ERROR: No primary summary found for {obs_date_dashed}')
                continue
            else:
                # Sort by number of stars
                star_df.sort_values(by='stars', ascending=False, inplace=True)
        
            star_hours[telescope] = star_df

        # Get list of telescopes and iterate through them
        nonzero_telescopes = star_hours.keys()
        nonzero_telescopes.sort()
        for telescope in nonzero_telescopes:
                
            # Get list of minutes for this telescope
            minutes = star_hours[telescope].index

            # Iterate through minutes, trying to find a minute with a match in all other telescopes
            for minute in minutes:

                # Check if minute has a match in all other telescopes
                minute_matches = {telescope:minute}
                for other_telescope in nonzero_telescopes:
                    if other_telescope != telescope:
                        minute_matches[other_telescope] = checkMinuteMatches(minute, star_hours[other_telescope].index)
                
                # If a match is found in all other telescopes, break out of all loops
                if len(minute_matches) == len(nonzero_telescopes):
                    minute_to_process = minute_matches[this_telescope]
                    break
            else:
                continue
            
            # If a match is found, break out of all loops
            break
        
        # If no match is found, process the minute with the most stars
        else:
            minute_to_process = star_hours[this_telescope].index[0]

        
        # Load minute data
        minute_npy = sorted(archive_dir.glob(f'{minute_to_process}*.npy'))
        if len(minute_npy) == 0:
            raise FileNotFoundError(f'No npy file found for minute {minute_to_process}')
        else:
            # Load star data from minute
            # [0]: x, [1]: y, [2]: half-light radius, 
            # [3]: SNR, [4]: RA, [5]: Dec
            star_data = np.load(minute_npy[0])


###########################
## Magnitude Matching
###########################

    # Breakdown of star_data
    coords_df = pd.DataFrame(star_data[:, [0,1,3,4,5]], columns=['x','y','SNR','RA','Dec'])
    radec = star_data[:, [4,5]]
    snr = star_data[:, 3]

    # Get list of stars in reference catalog
    print(f"Querying VizieR for field center {np.median(radec, axis=0)}...")
    gaia = VizieR_query.makeQuery(np.median(radec, axis=0), SEARCH_RADIUS)

    # Match stars in reference catalog to stars in image
    matched_coords_df = match_RADec(coords_df, gaia, MATCH_RADIUS)

    # Get azimuth and altitude of field
    airmass, altitude, azimuth = getAirmass(formatTimestamp(minute_to_process), np.median(radec, axis=0)[0], np.median(radec, axis=0)[1])


###########################
## Directory/File Management
###########################

    # Create directory for minute
    sensitivity_dir = archive_dir / 'Sensitivity'
    minute_dir = sensitivity_dir / minute_to_process

    # Create minute directory if it doesn't exist
    if not sensitivity_dir.exists():
        sensitivity_dir.mkdir()
    if not minute_dir.exists():
        minute_dir.mkdir()

    # Save star data to csv
    coords_df.to_csv(minute_dir / f'StarTable_{minute_to_process}_{this_telescope}.csv', index=False)


###########################
## Plotting
###########################

    # Save plot of mag vs SNR
    plt.scatter(matched_coords_df['GMAG'], matched_coords_df['SNR'], s=5,
                label='Airmass: %.2f\nAlt, Az: (%.2f, %.2f)' % (airmass, altitude, azimuth))
    plt.title(f'{minute_to_process} {this_telescope}')
    plt.xlabel('Gaia Magnitude')
    plt.ylabel('SNR')
    plt.legend()
    plt.grid()
    plt.savefig(minute_dir / f'MagSNR_{minute_to_process}_{this_telescope}.png')
    plt.close()


    #TODO: plot of (ra - GAIA_RA) vs (dec - GAIA_dec) to check for systematic errors
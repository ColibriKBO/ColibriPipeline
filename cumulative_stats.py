"""
Filename:   cumulative_stats.py
Author(s):  Peter Quigley
Contact:    pquigley@uwo.ca
Created:    Nov 22, 2022
Updated:    June 28, 2023
    
Usage: python cumulative_stats.py <obs_date>
       *This script is intended to run only on GREENBIRD
"""

import sys,os
import os
import re
import shutil
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator


#-------------------------------global vars-----------------------------------#

# Path variables
BASE_PATH = Path('D:/')
DATA_PATH = BASE_PATH / 'ColibriData'
IMGE_PATH = BASE_PATH / 'ColibriImages'
ARCHIVE_PATH = BASE_PATH / 'ColibriArchive'

# Cumulative stats structure
STATS_PATH = BASE_PATH / 'CentralRepo' / 'CumulativeStats'
DETEC_PATH = STATS_PATH / 'AllDetections'
SATELLITE_PATH = STATS_PATH / 'Matches-Satellite'
TIER3_PATH = STATS_PATH / 'Matches-Tier3'
TIER3_FILE = TIER3_PATH / 'tier3_log.csv'
STATS_FILE = STATS_PATH /'cumulative_stats.csv'

# STRP formats
OBSDATE_FORMAT = '%Y%m%d'
CLOCK_FORMAT   = '%H:%M:%S'
ACPLOG_STRP    = '%a %b %d %H:%M:%S %Z %Y'
MINUTEDIR_STRP = '%Y%m%d_%H.%M.%S.%f'
TIMESTAMP_STRP = '%Y-%m-%dT%H:%M:%S.%f'

# Regex patterns
MATCHEDDIR_REGEX = re.compile('([0-9\_\-]{24})-(Tier[123])')

# Verbose print statement
verboseprint = lambda *a, **k: None


#-------------------------------classes---------------------------------------#

class Telescope:

    def __init__(self, name, base_path, obs_date):

        # Telescope identifiers
        print(f"Initializing {name}...")
        self.name = name

        # Error logging
        self.errors = []

        # Path variables
        self.base_path = Path(base_path)
        self.obs_archive = self.base_path / "ColibriArchive" / hyphonateDate(obs_date)

        # Get nightly statistics
        if self.obs_archive.exists():
            # Get number of minute directories for the night
            medstack_file_list = list(self.obs_archive.glob('*medstacked.fits'))
            self.obs_time = len(medstack_file_list)

            # Get number of occultation candidates for the night
            occult_file_list = list(self.obs_archive.glob('det_*.txt'))
            self.occultations = len(occult_file_list)

            # Copy occultation files to cumulative occultation folder
            for occult_file in occult_file_list:
                verboseprint(f"Copying {occult_file} to {DETEC_PATH.name}")
                copyDetFiles(occult_file)

            # Get single-telescope star-hours for the night
            summary_file = self.obs_archive / 'primary_summary.txt'
            if summary_file.exists():
                self.star_hours = self.getStarHours(summary_file)
            else:
                self.star_hours = np.nan

        # If the night folder does not exist, log zeros
        else:
            self.obs_time = 0
            self.occultations = 0
            self.star_hours = 0

    
    def addError(self, error_msg):

        self.errors.append(error_msg)
        print(error_msg)

    
    def getStarHours(self, summarytxt):

        parse_summary = {
                    0: lambda timestamp: datetime.strptime(timestamp.decode('ascii')+'000', MINUTEDIR_STRP),
                    1: lambda stars: int(stars),
                    2: lambda detec: int(detec)
                         }

        # Load primary_summary.txt
        try:
            primary_summary = np.loadtxt(summarytxt, delimiter=',', converters=parse_summary, 
                                         ndmin=2, dtype=object)
            star_mins = np.sum(primary_summary[:,1])
            return star_mins/60.
        except IndexError:
            self.addError(f"ERROR: Could not read primary summary on {self.name}!")
            return np.nan


#-------------------------------functions-------------------------------------#

def getNightTime(f):
    """
    Get time of the night folder

    Parameters
    ----------
    f : path type
        Night folder.

    Returns
    -------
    datime
        date of the night folder.

    """
    try:
        NightTime=datetime.strptime(f.name[0:10], '%Y-%m-%d').date()
    except:
        return date(2020, 1, 1)
        pass
    return NightTime


def copyDetFiles(det_path):
    """
    Copy detection files to cumulative detection folder.
    Add sigma to the end of the file name.

    Parameters
    ----------
    det_dir_path : path type
        Detection file filepath.
    
    Returns
    -------
    None.
    """

    # Get sigma
    sigma = readSigma(det_path)

    # Rename file
    det_name = det_path.name.strip('.txt') + f'_sig{sigma:.2f}.txt'

    # Copy file
    shutil.copy(det_path, DETEC_PATH / det_name)


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
 
    return sigma


def parseMatchedDetDir(det_dir_path):
    """
    
    """

    # Count number of det files in directory
    det_files = list(det_dir_path.glob('det_*.txt'))
    det_count = len(det_files)

    # Get datetime and match tier of the directory
    match_regex = MATCHEDDIR_REGEX.match(det_dir_path.name)
    timestamp   = match_regex.group(1)
    match_tier  = match_regex.group(2)

    ## Sort detections by tier

    # Tier 1 and 2
    if match_tier in ['Tier1', 'Tier2']:
        print(f"Copying {det_dir_path.name} to satellite folder...")
        shutil.copytree(det_dir_path, SATELLITE_PATH / det_dir_path.name, dirs_exist_ok=True)
        return 'satellite'
    
    # Tier 3
    elif (match_tier == 'Tier3') and (det_count == 2):
        # Copy detections to Tier3 folder
        print(f"Copying {det_dir_path.name} to Tier3 folder...")
        shutil.copytree(det_dir_path, TIER3_PATH / det_dir_path.name, dirs_exist_ok=True)

        # Get sigma values
        sigma1 = readSigma(det_files[0])
        sigma2 = readSigma(det_files[1])

        # Return both values
        return 'double', timestamp, sigma1, sigma2
    
    elif (match_tier == 'Tier3') and (det_count == 3):
        # Copy detections to Tier3 folder
        print(f"Copying {det_dir_path.name} to Tier3 folder...")
        shutil.copytree(det_dir_path, TIER3_PATH / det_dir_path.name, dirs_exist_ok=True)

        # Get sigma values
        sigma1 = readSigma(det_files[0])
        sigma2 = readSigma(det_files[1])
        sigma3 = readSigma(det_files[2])

        # Return all values
        return 'triple', timestamp, sigma1, sigma2, sigma3

    # Error
    else:
        print(f"ERROR: {det_dir_path.name} does not match any known tier!")
        return (None)


def plotOccCandidates():
    """
    Plot histogram of all historical occultation candidates.

    Parameters
    ----------
    None

    Returns
    -------
    Plot of all historical occultation candidates.
    
    """

    # Get list of all det files
    det_files = list(DETEC_PATH.glob('det_*.txt'))

    # Get sigma values
    try:
        sigmas =  [float(sigma.name.split('sig')[-1].strip('.txt')) for sigma in det_files]
    except IndexError:
        print("ERROR: Could not read sigma values from det file names!")
        print("Manual inspection required.")
        return
    
    # Plot histogram with overflow & underflow bins
    binwidth = 0.25
    minbin = 6
    maxbin = 12
    bins = np.arange(minbin-binwidth, maxbin+binwidth, binwidth)
    plt.hist(np.clip(sigmas, bins[0], bins[-1]), bins=bins)

    # Set plot features
    plt.title(f"Cumulative Occultation Candidates: {len(sigmas)}")
    plt.xlabel("Significance")
    plt.ylabel("Count")
    plt.xlim(bins[0], bins[-1])
    plt.grid(axis='x')
    plt.savefig(STATS_PATH / 'occ_candidates.jpg', dpi=800)
    plt.close()


def plotMatchedCandidates(tier3_df):
    """
    Plot histogram of all historical matched candidates.

    Parameters
    ----------
    tier3_df : pandas dataframe
        Dataframe of all tier 3 matches. Read in from csv file.

    Returns
    -------
    Plot of all historical matched candidates.
    Seperate plots for 2 and 3 telescope matches.
    """

    # Get rows containing 2 and 3 telescope matches
    tel2_df = tier3_df[tier3_df['sigma3'].isnull()]
    tel3_df = tier3_df[~tier3_df['sigma3'].isnull()]

    # Get minimum sigma values
    tel2_min = tel2_df[['sigma1', 'sigma2']].min(axis=1)
    tel3_min = tel3_df[['sigma1', 'sigma2', 'sigma3']].min(axis=1)

    # Histogram binning parameters
    fig, (ax1, ax2) = plt.subplots(2, 1)
    binwidth = 0.25
    minbin = 6
    maxbin = 12
    bins = np.arange(minbin-binwidth, maxbin+binwidth, binwidth)


    ## Plot 2-telescope histogram with overflow & underflow bins

    # Plot histogram with overflow & underflow bins
    print("Plotting 2-telescope histogram...")
    ax1.hist(np.clip(list(tel2_min), bins[0], bins[-1]), bins=bins)

    # Set plot features
    ax1.title.set_text(f"2-Telescope Cumulative Occultations: {len(tel2_df)}")
    ax1.set_xlabel("Pair Min Significance")
    ax1.set_ylabel("Count")
    ax1.set_xlim(bins[0], bins[-1])
    ax1.grid(axis='x')

    
    ## Plot 3-telescope histogram with overflow & underflow bins

    # Plot histogram with overflow & underflow bins
    print("Plotting 3-telescope histogram...")
    ax2.hist(np.clip(list(tel3_min), bins[0], bins[-1]), bins=bins)

    # Set plot features
    ax2.title.set_text(f"3-TelescopeCumulative Occultation: {len(tel3_df)}")
    ax2.set_xlabel("Triplet Min Significance")
    ax2.set_ylabel("Count")
    ax2.set_xlim(bins[0], bins[-1])
    ax2.grid(axis='x')

    # Save figure
    print("Saving histograms...")
    fig.subplots_adjust(hspace=0.5)
    fig.savefig(STATS_PATH / 'occ_matches.jpg', dpi=800)
    plt.close()



def hyphonateDate(obsdate):

    # Convert the date to a datetime object
    obsdate = datetime.strptime(obsdate, OBSDATE_FORMAT)

    # Convert the date to a hyphonated string
    obsdate = obsdate.strftime('%Y-%m-%d')

    return obsdate


#-------------------------------main------------------------------------------#

if __name__ == '__main__':


###########################
## Argument Parser/Setup
###########################

    # Generate argument parser
    description = "Collect cumulative statistics for a given night."
    arg_parser = argparse.ArgumentParser(description=description,
                                         formatter_class=argparse.RawTextHelpFormatter)
    
    # Available argument functionality
    arg_parser.add_argument('date', help='Observation date (YYYYMMDD) of data to be processed.')
    arg_parser.add_argument('starhours', help='Matched star-hours for the night. Generated by wcsmatching.py on BLUEBIRD.',
                            type=float)
    arg_parser.add_argument('-n', '--noplot', help='Do not generate plots.',
                            action='store_false')
    arg_parser.add_argument('-v', '--verbose', help='Increase output verbosity.', 
                            action='store_true')

    # Process argparse list as useful variables
    cml_args  = arg_parser.parse_args()
    obsdate   = cml_args.date
    starhours = cml_args.starhours
    gen_plot  = cml_args.noplot
    
    # Update verboseprint function
    if cml_args.verbose:
        verboseprint = print

    # Environment variable to ensure GREENBIRD is used
    if os.environ['COMPUTERNAME'] != 'GREENBIRD':
        raise Exception("ERROR: This script must be run on GREENBIRD!")

    # Load the cumulative statistics compendium.
    # Columns are as follows:
    # [0] Obsdate;
    # [1] Red observing time (hours); [2] Red star hours; [3] Red occultations;
    # [4] Green observing time (hours); [5] Green star hours; [6] Green occultations;
    # [7] Blue observing time (hours); [8] Blue star hours; [9] Blue occultations;
    # [10] Matched star hours; [11] satellite matches;
    # [12] 2-telescope matched occultations; [13] 3-telescope matched occultations
    stats_df = pd.read_csv(STATS_FILE, parse_dates=['obsdate'], index_col='obsdate')
    verboseprint("SUCCESS: Loaded cumulative statistics.")

    # Check that this night has not been added already
    if obsdate in stats_df.index:
        print(f"WARNING: {obsdate} has already been added to the cumulative statistics!")


###########################
## Collect Statistics
###########################

    # Initialize telescope classes
    Red   = Telescope("RED", "R:", obsdate)
    Green = Telescope("GREEN", "D:", obsdate)
    Blue  = Telescope("BLUE", "B:", obsdate)

    ## Count matched detections
    # Tier 1: matched to the second (candidates for satellite)
    # Tier 2: matched to the 0.1 second (needs further examination)
    # Tier 3: matched in RA/Dec (candidate for KBO)
    obs_match_dir = Green.obs_archive / 'matched'

    # Load the tier3 dataframe
    tier3_df = pd.read_csv(TIER3_FILE, index_col='timestamp')


    # Check that detection matching has been done
    if not obs_match_dir.exists():
        print(f"ERROR: No matched directory for {obsdate}!")
        satellite_matches, match2, match3 = np.nan, np.nan, np.nan
    
    # Process the matched events and track each kind
    else:
        # Initialize counters
        satellite_matches, match2, match3 = 0, 0, 0

        # Iterate through matched detection directories
        for match_dir in obs_match_dir.iterdir():
            if match_dir.is_dir():

                # Parse the directory
                parse_output = parseMatchedDetDir(match_dir)

                # Decide what to do with the output
                if parse_output == 'satellite':
                    satellite_matches += 1
                elif parse_output[0] == 'double':
                    match2 += 1
                    tier3_df.loc[parse_output[1]] = [parse_output[2], parse_output[3], np.nan]
                elif parse_output[0] == 'triple':
                    match3 += 1
                    tier3_df.loc[parse_output[1]] = [parse_output[2], parse_output[3], parse_output[4]]


###########################
## Output
###########################

    # Add/replace the night's statistics to the dataframe
    stats_df.loc[obsdate] = [Red.obs_time, Red.star_hours, Red.occultations,
                            Green.obs_time, Green.star_hours, Green.occultations,
                            Blue.obs_time, Blue.star_hours, Blue.occultations,
                            starhours, satellite_matches, match2, match3]
    print(stats_df.loc[obsdate])
    
    # Save the updated dataframes as CSVs
    #STATS_FILE.unlink()
    stats_df.to_csv(STATS_FILE)
    tier3_df.to_csv(TIER3_FILE)

    # Generate plots
    if gen_plot:
        plotOccCandidates()
        plotMatchedCandidates(tier3_df)

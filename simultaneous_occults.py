# -*- coding: utf-8 -*-
"""
Created on Jul 29 10:04:14 2022

@author: Roman A.

Match dip detection txts throughout 3 telescopes based on time and coordinates, runs only on Green

2022-09-21 Roman A. simplified some steps and added data removal output
"""

# Module Imports
import numpy as np
import pandas as pd
from pathlib import Path
import os
import re
import time
import shutil
from astropy import wcs
from astropy.io import fits
import astrometrynet_funcs
import getRAdec
import itertools
from math import isclose
import sys
import fnmatch
import math
import argparse
from datetime import datetime,date,timedelta

# Custom Script Imports
import generate_specific_lightcurve as gsl


#-------------------------------global vars-----------------------------------#

# Path variables
BASE_PATH = Path('D:/')
DATA_PATH = BASE_PATH / 'ColibriData'
IMGE_PATH = BASE_PATH / 'ColibriImages'
ARCHIVE_PATH = BASE_PATH / 'ColibriArchive'
CENTRAL_PATH = BASE_PATH / 'CentralRepo'

# STRP formats
BARE_FORMAT = '%Y-%m-%d_%H%M%S_%f'
MINDIR_FORMAT = '%Y%m%d_%H.%M.%S.%f'
TIMESTAMP_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'
OBSDATE_FORMAT = '%Y%m%d'
NICE_FORMAT = '%Y-%m-%d_%H:%M:%S'

# Regex patterns
DET_TIME_REGEX = re.compile('det_(\d{4}-\d{2}-\d{2}_\d{6}_\d{6})\d{3}')

# Detection tolerances
TIME_TOLERANCE = 0.2  # seconds
COORD_TOLERANCE = 0.01  # degrees


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

        # Check if the archive for that night exists
        if self.obs_archive.exists():
            det_list = list(self.obs_archive.glob("det_*.txt"))
        else:
            print(f"ERROR: No archive found for {self.name} on {obs_date}!")
            self.det_list = []

        # Analyze time of detections
        self.det_times = [datetime.strptime(DET_TIME_REGEX.match(det.name).group(1), BARE_FORMAT)
                          for det in det_list]
        self.det_dict = dict(zip(self.det_times, det_list))


#-------------------------------functions-------------------------------------#

def readRAdec(filepath):
    """
    Reads Ra and Dec line in the detection .txt file

    Parameters
    ----------
    filepath : path-like obj.
        Path of the detection .txt

    Returns
    -------
    star_ra : float
        RA of the occulted star.
    star_dec : float
        Dec of the occulted star.

    """
    
    with open(filepath,'r') as f:
        
        #loop through each line of the file
        for i, line in enumerate(f):
            
            if i==6:
                
                try:
                    star_coords = line.split(':')[1].split(' ')[1:3]
                    star_ra = float(star_coords[0])
                    star_dec = float(star_coords[1])

                    return star_ra, star_dec
                except:
                    print(f'ERROR: could not read RA and Dec in {filepath}! Please reprocess data.')
                    return float('inf'), float('inf')


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
    description = "Match occultation candidate events. Tiers are as follows:\n" +\
                   "1. Match to 1 second\n2. Match to 0.2 seconds\n3. Match in time and coordinates"
    arg_parser = argparse.ArgumentParser(description=description,
                                         formatter_class=argparse.RawTextHelpFormatter)
    
    # Available argument functionality
    arg_parser.add_argument('date', help='Observation date (YYYYMMDD) of data to be processed.')

    # Process argparse list as useful variables
    cml_args  = arg_parser.parse_args()
    obsdate   = cml_args.date

    # Initialize telescope classes
    Red   = Telescope("RED", "R:", obsdate)
    Green = Telescope("GREEN", "D:", obsdate)
    Blue  = Telescope("BLUE", "B:", obsdate)

    # Setup matched directory structure
    matched_dir = Green.obs_archive / "matched"
    if not matched_dir.exists():
        matched_dir.mkdir(parents=True, exist_ok=True)


###########################
## 1 Second Matching
###########################

    ## How to do this:
    # 1. Loop through all Green detections
    # 2. Check if the time is within 1 second of a detection of any Red detections
    # 3. Check if the time is within 1 second of a detection of any Blue detections
    # 4. If both are true, copy the detections to a matched directory
    # 5. Repeat for all Red detections

    # Match detections in time to the second
    for G_time in Green.det_times:

        # Check if the time is within 1 second of a detection of any Red detections
        for R_time in Red.det_times:
            if abs((G_time - R_time).total_seconds()) <= 1:

                # Check if the time is within 1 second of a detection of any Blue detections
                for B_time in Blue.det_times:
                    if abs((G_time - B_time).total_seconds()) <= 1:

                        # Matched detections found, copy to matched directory
                        G_det = Green.det_dict[G_time]
                        R_det = Red.det_dict[R_time]
                        B_det = Blue.det_dict[B_time]

                        # Create matched directory if it doesn't exist
                        match_dir = matched_dir / R_time.strftime(BARE_FORMAT)
                        if not match_dir.exists():
                            match_dir.mkdir(parents=True, exist_ok=True)

                        # Copy files to matched directory
                        shutil.copy(G_det, match_dir)
                        shutil.copy(R_det, match_dir)
                        shutil.copy(B_det, match_dir)


                        # Log matched detections
                        print(f"Matched {G_det.name} to {R_det.name} and {B_det.name}!")

                        # Break out of the Blue loop
                        break

                # If no match was found with Blue, copy Red and Green
                else:
                    # Matched detections found, copy to matched directory
                    G_det = Green.det_dict[G_time]
                    R_det = Red.det_dict[R_time]

                    # Create matched directory if it doesn't exist
                    match_dir = matched_dir / R_time.strftime(BARE_FORMAT)
                    if not match_dir.exists():
                        match_dir.mkdir(parents=True, exist_ok=True)

                    # Copy files to matched directory
                    shutil.copy(G_det, match_dir)
                    shutil.copy(R_det, match_dir)

                    # Log matched detections
                    print(f"Matched {G_det.name} to {R_det.name}!")

                # Break out of the Red loop
                break

        # If no match was found with Red, check Blue
        else:

            for B_time in Blue.det_times:
                if abs((G_time - B_time).total_seconds()) <= 1:

                    # Matched detections found, copy to matched directory
                    G_det = Green.det_dict[G_time]
                    B_det = Blue.det_dict[B_time]

                    # Create matched directory if it doesn't exist
                    match_dir = matched_dir / G_time.strftime(BARE_FORMAT)
                    if not match_dir.exists():
                        match_dir.mkdir(parents=True, exist_ok=True)

                    # Copy files to matched directory
                    shutil.copy(G_det, match_dir)
                    shutil.copy(B_det, match_dir)


                    # Log matched detections
                    print(f"Matched {G_det.name} to {B_det.name}!")

                    # Break out of the Blue loop
                    break

    # Check matches between Red and Blue
    for R_time in Red.det_times:

        # Check if the time is within 1 second of a detection of any Red detections
        for B_time in Blue.det_times:
            if abs((R_time - B_time).total_seconds()) <= 1:

                # Matched detections found, copy to matched directory
                R_det = Red.det_dict[R_time]
                B_det = Blue.det_dict[B_time]

                # Create matched directory if it doesn't exist
                match_dir = matched_dir / R_time.strftime(BARE_FORMAT)
                if not match_dir.exists():
                    match_dir.mkdir(parents=True, exist_ok=True)

                # Copy files to matched directory
                shutil.copy(R_det, match_dir)
                shutil.copy(B_det, match_dir)

                # Log matched detections
                print(f"Matched {R_det.name} to {B_det.name}!")

                # Break out of the Blue loop
                break


    # Check that at least one match was found
    if not any(match_dir.iterdir()):
        print("No time matches tonight!")
        sys.exit()

    # Otherwise setup to tier the matches
    tier1,tier2,tier3 = [],[],[]


###########################
## Refine Second Matching
###########################

    # Iterate through match directories and refine time matching
    for match_dir in matched_dir.iterdir():

        # Get the time of the match files
        matched_det_times = [datetime.strptime(DET_TIME_REGEX.match(det.name).group(1), BARE_FORMAT)
                            for det in match_dir.iterdir()]
        
        # Check if the match times are within tolerance of each other
        # Tier 1 if only 1-second match, tier 2 if tolerance match
        for tel1,tel2 in itertools.combinations(matched_det_times, 2):
            if abs((tel1 - tel2).total_seconds()) <= TIME_TOLERANCE:
                tier2.append(match_dir)
                break
        else:
            print(f"{match_dir.name} is a tier 1 match.")
            tier1.append(match_dir)


###########################
## Coordinate Matching
###########################

    # Check coordinate matching for tier 2 matches
    for i,match_dir in enumerate(tier2):

        # Get the coordinates of the match files
        matched_det_coords = [readRAdec(det) for det in match_dir.iterdir()]

        # Check if the match coordinates are within tolerance of each other
        # Tier 2 if no match, tier 3 if tolerance match
        for tel1,tel2 in itertools.combinations(matched_det_coords, 2):
            if (abs(tel1[0] - tel2[0]) <= COORD_TOLERANCE) and \
               (abs(tel1[1] - tel2[1]) <= COORD_TOLERANCE):
                
                # Remove from tier 2 and add to tier 3
                print(f"{match_dir.name} is a tier 3 match.")
                del tier2[i]
                tier3.append(match_dir)
                break
        else:
            print(f"{match_dir.name} is a tier 2 match.")


###########################
## Generate Artificial Lightcurves
###########################

    # Check which telescopes are represented in each tier
    for match_dir in matched_dir.iterdir():
        # If we can find the name in here, request an aritifical lightcurve
        R_here,G_here,B_here = False,False,False

        # Check which telescopes are represented
        matched_det_file = [det for det in match_dir.iterdir()]
        for file in matched_det_file:
            if "REDBIRD" in file.name:
                R_here = True
            elif "GREENBIRD" in file.name:
                G_here = True
            elif "BLUEBIRD" in file.name:
                B_here = True

        # If we don't have all three, request an artificial lightcurve
        if not R_here:
            cmd_dir = "RED"
        elif not G_here:
            cmd_dir = "GREEN"
        elif not B_here:
            cmd_dir = "BLUE"
        else:
            continue

        # Get event parameters
        timestamp = datetime.strftime(match_dir.name.split('-Tier')[0],
                                        BARE_FORMAT)
        timestamp = timestamp.strptime(TIMESTAMP_FORMAT)
        radec = readRAdec(matched_det_file[0])

        # Write command to appropriate directory
        command = f"python generate_specific_lightcurve.py {obsdate} {timestamp} {radec[0]} {radec[1]}"
        (CENTRAL_PATH / "Commands" / cmd_dir).touch()


###########################
## Rename Directories
###########################

    # Rename tier 1 directories
    for match_dir in tier1:
        match_dir.rename(matched_dir / f"{match_dir.name}-Tier1")
    
    # Rename tier 2 directories
    for match_dir in tier2:
        match_dir.rename(matched_dir / f"{match_dir.name}-Tier2")

    # Rename tier 3 directories
    for match_dir in tier3:
        match_dir.rename(matched_dir / f"{match_dir.name}-Tier3")


    print("Done!")

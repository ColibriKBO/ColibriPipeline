# -*- coding: utf-8 -*-
"""
Created on Jul 29 10:04:14 2022

@author: Roman A.

Match dip detection txts throughout 3 telescopes based on time and coordinates, runs only on Green

2022-09-21 Roman A. simplified some steps and added data removal output
"""

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

#-------------------------------global vars-----------------------------------#

# Path variables
BASE_PATH = Path('D:/')
DATA_PATH = BASE_PATH / 'ColibriData'
IMGE_PATH = BASE_PATH / 'ColibriImages'
ARCHIVE_PATH = BASE_PATH / 'ColibriArchive'

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
            self.det_list = list(self.obs_archive.glob("det_*.txt"))
        else:
            print(f"ERROR: No archive found for {self.name} on {obs_date}!")
            self.det_list = []




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
                star_coords = line.split(':')[1].split(' ')[1:3]
                star_ra = float(star_coords[0])
                star_dec = float(star_coords[1])
                
       

    
    return (star_ra,star_dec)


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


#-------------------------------old------------------------------------------#
    arg_parser = argparse.ArgumentParser(description=""" Run secondary Colibri processing
        Usage:

        """,
        formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('-d', '--date', help='Observation date (YYYY/MM/DD) of data to be processed.')

    cml_args = arg_parser.parse_args()
    obsYYYYMMDD = cml_args.date
    obs_date = date(int(obsYYYYMMDD.split('/')[0]), int(obsYYYYMMDD.split('/')[1]), int(obsYYYYMMDD.split('/')[2]))
        


    night_dir=str(obs_date)
    green_path=Path('/','D:','/ColibriArchive',night_dir) #path for Green pipeline results
    red_path=Path('/','Y:',night_dir) #path for Red pipeline results
    blue_path=Path('/','Z:',night_dir)    #path for Blue pipeline results

    coord_tol=0.001 #max coordinates difference for matching
    milisec_tol=0.21 #max milliseconds window for matching


    #create directory for matched occultations

    if not os.path.exists(green_path.joinpath('matched')):
        os.mkdir(green_path.joinpath('matched'))
        
    matched_dir=green_path.joinpath('matched')

    if not os.path.exists(green_path.joinpath('milisec_unmatched')):
        os.mkdir(green_path.joinpath('milisec_unmatched'))
        
    milisec_unmatched=green_path.joinpath('milisec_unmatched')

    if not os.path.exists(green_path.joinpath('coords_unmatched')):
        os.mkdir(green_path.joinpath('coords_unmatched'))
        
    coords_unmatched=green_path.joinpath('coords_unmatched')



    #take occultation time from each txt file from each telescope
    print("Reading data...")# 21-09 Roman A. commented out unnecessary lines
    
    '''==========================================================#
    # PQuigley vanity project - Please ignore this snippet for now
    minutes = []
    radec   = []
    for telescope_path in (red_path,green_path,blue_path):
        # If the archive for that night exists, search it for det files
        if telescope_path.exists():
            det_files = telescope_path.glob("det_*.txt")

            # Get HHMMSS time for each det file from the name and the RA/DEC of the detection from the body
            det_times = []
            det_radec = []
            for det in det_files:
                det_times.append(re.findall("_(\d{6})_", det.name)[0])
                det_radec.append(getRAdec(det))

            # Append to greater list
            minutes.append(det_times)
            radec.append(det_radec)
        else:
            minutes.append([])
            radec.append([])
    #==============================================================='''

    Green_minutes=[minute for file in sorted(os.listdir(green_path)) if 'det' in file for minute in re.findall("_(\d{6})_", file)]
    # Green_stars=[minute for file in sorted(os.listdir(green_path)) if 'det' in file for minute in re.findall("(star\w{3})", file)]

    Red_minutes=[minute for file in sorted(os.listdir(red_path)) if 'det' in file for minute in re.findall("_(\d{6})_" , file)]
    # Red_stars=[minute for file in sorted(os.listdir(red_path)) if 'det' in file for minute in re.findall("(star\w{3})", file)]

    Blue_minutes=[minute for file in sorted(os.listdir(blue_path)) if 'det' in file for minute in re.findall("_(\d{6})_", file)]
    # Blue_stars=[minute for file in sorted(os.listdir(blue_path)) if 'det' in file for minute in re.findall("(star\w{3})", file)]



    '''------------------------------------1-second MATCHING--------------------------------'''
    print("Matching in time ...")
    start_time = time.time()#09-21 Roman A. commented out bad coding

    ##create intersections of hhmmss of detection files to find common elements
    GR=set(Green_minutes).intersection(Red_minutes)

    RB=set(Red_minutes).intersection(Blue_minutes)

    BG=set(Blue_minutes).intersection(Green_minutes)


    pattern=sorted(GR | RB | BG) #a list to store hhmmss that are common for telescopes

    if len(pattern)==0:
        print("No time matches today!")
        sys.exit()

    
    # Create dir for each hhmmss signature that was matched
    for hhmmss in set(pattern):
        Path(matched_dir / hhmmss).mkdir()

    '''loop through each telescope dir and copy time matched events to Matched directory'''

    for root, subdirs, filename in os.walk(red_path):
        #loop through all detection files and find hhmmss pattern in them
        for i in range(len(filename)):
            if 'det_' in filename[i]:
                for k in range(len(pattern)):
                    if pattern[k] in re.findall("_(\d{6})_", str(filename[i]))[0]: #copy det_ txts that have specific hhmmss
                        try:
                            shutil.copy(red_path.joinpath(filename[i]), matched_dir / pattern[k])
                            shutil.copy(red_path.joinpath(filename[i]), matched_dir)
                        except FileNotFoundError:
                            pass
                        #print(filename[i])
                        break
                
    for root, subdirs, filename in os.walk(green_path):
        
        for i in range(len(filename)):
            if 'det_' in filename[i]:
                for k in range(len(pattern)):
                    if pattern[k] in re.findall("_(\d{6})_", str(filename[i]))[0]:
                        
                        try:  
                            shutil.copy(green_path.joinpath(filename[i]), matched_dir / pattern[k])
                            shutil.copy(green_path.joinpath(filename[i]), matched_dir)
                        except FileNotFoundError:
                            pass
                        #print(filename[i])
                        break
                
                
    for root, subdirs, filename in os.walk(blue_path):
        
        for i in range(len(filename)):
            if 'det_' in filename[i]:
                for k in range(len(pattern)):
                    if pattern[k] in re.findall("_(\d{6})_", str(filename[i]))[0]:
                        
                        try:
                            shutil.copy(blue_path.joinpath(filename[i]), matched_dir / pattern[k])
                            shutil.copy(blue_path.joinpath(filename[i]), matched_dir)
                        except FileNotFoundError:
                            pass
                        #print(filename[i])
                        break



    '''------------------------------------millisecond MATCHING--------------------------------'''
    print("Matching in milliseconds...")

    milsec_match=[[]] #list to store 


    ##loop through each pattern of hhmmss and compare their milliseconds time with combinatorics

    for hhmmss in pattern:
        milsecond_couple=[]
        matched_ab=[]
        for file in sorted(matched_dir.iterdir()):
            
            if (hhmmss in file.name) and (file.is_file()):
                for minute in re.findall("_(\d{9})_", file.name):
                    milsecond_couple.append(float('.'+ minute))

        for a, b in itertools.combinations(milsecond_couple, 2):  
            if isclose(a, b,  abs_tol=milisec_tol):
    
                milsec_match.append([a,b]) 
            
    del(milsec_match[0]) #delete first row of zeroes
    milsec_match = list(np.unique(np.concatenate(milsec_match).flat))
    for n in range(len(milsec_match)):
        milsec_match[n]=str(milsec_match[n]).replace('0.','')
        if len(milsec_match[n])<9:
            milsec_match[n]=milsec_match[n]+'0'

    for matchedTXT in matched_dir.iterdir():
        if matchedTXT.is_dir():
            continue

        if re.findall("_(\d{9})_", str(matchedTXT))[0] not in milsec_match:
            
            if os.path.exists(matchedTXT):
                
                shutil.move(str(matchedTXT), str(milisec_unmatched)) 
                
            else:
                print("The file does not exist")

    print("Time matched in --- %s seconds ---" % (time.time() - start_time))
    print('')

    '''---------------------------------------Coordinates Calculation-----------------------------------------'''

    ##skipped because coordinates are now calculated using coordsfinder.py after main_pipeline

    '''---------------------------------------Coordinates Matching-----------------------------------------'''

    #list of matched time patterns
    #pattern=list(np.unique([minute for file in sorted(os.listdir(matched_dir)) if 'det' in file for minute in re.findall("_(\d{6})_", file)]))
    matched_files = sorted(matched_dir.glob('det_*.txt'))
    pattern = list(set([re.findall("_(\d{6})_", file.name)[0] for file in matched_files]))

    start_time = time.time()
    print("Matching coordinates...")
    time_keys=np.array(pattern) #list of unique matched times
    print('times matched:',len(time_keys) )
    print(time_keys)
    coords_match=[[]]


    #read RA and dec from each file
    for times in time_keys:
        time_matched_tuple=sorted(matched_dir.glob('*%s*.txt'%(times))) #list of matched det_s
        star_ra=[]
        star_dec=[]
        for star_path in time_matched_tuple:
            ra,dec=readRAdec(star_path) #read star coords from file
            star_ra.append(ra)
            star_dec.append(dec)
            star_radec=list(zip(star_ra,star_dec)) #array of Ra and dec
        for a, b in itertools.combinations(star_radec, 2): #compare RA dec coordinates of each event with combinatorics
            if isclose(a[0], b[0],  abs_tol=coord_tol/((math.cos(np.radians(a[1]))+math.cos(np.radians(a[1])))/2)) and isclose(a[1], b[1],  abs_tol=coord_tol):
                coords_match.append([a,b]) #array of matching coords
                #print(a,' - ',b)
    del(coords_match[0]) #delete first row of zeroes

    try: #list of matching coordinates
        lst=[item for sublist in coords_match for item in sublist] #there is index error when it's empty
    except IndexError:
        print('No matches today!')
        sys.exit()


    #Read RA and dec from files and compare them with list of matched coordinates, delete file if it doesn't have those coordinates
    for matchedTXT in matched_files: 
        star_ra=[]                          
        star_dec=[]
        ra,dec=readRAdec(matchedTXT)
        star_ra.append(ra)
        star_dec.append(dec)
        star_radec=list(zip(star_ra,star_dec))[0] #array of Ra and dec
        
        if star_radec not in lst: #if coords in txt file don't match coords in matched coords list
            if os.path.exists(matchedTXT):
                
                shutil.move(str(matchedTXT), str(coords_unmatched)) 
            
            else:
                print("The file does not exist")
        
    print("Coordinates matched in --- %s seconds ---" % (time.time() - start_time))
    print('')

    '''------------------------------------SECONDARY millisecond MATCHING--------------------------------'''
    ##Is done for events that were removed by coordinates matching
    print("Matching in milliseconds again...")

    matched_files = sorted(matched_dir.glob('det_*.txt'))
    pattern = list(set([re.findall("_(\d{6})_", file.name)[0] for file in matched_files]))
    milsec_match=[[]]

    for k in pattern:
        milsecond_couple=[]
        matched_ab=[]
        for file in matched_files:
            
            if k in file:

                for minute in re.findall("_(\d{9})_", file):
                    milsecond_couple.append(float('.'+ minute))

        for a, b in itertools.combinations(milsecond_couple, 2):
    
            if isclose(a, b,  abs_tol=milisec_tol):
                
                milsec_match.append([a,b]) 

            
    del(milsec_match[0]) #delete first row of zeroes
    milsec_match = list(np.unique(np.concatenate(milsec_match).flat))
    for n in range(len(milsec_match)):
        milsec_match[n]=str(milsec_match[n]).replace('0.','')
        if len(milsec_match[n])<9:
            milsec_match[n]=milsec_match[n]+'0'
            
    for matchedTXT in matched_files:
        if re.findall("_(\d{9})_", str(matchedTXT))[0] not in milsec_match: 
            if os.path.exists(matchedTXT):
                #os.unlink(matchedTXT)
                shutil.move(str(matchedTXT), str(milisec_unmatched)) 
                #print("moved away ",matchedTXT)
            else:
                print("The file does not exist")
            
    print("done")

    '''-------------------------------CREATING A LIST TO DELETE DATA WITHOUT EVENTS-------------------------'''    
        
    ##Path to ColibriData of 3 telescopes
    green_minutesdir=Path('/','D:','/ColibriData',str(night_dir).replace('-', ''))
    blue_minutesdir=Path('/','B:',str(night_dir).replace('-', ''))
    red_minutesdir=Path('/','R:',str(night_dir).replace('-', ''))

    #dictionary for looping porpuses
    telescope_dirs=[[green_minutesdir,'GREENBIRD'], [red_minutesdir,'REDBIRD'], [blue_minutesdir,'BLUEBIRD']]

    #time formats to read and compare time of detections and minute_dirs 
    format_data="%H.%M.%S.%f"
    format_mins="%H%M%S"

    matched_files = sorted(matched_dir.glob('det_*.txt'))
    detections = [f for f in matched_files ] #list of detections by one telescope
    
    detection_minutes=[] #list of detected files minutes
    for det in detections:
        time=det.name.split('_')[2]
        detection_minutes.append(datetime.strptime(time, format_mins))
    detection_minutes=np.unique(detection_minutes)

    for night_dir in telescope_dirs:
        
        print(night_dir)

        #subdirectories of minute-long datasets (~2400 images per directory)
        minute_dirs = [f for f in night_dir[0].iterdir() if f.is_dir()]  
        
        
        #remove bias directory from list of image directories and sort
        minute_dirs = [f for f in minute_dirs if 'Bias' not in f.name]
        
        minute_dirs.sort() #list of minute folders
        
        
        
        minute_names=[] #list of minutes in minute folders names
        
        for dirs in minute_dirs:
            
            minute_names.append(datetime.strptime(dirs.name.split('_')[1],format_data))
    
            #create a file with minute_dirs that need to be saved
        savefile=night_dir[0].joinpath('to_be_saved.txt') 
        
        
        with open(savefile, 'w') as filehandle:
            for det_time in detection_minutes:
                
                for i in range(len(minute_names)):
        
                    if det_time<=minute_names[i]:
                    
                
                        filehandle.write('%s\n' %(minute_names[i]))
                        
                        try:
                            #save aditional minute_dir just in case
                            filehandle.write('%s\n' %(minute_names[i-1]))
                        except IndexError:
                            break
                        break
        
        
                    
    print('DONE!')
                
            
                
        




















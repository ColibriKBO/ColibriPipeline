# -*- coding: utf-8 -*-
"""
Created on Jul 29 10:04:14 2022

@author: Roman A.
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

def readRAdec(filepath):                                    #modified readFile from Colibri Pipeline
    '''read in a .txt detection file and get information from it'''
    
    #make dataframe containing image name, time, and star flux value for the star
    starData = pd.read_csv(filepath, delim_whitespace = True, 
           names = ['filename', 'time', 'flux'], comment = '#')

    first_frame = int(starData['filename'][0].split('_')[-1].split('.')[0])
    
    #get header info from file
    with filepath.open() as f:
        
        #loop through each line of the file
        for i, line in enumerate(f):
            

            #get event frame number
            if i == 4:
                event_frame = int(line.split('_')[-1].split('.')[0])

            #get star coords
            
                
            elif i==6:
                star_coords = line.split(':')[1].split(' ')[1:3]
                star_ra = float(star_coords[0])
                star_dec = float(star_coords[1])
                
        #reset event frame to match index of the file
        event_frame = event_frame - first_frame

    #return    
    return (star_ra,star_dec)
    #return 

def readFile(filepath):                                      #redifinition from Colibri Pipeline's function
    '''read in a .txt detection file and get information from it'''
    
    #make dataframe containing image name, time, and star flux value for the star
    starData = pd.read_csv(filepath, delim_whitespace = True, 
           names = ['filename', 'time', 'flux'], comment = '#')

    first_frame = int(starData['filename'][0].split('_')[-1].split('.')[0])
    
    #get header info from file
    with filepath.open() as f:
        
        #loop through each line of the file
        for i, line in enumerate(f):
            

            #get event frame number
            if i == 4:
                event_frame = int(line.split('_')[-1].split('.')[0])

            #get star coords
            elif i == 5:
                star_coords = line.split(':')[1].split(' ')[1:3]
                star_x = float(star_coords[0])
                star_y = float(star_coords[1])
            
            #get event time
            elif i == 7:
                event_time = line.split('T')[2].split('\n')[0]
                
            elif i == 10:
                event_type = line.split(':')[1].split('\n')[0].strip(" ")
                
            elif i == 11:
                star_med = line.split(':')[1].split('\n')[0].strip(" ")
                
            elif i == 12:
                star_std = line.split(':')[1].split('\n')[0].strip(' ')
                
        #reset event frame to match index of the file
        event_frame = event_frame - first_frame

    #return all event data from file as a tuple    
    return (starData, event_frame, star_x, star_y, event_time, event_type, star_med, star_std)
    #return starData, event_frame, star_x, star_y, event_time, event_type, star_med, star_std
    
def getTransform(date):                                     #redifinition from Colibri Pipeline's function
    '''get astrometry.net transform for a given minute'''

    #if transformation has already been calculated, get from dictionary
    if date in transformations:
        return transformations[date]
    
    #calculate new transformation from astrometry.net
    else:
        #get median combined image
        median_image = [f for f in median_combos if date in f.name][0]
        
        #get name of transformation header file to save
        transform_file = median_image.with_name(median_image.name.strip('_medstacked.fits') + '_wcs.fits')
        
        #check if the tranformation has already been calculated and saved
        if transform_file.exists():
            
            #open file and get transformation
            wcs_header = fits.open(transform_file)
            transform = wcs.WCS(wcs_header[0])

        #calculate new transformation
        else:
            #get WCS header from astrometry.net plate solution
            soln_order = 4
            wcs_header = astrometrynet_funcs.getSolution(median_image, transform_file, soln_order)
        
            #calculate coordinate transformation
            transform = wcs.WCS(wcs_header)
        
        #add to dictionary
        transformations[date] = transform
        
        return transform
    
'''---------------------------------------------SCRIPT STARTS HERE--------------------------------------'''

#base_path=Path('/', 'C:','\\Users', 'Admin', 'Desktop', 'Colibri', 'occultations') #general path, probably a nightdir in a ColibriArchive

#green_path=base_path.joinpath('GREEN') #path for Green pipeline results
#red_path=base_path.joinpath('RED')  #path for Red pipeline results
#blue_path=base_path.joinpath('BLUE')    #path for Blue pipeline results
night_dir="2022-08-12"
green_path=Path('/','D:','/ColibriArchive',night_dir) #path for Green pipeline results
red_path=Path('/','Y:',night_dir) #path for Red pipeline results
blue_path=Path('/','Z:',night_dir)    #path for Blue pipeline results

coord_tol=0.001 #max coordinates difference for matching
milisec_tol=0.21 #max milliseconds window for matching

#matched_dir=base_path.joinpath('matched') #create directory for matched occultations

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


# GreenList = sorted(green_path.glob('*Green.txt'))
# RedList = sorted(red_path.glob('*Red.txt'))
# BlueList = sorted(blue_path.glob('*Blue.txt'))

#take star index and occultation time from each txt file from each telescope
print("Reading data...")
Green_minutes=[minute for file in sorted(os.listdir(green_path)) if 'det' in file for minute in re.findall("_(\d{6})_", file)]
Green_stars=[minute for file in sorted(os.listdir(green_path)) if 'det' in file for minute in re.findall("(star\w{3})", file)]

Red_minutes=[minute for file in sorted(os.listdir(red_path)) if 'det' in file for minute in re.findall("_(\d{6})_" , file)]
Red_stars=[minute for file in sorted(os.listdir(red_path)) if 'det' in file for minute in re.findall("(star\w{3})", file)]

Blue_minutes=[minute for file in sorted(os.listdir(blue_path)) if 'det' in file for minute in re.findall("_(\d{6})_", file)]
Blue_stars=[minute for file in sorted(os.listdir(blue_path)) if 'det' in file for minute in re.findall("(star\w{3})", file)]

#create array with star number and occultation time
Greens=np.column_stack((Green_stars,Green_minutes))
Reds=np.column_stack((Red_stars,Red_minutes))
Blues=np.column_stack((Blue_stars,Blue_minutes))


'''------------------------------------1-second MATCHING--------------------------------'''
print("Matching in 1 second window...")
start_time = time.time()
#match Green and Red times
GRMatched=np.empty((1,2),dtype=np.string_) #create an empty 2D array of green and red matches
for i in range(len(Greens)):
    for k in range(len(Reds)):
        
        if (Greens[i][1]==Reds[k][1]):
            
            try:
                GRMatched=np.vstack([GRMatched,Greens[i]]) #if no matches in Red and Green python will rise an 
                                                #error that it cannot read random byte in empty array
            except UnicodeDecodeError:
                print('No matches today!')
                sys.exit()
                
            
            
GRMatched=np.delete(GRMatched,(0),axis=0) #delete first randomly filled row
GRMatched=np.unique(GRMatched,axis=0) #delete repeating matches

total_match=np.empty((1,2),dtype=np.string_) #create an empty array of grenn red blue time matches

second_matched=0
for i in range(len(GRMatched)):
    for k in range(len(Blues)):
        
        if (GRMatched[i][1]==Blues[k][1]):
            second_matched+=1
            total_match=np.vstack([total_match,GRMatched[i]])
            
print("one second window matches: ",second_matched)            
total_match=np.delete(total_match,(0),axis=0) #delete first randomly filled row
total_match=np.unique(total_match,axis=0)  #delete repeating matches

#key_name=total_match.flatten(order='C')

#check if there are any time matches
if len(total_match)==0: #this list is empty only if there are no matches between red green and blue
    print("No time matches today!")
    sys.exit()

pattern=[] #create list of time strings that match
for i in range(len(total_match)):
    pattern.append(str(total_match[i,1]))
pattern=list(np.unique(pattern,axis=0))
    
'''loop through each telescope dir and copy time matched events to Matched directory'''

for root, subdirs, filename in os.walk(red_path):

    for i in range(len(filename)):
        if 'det_' in filename[i]:
            for k in range(len(pattern)):
                if pattern[k] in re.findall("_(\d{6})_", str(filename[i]))[0]:
                    try:
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
                        shutil.copy(blue_path.joinpath(filename[i]), matched_dir)
                    except FileNotFoundError:
                        pass
                    #print(filename[i])
                    break

print("Time matched in --- %s seconds ---" % (time.time() - start_time))
print('')

'''------------------------------------millisecond MATCHING--------------------------------'''
print("Matching in milliseconds...")
#milisec_list=[minute for file in sorted(os.listdir(matched_dir)) if 'det' in file for minute in re.findall("_(\d{9})_", file)]
milsec_match=[[]]

#for k in pattern:
#    milsecond_couple=[milsec for file in sorted(os.listdir(matched_dir)) if k in file for milsec in re.findall("_(\d{9})_", file)]
#    for i in range(len(milsecond_couple)):
#        milsecond_couple[i]=float('.'+milsecond_couple[i])
#    for a, b in itertools.combinations(milsecond_couple, 2): #compare RA dec coordinates of each event 
#        if isclose(a, b,  abs_tol=0.2):
#            milsec_match.append(k) #array of matching coords
#            #print(a,' - ',b)
#del(milsec_match[0]) #delete first row of zeroes

for k in pattern:
    milsecond_couple=[]
    matched_ab=[]
    for file in sorted(os.listdir(matched_dir)):
        if k in file:
            for minute in re.findall("_(\d{9})_", file):
                milsecond_couple.append(float('.'+ minute))
    for a, b in itertools.combinations(milsecond_couple, 2):  
        if isclose(a, b,  abs_tol=milisec_tol):
#            matched_ab.append(a)
#            matched_ab.append(b)
#            matched_ab=list(np.unique(matched_ab,axis=0))  
            milsec_match.append([a,b]) 
        
del(milsec_match[0]) #delete first row of zeroes
milsec_match = list(np.unique(np.concatenate(milsec_match).flat))
for n in range(len(milsec_match)):
    milsec_match[n]=str(milsec_match[n]).replace('0.','')+'0'

for matchedTXT in matched_dir.iterdir():
    if re.findall("_(\d{9})_", str(matchedTXT))[0] not in milsec_match: 
        if os.path.exists(matchedTXT):
            #os.unlink(matchedTXT)
            shutil.move(str(matchedTXT), str(milisec_unmatched)) 
            #print("moved away ",matchedTXT)
        else:
            print("The file does not exist")


#for k in range(len(pattern)):
#    for file in sorted(os.listdir(matched_dir)):
#        
#        if pattern[k] in file:
#            print(file)
#            for milis in milsec_match[k]:
#               
#                if milsec_match[k]!=[]:
#                    
#                        if str(milis) not in file:
#                            if os.path.exists(matched_dir.joinpath(file)):
#                                #os.unlink(matched_dir.joinpath(file)) 
#                                print("deleted ",file)
                
                    
                    
                
    


'''---------------------------------------Coordinates Calculation-----------------------------------------'''

start_time = time.time()

telescope_tuple=[['REDBIRD',red_path], #for looping convinience
                 ['GREENBIRD',green_path],
                 ['BLUEBIRD',blue_path]]

for i in range(len(telescope_tuple)):

    detect_files = [f for f in matched_dir.iterdir() if str(telescope_tuple[i][0]) in f.name] #list of time matches
    
    ''' get astrometry.net plate solution for each median combined image (1 per minute with detections)'''
    median_combos = [f for f in telescope_tuple[i][1].iterdir() if 'medstacked' in f.name]
    
    #dictionary to hold WCS transformations for each transform file
    transformations = {}
    
    for filepath in detect_files:
        
        
    
        
        #read in file data as tuple containing (star lightcurve, event frame #, star x coord, star y coord, event time, event type, star med, star std)
        eventData = readFile(filepath)
    
        #get corresponding WCS transformation
        date = Path(eventData[0]['filename'][0]).parent.name.split('_')[1]
    
        transform = getTransform(date)
        
        #get star coords in RA/dec
        star_wcs = getRAdec.getRAdecSingle(transform, (eventData[2], eventData[3]))
        star_RA = star_wcs[0]
        star_DEC = star_wcs[1]
        
        #write a line with RA dec in each file
        with open(filepath, 'r') as filehandle:
            lines = filehandle.readlines()
            lines[6] = '#    RA Dec Coords: %f %f\n' %(star_RA, star_DEC)
        with open(filepath, 'w') as filehandle:
            filehandle.writelines( lines )
            
print("Coordinates calculated in --- %s seconds ---" % (time.time() - start_time))
print('')

'''---------------------------------------Coordinates Matching-----------------------------------------'''

start_time = time.time()
print("Matching coordinates...")
time_keys=np.unique(pattern,axis=0) #list of unique matched times
print('times matched:',len(time_keys) )
coords_match=[[]]
matched_files = os.listdir(matched_dir) #list of time matched files


#read RA and dec from each file
for times in time_keys:
    time_matched_tuple=sorted(matched_dir.glob('*%s*'%(times)))
    star_ra=[]
    star_dec=[]
    for star_path in time_matched_tuple:
        ra,dec=readRAdec(star_path)
        star_ra.append(ra)
        star_dec.append(dec)
        star_radec=list(zip(star_ra,star_dec)) #array of Ra and dec
    for a, b in itertools.combinations(star_radec, 2): #compare RA dec coordinates of each event 
        if isclose(a[0], b[0],  abs_tol=coord_tol) and isclose(a[1], b[1],  abs_tol=coord_tol):
            coords_match.append([a,b]) #array of matching coords
            #print(a,' - ',b)
del(coords_match[0]) #delete first row of zeroes

try: #list of matching coordinates
    lst=[item for sublist in coords_match for item in sublist] #there is index error when it's empty
except IndexError:
    print('No matches today!')
    sys.exit()

for matchedTXT in matched_dir.iterdir(): #Read RA and dec from files and compare them with list of matched
    star_ra=[]                          #coordinates, delete file if it doesn't have those coordinates
    star_dec=[]
    ra,dec=readRAdec(matchedTXT)
    star_ra.append(ra)
    star_dec.append(dec)
    star_radec=list(zip(star_ra,star_dec))[0] #array of Ra and dec
    
    if star_radec not in lst: #if coords in txt file don't match coords in matched coords list
        if os.path.exists(matchedTXT):
            #os.unlink(matchedTXT) #delete all files that don't have coords matched
            shutil.move(str(matchedTXT), str(coords_unmatched)) 
            #print("moved away ",matchedTXT)
        else:
            print("The file does not exist")
    else:
        print("WE HAVE A MATCH!")
    
print("Coordinates matched in --- %s seconds ---" % (time.time() - start_time))
print('')

'''------------------------------------SECONDARY millisecond MATCHING--------------------------------'''
print("Matching in milliseconds again...")
#milisec_list=[minute for file in sorted(os.listdir(matched_dir)) if 'det' in file for minute in re.findall("_(\d{9})_", file)]
matched_pattern=list(np.unique([minute for file in sorted(os.listdir(matched_dir)) if 'det' in file for minute in re.findall("_(\d{6})_", file)]))
milsec_match=[[]]

#for k in pattern:
#    milsecond_couple=[milsec for file in sorted(os.listdir(matched_dir)) if k in file for milsec in re.findall("_(\d{9})_", file)]
#    for i in range(len(milsecond_couple)):
#        milsecond_couple[i]=float('.'+milsecond_couple[i])
#    for a, b in itertools.combinations(milsecond_couple, 2): #compare RA dec coordinates of each event 
#        if isclose(a, b,  abs_tol=0.2):
#            milsec_match.append(k) #array of matching coords
#            #print(a,' - ',b)
#del(milsec_match[0]) #delete first row of zeroes

for k in matched_pattern:
    milsecond_couple=[]
    matched_ab=[]
    for file in sorted(os.listdir(matched_dir)):
        if k in file:
            for minute in re.findall("_(\d{9})_", file):
                milsecond_couple.append(float('.'+ minute))
    for a, b in itertools.combinations(milsecond_couple, 2):  
        if isclose(a, b,  abs_tol=milisec_tol):
#            matched_ab.append(a)
#            matched_ab.append(b)
#            matched_ab=list(np.unique(matched_ab,axis=0))  
            milsec_match.append([a,b]) 
        
del(milsec_match[0]) #delete first row of zeroes
milsec_match = list(np.unique(np.concatenate(milsec_match).flat))
for n in range(len(milsec_match)):
    milsec_match[n]=str(milsec_match[n]).replace('0.','')+'0'

for matchedTXT in matched_dir.iterdir():
    if re.findall("_(\d{9})_", str(matchedTXT))[0] not in milsec_match: 
        if os.path.exists(matchedTXT):
            #os.unlink(matchedTXT)
            shutil.move(str(matchedTXT), str(milisec_unmatched)) 
            print("moved away ",matchedTXT)
        else:
            print("The file does not exist")
         
    
    
    
            
        
            
    























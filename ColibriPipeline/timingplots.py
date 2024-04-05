#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 14:46:28 2021

@author: rbrown

makes plot of time of all images (from .rcd files)
"""

import sys
import matplotlib.pyplot as plt
from astropy.time import Time
import pathlib

# Function for reading specified number of bytes
def readxbytes(numbytes):
    '''Written by Mike Mazur
    Read in the specified number of bytes (for reading .rcd images)
    input: number of bytes [int]
    returns: data in file [array]'''

    for i in range(1):
        data = fid.read(numbytes)
        if not data:
            break
        return data

'''---------------------------------main program-----------------------------------'''

''' set up input/output '''

#name of parent directory to test on
#nightDir = pathlib.Path(sys.argv[1])
obs_date=input("Input obesrvation date (ex. 2022-07-30): ")  #17-07-2022 Roman A.
nightDir = pathlib.Path('/', 'D:/','ColibriData',str(obs_date).replace('-', ''))

#get list of subdirectories (minute directories)
minuteDirs = [d for d in nightDir.iterdir() if d.is_dir() and 'Dark' not in d.name and 'badimages' not in d.name ]

#path to save script output
outputPath = nightDir.parents[1].joinpath('ColibriArchive', nightDir.name + '_diagnostics', 'timingTests')

#make output directory if doesn't exist
if not outputPath.exists():
    outputPath.mkdir(parents=True, exist_ok=True)

#text file to save script printouts
outputFile = outputPath.joinpath('timingplots_output.txt')

''' check each minute for anomalies and make a timing plot'''

#loop through each directory
for minutePath in minuteDirs:

    with open(outputFile, 'a') as f:
        f.write('%s\n' % minutePath.name)
    
    #list of images in directory
    imageList = sorted(minutePath.glob('*.rcd'))
    
    #check if directory empty
    if not imageList:
        with open(outputFile, 'a') as f:
            f.write('Empty directory\n')
        continue

    frameList = []       #list to hold image frame numbers
    timeList = []        #list to hold image timestamps
    
    #minute current directory was created
    dirMinute = minutePath.name.split('_')[1].split('.')[1]

    #loop through each image
    for filepath in imageList:
        
        '''open file to get timestamp'''
        fid = open(filepath, 'rb')
        fid.seek(152,0)
        timestamp = readxbytes(30)
  
        '''check for corrupted files'''
        try:
            #get hour and minute of image, minute of directory
            hour = str(timestamp).split('T')[1].split(':')[0]
            fileMinute = str(timestamp).split(':')[1]

        #script will fail if there are no files in directory
        except:
            with open(outputFile, 'a') as f:
                f.write('Corrupted file: %s\n' % (filepath.name))
            continue
        
        
        '''check if hour is bad (>23:59), if so take hour from directory name'''
        if int(hour) > 23:
            
            with open(outputFile, 'a') as f:
                f.write('Bad Hour: %s - %s\n' % (filepath.name, hour))
            
            #get correct hour from directory name
            dirHour = int(filepath.parent.name.split('_')[1].split('.')[0])
        

            #check if hour has rolled over
            if int(fileMinute) < int(dirMinute):
                newImageHour = dirHour + 1          #add 1 if hour changed over during minute
                
                #if converting from local to UTC
                #newImageHour = dirHour + 4 + 1     #add 1 if hour changed over during minute
            
            else:
                newImageHour = dirHour              #set image hour to directory hour
                
                 #if converting from local to UTC
                #newImageHour = dirHour + 4         #set image hour to directory hour
           
            #check if new hour is greater than 23:00 (because of additions above)
            if newImageHour > 23:
                newImageHour = newImageHour - 24    #roll over hour
        
            #match new hour format to timestamp string
            newImageHour = str(newImageHour)
            newImageHour = newImageHour.zfill(2)
        
            #replace bad hour in timestamp string with new hour
            replaced = str(timestamp).replace('T' + hour, 'T' + newImageHour).strip('b').strip(' \' ')
        
            #encode into bytes
            #newTimestamp = replaced.encode('utf-8')
            timestamp = replaced
        
        '''add frame number and timestamp to lists'''

        frameList.append(filepath.name.split('_')[-1].strip('.rcd').lstrip('0'))
        timeList.append(str(timestamp).strip('b').strip('\'').strip('Z'))

    '''check if all times are in order'''
    
    zero = Time(timeList[0], precision = 9).unix    #starting time
    
    #get time elapsed since first frame for each time in list
    for i in range(0, len(timeList)):

        timeList[i] = Time(timeList[i], precision = 9).unix
        timeList[i] = timeList[i] - zero

    #check for time going backwards         
    for i in range(1, len(timeList)):
        if timeList[i] < timeList[i-1]:
            
            with open(outputFile, 'a') as f:
                f.write('GREAT SCOTT! %i %i\n' %(frameList[i], frameList[i-1]))

            #make plot of the event
            fig, ax = plt.subplots()
            
            ax.plot(frameList[i-50:i+50], timeList[i-50:i+50])
            ax.set_xticks(ax.get_xticks()[::10])
            ax.set_xlabel('frame number')
            ax.set_ylabel('seconds since beginning of minute')
            ax.set_title(minutePath.name)
            #labels = fileList[::200]
            #ax.set_xticklabels(labels, rotation = 45)
            
            saveFilepath = outputPath.joinpath('timeTravel_' + frameList[i] + '_' + minutePath.name + '.png')

            plt.savefig(saveFilepath)
            plt.close()
    
    '''make plot of header times for each frame'''
    fig, ax = plt.subplots()

    ax.plot(frameList, timeList)

    ax.set_xticks(ax.get_xticks()[::300])
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Seconds since beginning of minute')
    ax.set_title(minutePath.name)
    #labels = fileList[::200]
    #ax.set_xticklabels(labels, rotation = 45)
    
    saveFilepath = outputPath.joinpath('timestamps_' + minutePath.name + '.png')
        
    plt.savefig(saveFilepath)
    plt.close()


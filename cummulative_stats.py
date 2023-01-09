# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 14:39:39 2022

Cummulative statistics for detections and occultations

@author: Roman A.
"""

from pathlib import Path
import os
import shutil
import numpy as np
from datetime import datetime, date

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator

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

base_path=Path('/','D:')
#archive pathes
green_arch=base_path.joinpath('/ColibriArchive')
red_arch=Path('/','Z:',)
blue_arch=Path('/','Y:')

#create directories that will store all the data needed

cumm_dets=green_arch.joinpath('cummulative_detections')

if not os.path.exists(cumm_dets):
    os.mkdir(cumm_dets)
    
cumm_occults2=green_arch.joinpath('cummulative_occultations2')

if not os.path.exists(cumm_occults2):
    os.mkdir(cumm_occults2)
    
cumm_occults3=green_arch.joinpath('cummulative_occultations3')

if not os.path.exists(cumm_occults3):
    os.mkdir(cumm_occults3)
    
archives=[green_arch,red_arch,blue_arch]

#%% cummulative detections
for archive in archives:
    #sort out night dirs to reduce time
    nights=[f for f in archive.iterdir() if ('diagnostics' not in f.name and getNightTime(f)>date(2022, 9, 19) and 'cummulative' not in f.name and f.is_dir())]
    for night in nights:
        print(night)
        detections=[f for f in night.iterdir() if 'det' in f.name]
        if len(detections)==0:
            continue
        else:
            for detection in detections:
                with open(detection) as f:
                    if 'significance' in f.read():
                        try:
                            shutil.copy2(detection,os.path.join(cumm_dets,detection.name)) #copy detection files
                        except shutil.SameFileError:
                            pass

        print(night)
#%%  cummulative telescope matches    

#sort out night dirs to reduce time
nights=[f for f in green_arch.iterdir() if ('diagnostics' not in f.name and os.path.getctime(f)>1663609502 and 'cummulative' not in f.name and f.is_dir())]
for night in nights:
    if os.path.exists(night.joinpath('matched')):
        matched_dirs=[d for d in night.joinpath('matched').iterdir() if d.is_dir()]
        for d in matched_dirs:
            dets=[f for f in d.iterdir() if ('det' in f.name and '.txt' in f.name)]
            if len(dets)==3:
                for detection in dets:
                    try:
                        shutil.copy2(detection,os.path.join(cumm_occults3,detection.name)) #copy to 3-telescope candidate
                    except:
                        continue
                    
            elif len(dets)==2:
                for detection in dets:
                    try:
                        shutil.copy2(detection,os.path.join(cumm_occults2,detection.name)) #copy to 2-telescope candidate
                    except:
                        continue
            elif len(dets)==1:
                continue

#%% plot 1

detected_files=[f for f in cumm_dets.iterdir() if 'det' in f.name]

sigmas=[]
for file in detected_files:
    sigmas.append(readSigma(file))

sigmas=np.array(sigmas)
# q25, q75 = np.percentile(sigmas, [25, 75])
# bin_width = 2 * (q75 - q25) * len(sigmas) ** (-1/3)
# bins = round((sigmas.max() - sigmas.min()) / bin_width)
fig, ax = plt.subplots()

# We change the fontsize of minor ticks label 
bins=np.arange(6,12.25,0.25)

plt.hist(sigmas,bins=bins,log=False)
# ax.yaxis.set_ticks(np.arange(0, 3, 1))

plt.xticks(np.arange(6, 12.25, 0.5),fontsize=8)
# ax.tick_params(axis='both', which='major', labelsize=7)
ax.tick_params(axis='both', which='minor', labelsize=8)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
# plt.xlim([min(sigmas),12])
ax.set_xlabel('Signisifance')
ax.set_ylabel('number of events')
# ax.set_ylim([0,2])
formatter = ScalarFormatter()
formatter.set_scientific(False)
ax.yaxis.set_major_formatter(formatter)

#plt.ylim([1,600])
plt.title('cummulative number of detections: '+str(np.size(sigmas)))
# plt.title(' Number of occulatations: '+str(np.size(sigmas)))
plt.grid(which='both',axis='x')
plt.savefig(cumm_dets.joinpath("sigma.png"),dpi=300)
# notbase_path=Path('/','D:','/Colibri','Green')
operations_savepath=base_path.joinpath('/Logs','Operations')
plt.savefig(operations_savepath.joinpath("sigma_det.svg"),dpi=800) 
# plt.show()

plt.close()  

#%% plot 2

detected_files=[f for f in cumm_occults2.iterdir() if 'det' in f.name]

sigmas=[]
for file in detected_files:
    sigmas.append(readSigma(file))

sigmas=np.array(sigmas)
# q25, q75 = np.percentile(sigmas, [25, 75])
# bin_width = 2 * (q75 - q25) * len(sigmas) ** (-1/3)
# bins = round((sigmas.max() - sigmas.min()) / bin_width)
fig, ax = plt.subplots()

# We change the fontsize of minor ticks label 
bins=np.arange(6,12.25,0.25)

plt.hist(sigmas,bins=bins,log=False)
# ax.yaxis.set_ticks(np.arange(0, 3, 1))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.xticks(np.arange(6, 12.25, 0.5),fontsize=8)
# ax.tick_params(axis='both', which='major', labelsize=7)
ax.tick_params(axis='both', which='minor', labelsize=8)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
# plt.xlim([min(sigmas),12])
ax.set_xlabel('Signisifance')
ax.set_ylabel('number of events')
# ax.set_ylim([0,2])
formatter = ScalarFormatter()
formatter.set_scientific(False)
ax.yaxis.set_major_formatter(formatter)

#plt.ylim([1,600])
plt.title('cummulative number of occultations for 2 telescopes: '+str(np.size(sigmas)))
# plt.title(' Number of occulatations: '+str(np.size(sigmas)))
plt.grid(which='both',axis='x')
plt.savefig(cumm_dets.joinpath("sigma.png"),dpi=300)
# notbase_path=Path('/','D:','/Colibri','Green')
operations_savepath=base_path.joinpath('/Logs','Operations')
plt.savefig(operations_savepath.joinpath("sigma_2.svg"),dpi=800) 
# plt.show()

plt.close()

#%% plot 3        

detected_files=[f for f in cumm_occults3.iterdir() if 'det' in f.name]

sigmas=[]
for file in detected_files:
    sigmas.append(readSigma(file))

sigmas=np.array(sigmas)
# q25, q75 = np.percentile(sigmas, [25, 75])
# bin_width = 2 * (q75 - q25) * len(sigmas) ** (-1/3)
# bins = round((sigmas.max() - sigmas.min()) / bin_width)
fig, ax = plt.subplots()

# We change the fontsize of minor ticks label 
bins=np.arange(6,12.25,0.25)

plt.hist(sigmas,bins=bins,log=False)
# ax.yaxis.set_ticks(np.arange(0, 3, 1))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.xticks(np.arange(6, 12.25, 0.5),fontsize=8)
# ax.tick_params(axis='both', which='major', labelsize=7)
ax.tick_params(axis='both', which='minor', labelsize=8)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
# plt.xlim([min(sigmas),12])
ax.set_xlabel('Signisifance')
ax.set_ylabel('number of events')
# ax.set_ylim([0,2])
formatter = ScalarFormatter()
formatter.set_scientific(False)
ax.yaxis.set_major_formatter(formatter)

#plt.ylim([1,600])
plt.title('cummulative number of occultations for 3 telescopes: '+str(np.size(sigmas)))
# plt.title(' Number of occulatations: '+str(np.size(sigmas)))
plt.grid(which='both',axis='x')
plt.savefig(cumm_dets.joinpath("sigma.png"),dpi=300)
# notbase_path=Path('/','D:','/Colibri','Green')
operations_savepath=base_path.joinpath('/Logs','Operations')
plt.savefig(operations_savepath.joinpath("sigma_3.svg"),dpi=800) 
# plt.show()

plt.close()                        

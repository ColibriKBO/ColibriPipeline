# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 14:39:39 2022

@author: Roman A.
"""
import datetime as dt
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import PolyCollection
import pandas as pd
from scipy import interpolate
import numpy as np
import seaborn as sns
from matplotlib.dates import DateFormatter
# import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
import os
import csv
import numba as nb
# from datetime import timezone
from astropy import time
import shutil
import argparse
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator
from getStarHour import getStarHour
from astropy.time import Time 

#!!!
# import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)

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

def readSigma(filepath):
   
    with filepath.open() as f:
        
        #loop through each line of the file
        for i, line in enumerate(f):
            

            
            if i==10:
                sigma = float(line.split(':')[1])
 
    return (sigma)

def twilightTimes(julian_date, site=[43.0,-81.0]):
    n = np.floor(julian_date -2451545.0 + 0.0008)
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
    Line that matched patter and time in Log of that line.

    """
    
    if Break==False:
       
        messages=[]
        times=[]
        for line in log_list:
            
            for pat in pattern:
                if pat in line:
                    messages.append((line))
                    if int(line.split(" ")[3].split(':')[0])>20:
                        times.append(obs_date+' '+line.split(" ")[3])
                    else:
                        times.append(str(obs_date)+' '+line.split(" ")[3])
        
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

def readxbytes(fid, numbytes):
    for i in range(1):
        data = fid.read(numbytes)
        if not data:
            break
    return data
@nb.njit(nb.uint16[::1](nb.uint8[::1]),fastmath=True,parallel=True)

def nb_read_data(data_chunk):
    """data_chunk is a contigous 1D array of uint8 data)
    eg.data_chunk = np.frombuffer(data_chunk, dtype=np.uint8)"""
    #ensure that the data_chunk has the right length

    assert np.mod(data_chunk.shape[0],3)==0

    out=np.empty(data_chunk.shape[0]//3*2,dtype=np.uint16)
 #   image1 = np.empty((2048,2048),dtype=np.uint16)
 #   image2 = np.empty((2048,2048),dtype=np.uint16)

    for i in nb.prange(data_chunk.shape[0]//3):
        fst_uint8=np.uint16(data_chunk[i*3])
        mid_uint8=np.uint16(data_chunk[i*3+1])
        lst_uint8=np.uint16(data_chunk[i*3+2])

        out[i*2] =   (fst_uint8 << 4) + (mid_uint8 >> 4)
        out[i*2+1] = ((mid_uint8 % 16) << 8) + lst_uint8

    return out

def readRCDheader(filename):
    """
    Reads .rcd file
    
        Parameters:
            filename (str): Path to .rcd file
            
        Returns:
            table (arr): Table with image pixel data
            hdict (dic?): Header dictionary
    """

    hdict = {}

    with open(filename, 'rb') as fid:

        # Go to start of file
        fid.seek(0,0)

        # Serial number of camera
        fid.seek(63,0)
        hdict['serialnum'] = readxbytes(fid, 9)

        # Timestamp
        fid.seek(152,0)
        hdict['timestamp'] = readxbytes(fid, 29).decode('utf-8')

        # Load data portion of file

    return  hdict
    
def importTimesRCD(imagePaths, startFrameNum, numFrames):
    """
    Reads in frames from .rcd files starting at a specific frame
    
        Parameters:
            imagePaths (list): List of image paths to read in
            startFrameNum (int): Starting frame number
            numFrames (int): How many frames to read in
            bias (arr): 2D array of fluxes from bias image
            
        Returns:
            imagesData (arr): Image data
            imagesTimes (arr): Header times of these images
    """
    
    imagesTimes = []   #array to hold image times
    
    

    
    '''list of filenames to read between starting and ending points'''
    files_to_read = [imagePath for i, imagePath in enumerate(imagePaths) if i >= startFrameNum and i < startFrameNum + numFrames]
    
    for imagePath in files_to_read:
        
        

        header = readRCDheader(imagePath)
        headerTime = header['timestamp']

       

        #change time if time is wrong (29 hours)
        hour = str(headerTime).split('T')[1].split(':')[0]
        imageMinute = str(headerTime).split(':')[1]
        dirMinute = imagePath.parent.name.split('_')[1].split('.')[1]
        
        #check if hour is bad, if so take hour from directory name and change header
        if int(hour) > 23:
            
            #directory name has local hour, header has UTC hour, need to convert (+4)
            #for red: local time is UTC time (don't need +4)
            newLocalHour = int(imagePath.parent.name.split('_')[1].split('.')[0])
        
            if int(imageMinute) < int(dirMinute):
                #newUTCHour = newLocalHour + 4 + 1     #add 1 if hour changed over during minute
                newUTCHour = newLocalHour + 1
            else:
                #newUTCHour = newLocalHour + 4
                newUTCHour = newLocalHour
        
            #replace bad hour in timestamp string with correct hour
            newUTCHour = str(newUTCHour)
            newUTCHour = newUTCHour.zfill(2)
        
            replaced = str(headerTime).replace('T' + hour, 'T' + newUTCHour).strip('b').strip(' \' ')
        
            #encode into bytes
            headerTime = replaced


        
        imagesTimes.append(headerTime)

    
        
    return imagesTimes

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
    #reading first minute of the night on Red, if no data then switch to Green
    try:
        first_min=[f for f in red_datapath.iterdir() if obs_date.replace('-','') in f.name][0]
    except:
        first_min=[f for f in green_datapath.iterdir() if obs_date.replace('-','') in f.name][0]
    return int(first_min.name.split('_')[1].split('.')[0])

#%% GETTING OBSERVATION TIMES    

arg_parser = argparse.ArgumentParser(description=""" Run timeline process
    Usage:

    """,
    formatter_class=argparse.RawTextHelpFormatter)


arg_parser.add_argument('-d', '--date', help='Observation date (YYYY/MM/DD) of data to be processed.')


cml_args = arg_parser.parse_args()
obsYYYYMMDD = cml_args.date
obs_date = date(int(obsYYYYMMDD.split('/')[0]), int(obsYYYYMMDD.split('/')[1]), int(obsYYYYMMDD.split('/')[2]))
obs_date=str(obs_date)

base_path=Path('/','D:') #default data path
operations_savepath=base_path.joinpath('/Logs','Operations') #path for logging the results on Green

prev_date=getPrevDate(operations_savepath) #previous observations' date

#create a directory with previous observation results
try:
    old_data=operations_savepath.joinpath(prev_date)
    if not os.path.exists(old_data):
        os.mkdir(old_data)
    files = [file for file in os.listdir(operations_savepath) #list all files from previous results
             if (os.path.isfile(os.path.join(operations_savepath, file)) and '.html' not in file and 'sigma' not in file
                and 'icon' not in file)]
    
    for file in files: #move all old data to the designated folder
        shutil.move(os.path.join(operations_savepath, file),os.path.join(old_data, file))
    #copy html file for future analisys    
    shutil.copy2(os.path.join(operations_savepath, 'observation_results.html'),os.path.join(old_data, 'observation_results.html'))
except TypeError:
    print("no previous data")

#night directories for 3 telescopes
red_datapath=Path('R:/ColibriData/'+obs_date.replace('-',''))
green_datapath=base_path.joinpath('/ColibriData',obs_date.replace('-',''))
blue_datapath=Path('B:/ColibriData/'+obs_date.replace('-',''))



datapaths=[[green_datapath,'GREENBIRD'], [red_datapath,'REDBIRD'], [blue_datapath,'BLUEBIRD']]

tables=[]

for data_path in datapaths:

    try:
        minutes=[f for f in data_path[0].iterdir() if obs_date.replace('-','') in f.name] #get a list of minute-folders
    except FileNotFoundError:
        print(str(data_path[0]), ' does not exist')
        continue


    minute_start=[]
    minute_end=[]
    for minute in minutes: #loop through all minute-folders and get the time of first and last frame in the folder
        imagePaths = sorted(minute.glob('*.rcd'))
        minute_start.append(importTimesRCD(imagePaths, 0, 1)[0])
        minute_end.append(importTimesRCD(imagePaths, len(imagePaths)-1, 1)[0])
    
    table=[minute_start,minute_end,['observing']*len(minute_end),[data_path[1]]*len(minute_end)] #create a table of times
    table=zip(*table)
    tables.append(table)

fields = ['Start', 'End', 'Event', 'telescope']  #csv headers
    

filename = operations_savepath.joinpath(obs_date+ "_timeline.csv") #file that stores observation times for this 
                                                                    #night for 3 telescopes
# writing times to csv file that will be ploted 
with open(filename, 'w',newline='') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
        
    # writing the headers 
    csvwriter.writerow(fields) 
    
    for table in tables:
    # writing the data rows 
        csvwriter.writerows(table)
        
 #%% GETTING CLOUD/TRANSPERENCY DATA   

# Cloud_logpath=base_path.joinpath('/Logs','Weather','CloudMonitor')
Cloud_logpath=base_path.joinpath('/Logs','Weather','Weather') #path for Polaris data on Green

todayT=datetime.strptime(obs_date, '%Y-%m-%d').date() #date of tonight's observations
yesterdayT=todayT-timedelta(days=1) #date of yesterday
tomorrowT=todayT+timedelta(days=1) #date of tomorrow
#these dates are needed because observations take place through out 2 days 

cloud_logs=[log for log in Cloud_logpath.iterdir() if obs_date in log.name or str(yesterdayT) in log.name or str(tomorrowT) in log.name]

#%% GETTING LOG EVENTS

#pathes for ACP logs created by RunColibri.js for 3 telescopes
red_ACPpath=Path('/','R:','/Logs','ACP')
green_ACPpath=base_path.joinpath('/Logs','ACP')
blue_ACPpath=Path('/','B:''/Logs','ACP')

ACP_logpaths=[red_ACPpath,green_ACPpath,blue_ACPpath]
color=['r','g','b'] #color list for ploting

    
#%% SNR V GMAG 

#pathes for sensitivity measurements of 3 telescopes

red_sens=Path('/','Y:','/'+obs_date.replace('-','')+'_diagnostics','Sensitivity')

green_sens=base_path.joinpath('/ColibriArchive',obs_date.replace('-','')+'_diagnostics','Sensitivity')

blue_sens=Path('/','Z:','/'+obs_date.replace('-','')+'_diagnostics','Sensitivity')

#path for sensitivity results folder
try:    
    red_sens=[f for f in red_sens.iterdir()][0]
except (FileNotFoundError, IndexError):
    print("Red sensitivity data is not available!")
try:
    green_sens=[f for f in green_sens.iterdir()][0]
except (FileNotFoundError, IndexError):
    print("Green sensitivity data is not available!")
try:
    blue_sens=[f for f in blue_sens.iterdir()][0]
except (FileNotFoundError, IndexError):
    print("Blue sensitivity data is not available!")


#reading starTable txt that contains SNR and GMAG, saving those tables
try:

    red_table=pd.read_csv([f for f in red_sens.iterdir() if 'starTable' in f.name][0], 
            names = [ 'X', 'Y', 'ra', 'dec' ,'GMAG', 'Gaia_RA' ,'Gaia_dec', 'Gaia_B-R', 'med' ,'std', 'SNR'],
            sep = ' ',header=0,index_col=0)
    
    
    starTxtFile = sorted(red_sens.joinpath('high_4sig_lightcurves').glob('*.txt'))[0]

    #get header info from file
    with starTxtFile.open() as f:
            
        #loop through each line of the file
        for i, line in enumerate(f):
                
            #get event time
            if i == 6:
                timestamp = line.split(' ')[-1].split('\n')[0]
    
    red_airmass = getAirmass(timestamp, red_table['ra'][int(len(red_table['ra']) / 2)], 
                             red_table['dec'][int(len(red_table['dec']) / 2)])[0]
    red_snrtime=timestamp.split('T')[1].split('.')[0]
    
except (FileNotFoundError, IndexError):
    red_table=[]
    red_snrtime=''
    red_airmass=0
    print("Red sensitivity data is not available!")
    
    
try:
    blue_table=pd.read_csv([f for f in blue_sens.iterdir() if 'starTable' in f.name][0], 
            names = [ 'X', 'Y', 'ra', 'dec' ,'GMAG', 'Gaia_RA' ,'Gaia_dec', 'Gaia_B-R', 'med' ,'std', 'SNR'], 
            sep = ' ',header=0,index_col=0)
    
    starTxtFile = sorted(blue_sens.joinpath('high_4sig_lightcurves').glob('*.txt'))[0]

    #get header info from file
    with starTxtFile.open() as f:
            
        #loop through each line of the file
        for i, line in enumerate(f):
                
            #get event time
            if i == 6:
                timestamp = line.split(' ')[-1].split('\n')[0]
    
    
    blue_snrtime=timestamp.split('T')[1].split('.')[0]
    blue_airmass = getAirmass(timestamp, blue_table['ra'][int(len(blue_table['ra']) / 2)], 
                             blue_table['dec'][int(len(blue_table['dec']) / 2)])[0]
    
except (FileNotFoundError, IndexError):
    blue_table=[]
    blue_snrtime=''
    blue_airmass=0
    print("Blue sensitivity data is not available!")

try:
    green_table=pd.read_csv([f for f in green_sens.iterdir() if 'starTable' in f.name][0],
            names = [ 'X', 'Y', 'ra', 'dec' ,'GMAG', 'Gaia_RA' ,'Gaia_dec', 'Gaia_B-R', 'med' ,'std', 'SNR'], 
            sep = ' ',header=0,index_col=0)
    
    starTxtFile = sorted(green_sens.joinpath('high_4sig_lightcurves').glob('*.txt'))[0]

    #get header info from file
    with starTxtFile.open() as f:
            
        #loop through each line of the file
        for i, line in enumerate(f):
                
            #get event time
            if i == 6:
                timestamp = line.split(' ')[-1].split('\n')[0]
    
    green_snrtime=timestamp.split('T')[1].split('.')[0]
    green_airmass = getAirmass(timestamp, green_table['ra'][int(len(green_table['ra']) / 2)], 
                             green_table['dec'][int(len(green_table['dec']) / 2)])[0]
except (FileNotFoundError, IndexError):
    green_table=[]
    green_snrtime=''
    green_airmass =0
    print("Green sensitivity data is not available!")


# %% Sunset Sunrise Time

now=time.Time(obs_date)
now.format='jd'

time_of_first_min=ToFM() #get hour time of observations.
#This is needed in order to adjust times in cases when obs_date is 08-16 but observations started on 08-15 and oposite 

if time_of_first_min>20:
    sunset=twilightTimes(now.value+0.5)[1]

    sunrise=twilightTimes(now.value+1.5)[0]

else:
    
    sunset=twilightTimes(now.value+0)[1]

    sunrise=twilightTimes(now.value+1)[0]
    

sunrise=time.Time(sunrise, format='jd') #sunrise time for this night
sunrise.format='fits'

sunset=time.Time(sunset, format='jd') #sunset time for this night
sunset.format='fits'

#add sunrise and sunset time to the timeline csv for future ploting
# with open(filename, 'a',newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     csvwriter.writerow([sunset, sunrise,'nighting','Night']) 


# %% 

'''--------------------------------------------------PLOTTING PART--------------------------------------------'''

#%% 1st GRAPH OF THE TIMELINE
fig123, (ax1, ax3) = plt.subplots(2, 1) #create a plot that will contain timeline bars + transperancy + log events

df = pd.read_csv(filename, delimiter=',')

data = [list(row) for row in df.values]


telescopes = {"GREENBIRD" : 1, "REDBIRD" : 2, "BLUEBIRD" : 3} #bars are located in ascending order
colormapping = {"GREENBIRD" : "#66ff66", "REDBIRD" : "r", "BLUEBIRD" : "b"}

verts = []
colors = []
for d in data:
    
    v =  [(mdates.datestr2num(d[0]), telescopes[d[3]]-.4),
          (mdates.datestr2num(d[0]), telescopes[d[3]]+.4),
          (mdates.datestr2num(d[1]), telescopes[d[3]]+.4),
          (mdates.datestr2num(d[1]), telescopes[d[3]]-.4),
          (mdates.datestr2num(d[0]), telescopes[d[3]]-.4)]
    verts.append(v)
    colors.append(colormapping[d[3]])

bars = PolyCollection(verts, facecolors=colors)


ax1.add_collection(bars)
ax1.autoscale()
ax1.zorder=5
ax1.patch.set_alpha(0.01)
loc = mdates.HourLocator(interval=1)
ax1.xaxis.set_major_locator(loc)
xfmt = DateFormatter('%H')
ax1.xaxis.set_major_formatter(xfmt)
ax1.set_xlim([mdates.datestr2num(str(sunset)), mdates.datestr2num(str(sunrise))])
ax1.set_yticks([1,2,3])
ax1.set_ylim(top=3+.4)
ax1.set_yticklabels(['Green','Red','Blue'])
ax1.xaxis.set_tick_params(labelsize=9)
ax1.xaxis.grid(True)
ax1.xaxis.tick_top()

#%% CLOUD/TRANSPERANCY PLOT


# today = pd.read_csv(cloud_logs[1])
today = pd.read_csv(cloud_logs[1], header=None, usecols=[0,9])
# today.columns = ['timestamp','SkyT','GroundT']
today.columns = ['timestamp','Mag']
# yesterday=pd.read_csv(cloud_logs[0])
yesterday = pd.read_csv(cloud_logs[0], header=None, usecols=[0,9])
# yesterday.columns = ['timestamp','SkyT','GroundT']
yesterday.columns = ['timestamp','Mag']
# tomorrow=pd.read_csv(cloud_logs[2])

try: #Polaris data for next day might not be available
    tomorrow = pd.read_csv(cloud_logs[2], header=None, usecols=[0,9])
    # tomorrow.columns = ['timestamp','SkyT','GroundT']
    tomorrow.columns = ['timestamp','Mag']
    data = pd.concat([yesterday,today,tomorrow],axis=0)
except:
    data = pd.concat([yesterday,today],axis=0)


t = data['timestamp']
# y = data['SkyT']-data['GroundT']
y=data['Mag']+14.55


samplerate = 6
f = interpolate.interp1d(t,y)
xnew = np.arange(min(data['timestamp']),max(data['timestamp']),samplerate)
x=[]
for i in range(len(xnew)):
    
    x.append(dt.datetime.utcfromtimestamp(xnew[i]).strftime('%Y-%m-%d %H:%M:%S'))

ynew = f(xnew)


Tformat='%Y-%m-%d %H:%M:%S'
for i in range(len(x)):
    
    if dt.datetime.strptime(str(sunset).replace('T', ' ')[:-4], Tformat)<dt.datetime.strptime(x[i], Tformat):
        start_idx=i
        break
    
for i in range(len(x)):
    if dt.datetime.strptime(str(sunrise).replace('T', ' ')[:-4], Tformat)<dt.datetime.strptime(x[i], Tformat):
        end_idx=i
        break

x=x[start_idx:end_idx]
ynew=ynew[start_idx:end_idx]

n = int(len(ynew))

a = ynew[0:(n)].reshape(1,1,n)
block = np.mean(a, axis=0)
block=pd.DataFrame(block,columns=x,index=['transparency'])
# ax2=inset_axes(ax1, width="100%", height="100%",loc=3, bbox_to_anchor=(-0.016,-0.27,1,1), bbox_transform=ax1.transAxes)

ax=inset_axes(ax1, width="100%", height="100%",loc=3, bbox_to_anchor=(-0.014,-0.06,1,1), bbox_transform=ax1.transAxes)
# ax2=sns.heatmap(block,cmap='Blues_r',vmax=-10,vmin=-20,cbar=False,zorder=1)

ax=sns.heatmap(block,cmap='Blues_r',vmin=0,vmax=5,cbar=False,zorder=1)

ax.axes.invert_yaxis()

ax2=plt.twinx()
sns.lineplot(x=x,y=ynew,color='k',ax=ax2)
ax2.yaxis.set_ticks(np.arange(0, 4, 1))
ax.set_yticklabels([])
ax.set_xticklabels([])
ax2.set_xticklabels([])
ax2.set_xticks([])
# ax2.patch.set_alpha(0.01)



#%% LOG EVENTS PLOT

c=0#counter to loop through each telescope and colormap
markers={}#markers on the plot
for logpath in ACP_logpaths:
    #try reading ACP log
    try:
        file=[file for file in logpath.iterdir() if obs_date.replace('-','') in file.name][0]
    except IndexError:
        c+=1
        continue
        
    
    log=ReadLog(file)
    #list of events in log that are worth noting, this list can be expanded 
    pattern=['Weather unsafe!','Dome closed!','Field index:']
    event_list=ReadLogLine(log, pattern, Break=False)
    
    
    
    names=event_list[0]

    for i in range(len(names)):
        if pattern[0] in names[i]:
            names[i]='bad weather'
            markers[names[i]]="x"
        if pattern[1] in names[i]:
            names[i]='dome close'
            markers[names[i]]="D"
        if pattern[2] in names[i]:
            names[i]=names[i].split('INFO: ')[1]
            field_num=names[i].split(": ")[1].strip("\n")
            markers[names[i]]=fr"${field_num}$"
            markers[names[i]]=markers[names[i]]



    dates=event_list[1]

    dates = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in dates]
    
    # levels = np.tile(np.random.choice([-2, -5, -1, -3, -4], size=(1, 5)),
    #                  int(np.ceil(len(dates)/6)))[:len(dates)]
#    length=[[-1,-2,-3,-4,-5],[-5,-3,-4,-2,-1],[-2,-4,-2,-5,-3]]
    length=np.random.randint(-5,-1,len(names))
    levels = np.tile(length,
                     int(np.ceil(len(dates)/6)))[:len(dates)]

    
    # Create figure and plot a stem plot with the date
    # fig, ax = plt.subplots(figsize=(8.8, 4), constrained_layout=True)
    # ax4.set(title="Observatory events")

#    ax3.vlines(dates, 0, levels, color=color[c])  # The vertical stems.
#    ax3.plot(dates, np.zeros_like(dates), "-o",
#            color="k", markerfacecolor="w")  # Baseline and markers on it.
    d=0
    
    for name in names:
        m = markers.get(name)
        
        ax3.plot(dates[d], 0+c*2,
                color=color[c], marker=m, markersize=10,markerfacecolor='k',markeredgecolor='k')  # Baseline and markers on it.
        ax3.axhline(y = 0+c*2, color = color[c], linestyle = '-')
        d+=1


    # annotate lines
#    for d, l, r in zip(dates, levels, names):
#        ax3.annotate(r, xy=(d, l),
#                    xytext=(np.random.randint(-20,20), np.sign(l)*3),  color=color[c], textcoords="offset points",
#                    horizontalalignment="right",
#                    verticalalignment="bottom")

    # format xaxis with 4 month intervals
    ax3.plot([],[],label="X - bad weather \n ♦ - dome close")

    ax3.set_xlim([mdates.datestr2num(str(sunset)), mdates.datestr2num(str(sunrise))])
    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # plt.setp(ax1.get_xticklabels(), rotation=30, ha="right")
    
    # remove y axis and spines
    ax3.yaxis.set_visible(False)
    ax3.spines[:].set_visible(False)
    ax3.xaxis.tick_top()
    # ax3.tick_params(axis="x",direction="in", pad=-15, labelsize=8)
    ax3.set_xticks([])
    ax3.margins(y=0.1)
    c+=1
field_info=getStarHour('D:/',str(obs_date))
    #field_info=str(field_info)+'\n'+'$*$ - star collection '+'$x$ - bad weather '+'$D$ - dome close'
text=''
for info in field_info:
    text+=info
ax3.text(1, -0.5, text, ha='right',  fontsize=14,
    verticalalignment='bottom',transform=ax3.transAxes)
    
# fig3.savefig('Events.svg',dpi=800,bbox_inches='tight')

#%% Combine plot 1 2 3 into one

fig123.subplots_adjust(hspace=0)
plt.title(str(obs_date)+' UT'+'\n'+str(sunset)+' - '+str(sunrise),pad=20)
plt.legend()
#plt.show(fig123)
fig123.savefig(operations_savepath.joinpath("event.svg"),dpi=800,bbox_inches='tight')
plt.close()

#%% PLOT SNR VS GMAG

fig4, ax4=plt.subplots()
try:
    ax4.scatter(x=red_table['GMAG'],y=red_table['SNR'],color='r', s=3,alpha=0.4,label='Airmass: %.2f Time: %s' % (red_airmass, red_snrtime))
except:
    pass
try:
    ax4.scatter(x=blue_table['GMAG'],y=blue_table['SNR'],color='b',s=3, alpha=0.4,label='Airmass: %.2f Time: %s' % (blue_airmass, blue_snrtime))
except:
    pass
try:
    ax4.scatter(x=green_table['GMAG'],y=green_table['SNR'],color='g',s=3, alpha=0.4,label='Airmass: %.2f Time: %s' % (green_airmass, green_snrtime))
except:
    pass
ax4.set_ylim([0,15])
ax4.set_xlabel('Gmag')
ax4.set_ylabel('Temporal SNR')

#plt.title()#!!!
# snr_info='Airmass: %.2f Time: %s \n' % (red_airmass, red_snrtime)+'Airmass: %.2f Time: %s /n' % (blue_airmass, blue_snrtime)+'Airmass: %.2f Time: %s' % (green_airmass, green_snrtime)
# ax4.text(1, -2, field_info, ha='right',  fontsize=14,
#     verticalalignment='bottom',transform=ax3.transAxes)
plt.legend()
plt.grid()
# plt.show(fig4)

fig4.savefig(operations_savepath.joinpath('SNR.svg'),dpi=800,bbox_inches='tight')
plt.close()

#%% PLOT STATISTICS FOR SINGLE DETECTIONS FOR 3 TELESCOPES

matched_dir=base_path.joinpath('/ColibriArchive',str(obs_date),'matched')

green_dets=base_path.joinpath('/ColibriArchive',str(obs_date))
red_dets=Path('/','Y:/',str(obs_date))
blue_dets=Path('/','Z:/',str(obs_date))
try:
    green_det_list=[f for f in green_dets.iterdir() if 'det' in f.name]
except FileNotFoundError:
    print('no Green data')
    green_det_list=[]
try:
    red_det_list=[f for f in red_dets.iterdir() if 'det' in f.name]
except FileNotFoundError:
    print('no Red data')
    red_det_list=[]
try:
    blue_det_list=[f for f in blue_dets.iterdir() if 'det' in f.name]
except FileNotFoundError:
    print('no Blue data')
    blue_det_list=[]

detections=green_det_list+red_det_list+blue_det_list

sigmas=[]
for file in detections:
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
plt.title('todays number of detections: '+str(np.size(sigmas)))
# plt.title(' Number of occulatations: '+str(np.size(sigmas)))
plt.grid(which='both',axis='x')


operations_savepath=base_path.joinpath('/Logs','Operations')
plt.savefig(operations_savepath.joinpath("sigma_det_today.svg"),dpi=800) 


plt.close()  

#%% PLOT OCCULTATION CANDIDATES ON 2 AND 3 TELSCOPES
try:
    matched_dirs=[d for d in matched_dir.iterdir() if d.is_dir()] 
except:
    print("no matches!")
    matched_dirs=[]
for match in matched_dirs:
    detected_files=[f for f in match.iterdir() if ('det' in f.name and '.txt' in f.name)]
    if len(detected_files)==2:

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
        plt.title('number of occultations for 2 telescopes: '+str(np.size(sigmas)))
        # plt.title(' Number of occulatations: '+str(np.size(sigmas)))
        plt.grid(which='both',axis='x')
        # plt.savefig(cumm_dets.joinpath("sigma.png"),dpi=300)
        # notbase_path=Path('/','D:','/Colibri','Green')
        operations_savepath=base_path.joinpath('/Logs','Operations')
        plt.savefig(operations_savepath.joinpath("sigma_occ2_today.svg"),dpi=800) 
        # plt.show()
        
        plt.close()
    elif len(detected_files)==3:
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
        plt.title(' number of occultations for 3 telescopes: '+str(np.size(sigmas)))
        # plt.title(' Number of occulatations: '+str(np.size(sigmas)))
        plt.grid(which='both',axis='x')
        # plt.savefig(cumm_dets.joinpath("sigma.png"),dpi=300)
        # notbase_path=Path('/','D:','/Colibri','Green')
        operations_savepath=base_path.joinpath('/Logs','Operations')
        plt.savefig(operations_savepath.joinpath("sigma_occ3_today.svg"),dpi=800) 
        # plt.show()
        
        plt.close()


#%% Edit HTML file with new events

matched_dir=green_sens.parent.parent #!!!

dirs=[d for d in matched_dir.iterdir() if d.is_dir()]

for d in dirs:
    try:
        event_svg=[f for f in d.iterdir() if '.svg' in f.name ][0]
        shutil.copy2(event_svg,operations_savepath) #copy any candidate plots of the 3rd pipeline
    except IndexError:
#        print("no occultations!")
        continue
    
from bs4 import BeautifulSoup
with open(operations_savepath.joinpath('observation_results.html')) as inf:
    txt = inf.read()
soup=BeautifulSoup(txt,'html.parser')
    
event_svg=[f for f in operations_savepath.iterdir() if '_star_' in f.name ]

prev_svg=soup.find_all('img', class_="occult")
for i in range(len(prev_svg)):
    prev_svg[i].decompose()
    
if len(event_svg)!=0:

    for event in event_svg:
        new_tag = soup.new_tag('img', src=str(event),attrs={'class':'occult'}) #add candidate plots to the html
        
        # soup.find('img', class_='SNR').append(new_tag)
        soup.center.append(new_tag)
    
    
new_br=soup.new_tag('br')
images=soup.find_all('img')


for i in range(len(images)):
    images[i].append(new_br)



html = soup.prettify("utf-8") #save modified html
with open(operations_savepath.joinpath('observation_results.html'), "wb") as file: 
    file.write(html)
    
    

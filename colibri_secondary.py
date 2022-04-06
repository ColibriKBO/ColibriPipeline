"""
Created March 8, 2022 by Rachel Brown

Update: March 10, 2022 by Rachel Brown

-secondary Colibri pipeline for matching identified events to a kernel
"""


import numpy as np
from astropy.io import fits
from astropy.convolution import convolve_fft, RickerWavelet1DKernel
from astropy.time import Time
from joblib import delayed, Parallel
from copy import deepcopy
import multiprocessing
import time
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import datetime
import lightcurve_looker

def getKernelParams(kernel_i):
    '''get parameters for best fitting kernel from the output .txt file'''
    
    param_filename = base_path.joinpath('params_kernels_031522.txt')
    kernel_params = pd.read_csv(param_filename, delim_whitespace = True)
    
    return kernel_params.iloc[kernel_i]


def plotKernel(lightcurve, fluxTimes, kernel, start_i, eventFrame, starNum, directory, params, fit_line):
    '''make plot of the convolution of lightcurve and best fitting kernel'''
    
    '''prepare kernel list'''
    #extend the kernel to fill the whole 2s segment
    extended_kernel = [1]*len(lightcurve)
    frame_nums = list(range(0, len(lightcurve)))
    
    #insert kernel values at the correct indices
    for i in range(len(kernel)):
        new_i = start_i + i
        extended_kernel[new_i] = kernel[i]
          
    #get time and date of the segment start for file naming
    eventDate = pathlib.Path(starData['filename'][0]).parent.name
    
    
    #get list of residuals from the kernel and the data
    residuals = lightcurve - extended_kernel
    
    #strip whitepace at beginning of fit line string
    slope = fit_line.c[0]
    intercept = fit_line.c[1] 
    fit_line = str(fit_line).strip()
    std = np.std(lightcurve)
    
    #fix time to account for minute rollover
    seconds = []    #list of times since first frame
    t0 = fluxTimes[0]    #time of first frame
        
    for t in fluxTimes:
    
        if t < t0:          #check if time has gone back to 0
            t = t + 60.
            
        if t != t0:         #check if minute has rolled over
            if t - t0 < seconds[-1]:
                t = t + 60.
            
        seconds.append(t - t0)
    
    fluxTimes = seconds
    
    flux_min = min(lightcurve) - 0.2
    flux_max = max(lightcurve) + 0.1
    flux_height = flux_max - flux_min
    
    residuals_min = min(residuals) - 0.1
    residuals_max = max(residuals) + 0.1
    residuals_height = residuals_max - residuals_min

    '''make plot'''
    fig, (ax1, ax3) = plt.subplots(2, figsize = (10, 10), gridspec_kw={'height_ratios': [flux_height, residuals_height]})
    
    #Linear fit equation
    textstr1 = '\n'.join((
    'Linear fit to data:',
    '%s' % str(fit_line)))
    
    #best matching kernel params
    textstr2 = '\n'.join((
    'Kernel params:',
    ' ',
    'Object R [m] = %.0f' % (params[2], ),
    'Star d [mas] = %.2f' % (params[3], ),
    'b [m] = %.0f' % (params[4], ),
    'Shift [frames] = %.2f' % (params[5], )))
    
    #box to display data
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    #plot lightcurve as a function of time
    ax1.plot(fluxTimes, lightcurve)

    #place linear fit equation in box
    ax1.text(1.02, 0.95, textstr1, transform=ax1.transAxes, fontsize=15,
    verticalalignment='top', bbox=props)
    
    #place kernel params in box
    ax1.text(1.02, 0.70, textstr2, transform=ax1.transAxes, fontsize=15,
    verticalalignment='top', bbox=props)
    
    #lines for the mean (1.0), and standard deviation
    ax1.hlines(1.0, min(fluxTimes), max(fluxTimes), color = 'black')
    ax1.hlines(1.0 + std, min(fluxTimes), max(fluxTimes), linestyle = '--', color = 'black', label = 'stddev: %.3f' % std)
    ax1.hlines(1.0 - std, min(fluxTimes), max(fluxTimes), linestyle = '--', color = 'black')
    
    ax1.tick_params(direction = 'inout', length = 10, right = True, labelsize = 12)
    
    ax1.set_ylim(flux_min, flux_max)
    
    #plot best matched kernel as a function of frame number
    ax2 = ax1.twiny()
    ax2.plot(frame_nums, extended_kernel, label = 'Best fit kernel', color = 'tab:orange')
    
    #plot event location
    ax2.vlines(eventFrame, min(lightcurve), max(lightcurve), color = 'red', label = 'Event middle')
    
    ax2.tick_params(direction = 'inout', length = 10, labelsize = 12)

    #titles and labels for main plot
    ax1.set_title('Normalized light curve matched with kernel ' + str(int(params[0])), fontsize = 18)
    ax1.set_xlabel('Time (s)', fontsize = 15)
    ax2.set_xlabel('Frame number', fontsize = 15)
    ax1.set_ylabel('Flux normalized by linear fit', fontsize = 15)
    
    #plot residuals on secondary panel
    ax3.scatter(frame_nums, residuals, s = 8)
    ax3.set_ylabel('Residuals', fontsize = 15)
    ax3.hlines(0, min(frame_nums), max(frame_nums), color = 'gray', linestyle = '--')
    ax3.set_xlabel('Frame Number', fontsize = 15)
    ax3.tick_params(direction = 'inout', length = 10, labelsize = 12)
    
    ax3.set_ylim(residuals_min, residuals_max)
    
    #add times to upper y axis
    ax4 = ax3.twiny()
    ax4.scatter(fluxTimes, residuals, s = 0)
    ax4.tick_params(direction = 'inout', length = 10, labelsize = 12)
    ax4.set_xticklabels([])
    
    #place legend 
    fig.legend(loc = 'right', fontsize = 15)
    
    plt.show()
    #plt.savefig(directory.joinpath('star' + starNum + '_' + eventDate + '_matched.png'), bbox_inches = 'tight')
    plt.close()
    
    return
    


def diffMatch(template, data, sigmaP):
    """ Calculates the best start position (minX) and the minimization constant associated with this match (minChi) """

    minChi = np.inf       #statistical minimum variable
    minX = np.inf         #starting index variable
    
    matchTolerance = 3    #number of frames difference to tolerate for match (~150 ms)

    #loop through possible starting values for the best match dip
    for dipStart in range((int(len(data) / 2) - int(len(template) / 2)) - matchTolerance, int(len(data) / 2) - int(len(template) / 2) + matchTolerance):
        
        chiSquare = 0
        
        #loop through each flux value in template and compare to data
        for val in range(0, len(template)):
            chiSquare += (abs(template[val] - data[dipStart + val])) / abs(sigmaP[dipStart + val])  #chi^2 expression
            
        #if the curren chi^2 value is smaller than previous, set new stat. minimum and get new index
        if chiSquare < minChi:
            minChi = chiSquare
            minX = dipStart
            
    return minChi, minX


def kernelDetection(fluxProfile, fluxTimes, dipFrame, kernels, num, directory):
    """ Detects dimming using Mexican Hat kernel for dip detection and set of Fresnel kernels for kernel matching """

    med = np.median(fluxProfile)         

    #normalized sigma corresponding to RHS of Eq. 2 in Pass et al. 2018
    #sigmaNorm = np.std(trunc_profile[(FramesperMin-NumBkgndElements):FramesperMin]) / np.median(trunc_profile[(FramesperMin-NumBkgndElements):FramesperMin])
    #really not sure how this matches what is in Emily's paper.....
    sigmaNorm = np.std(fluxProfile) / np.median(fluxProfile)
    

    """ Dip detection"""
    Gain = 16.5     #For high gain images: RAB - 031722
    #Gain = 2.8     #For low gain images: RAB - 031722
      
        #normalized fractional uncertainty
    sigmaP = ((np.sqrt(np.abs(fluxProfile) / np.median(fluxProfile) / Gain)) * Gain)   #Eq 7 in Pass 2018 (Poisson)
    sigmaP = np.where(sigmaP == 0, 0.01, sigmaP)
    

    
    '''get curve with dip removed (called 'background') for normalization'''
    '''RAB 032321'''
    half_dip_width = 10     #approximate length of a dip event [frames]
    
    #set the beginning of the dip region
    if dipFrame - half_dip_width > 0:    
        dip_start = dipFrame - half_dip_width
    
    #if beginning should be the beginning of the light curve
    else:
        dip_start = 0
        
    #set the end of the dip zone
    if dipFrame + half_dip_width < len(fluxProfile):
        dip_end = dipFrame + half_dip_width
    
    #if end should be the end of the light curve
    else:
        dip_end = len(fluxProfile) - 1
    
    #copy the original flux profile and list of times
    background = fluxProfile[:]
    background_time = fluxTimes[:]
    
    #remove the dip region from the background flux and time arrays
    del background[dip_start:dip_end]
    del background_time[dip_start:dip_end]
    
    #fit line to background portion
    bkgd_fitLine = np.poly1d(np.polyfit(background_time, background, 1))
    
    #divide original flux profile by fitted line
    fluxProfile =  fluxProfile/bkgd_fitLine(fluxTimes)
    
    """ Kernel matching"""

    StatMin = np.inf    #statistical minimum used for finding best matching kernel
        
    #loop through each kernel in set, check to see if it's a better fit
    for ind in range(0, len(kernels)):
            

            
        #get a statistical minimum and location of the starting point for the kernel match
        new_StatMin, loc = diffMatch(kernels[ind], fluxProfile, sigmaP)
            
        #checks if current kernel is a better match
        if new_StatMin < StatMin:
            active_kernel = ind
            MatchStart = loc    #index of best starting position for kernel matching
            StatMin = new_StatMin
                
    #list of Poisson uncertainty values for the event
    eventSigmaP = sigmaP[MatchStart:MatchStart + len(kernels[active_kernel])]   
    thresh = 0   #starting for sum in LHS of Eq. 2 in Pass 2018
    
    #unnecessary if we're sure kernels span 20-40% dip RAB Mar 15 2022
    for P in eventSigmaP:
        thresh += (abs(sigmaNorm)) / (abs(P))  # kernel match threshold - LHS of Eq. 2 in Pass 2018
            
    #check if dip is significant to call a candidate event
    if StatMin < thresh:      #Eq. 2 in Pass 2018
            
      #  critFrame = np.where(fluxProfile == fluxProfile[dipFrame])[0]    #time of event
     #   critFrame = dipFrame    
     #   if len(critFrame) > 1:
     #       raise ValueError
        
        params = getKernelParams(active_kernel)
        
        plotKernel(fluxProfile, fluxTimes, kernels[active_kernel], MatchStart, dipFrame, num, directory, params, bkgd_fitLine)
        
        return active_kernel, StatMin, MatchStart, params  # returns location in original time series where dip occurs
        
    else:
        print('Event in star %s did not pass threshold' % num)
        
        params = getKernelParams(active_kernel)
        
        plotKernel(fluxProfile, fluxTimes, kernels[active_kernel], MatchStart, dipFrame, num, directory, params, bkgd_fitLine)
        
        return -1  # reject events that do not pass kernel matching
    
    

def readFile(filepath):
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
            elif i == 6:
                event_time = line.split('T')[2].split('\n')[0]
                
            elif i == 9:
                event_type = line.split(':')[1].split('\n')[0].strip(" ")
                
            elif i == 10:
                star_med = line.split(':')[1].split('\n')[0].strip(" ")
                
            elif i == 11:
                star_std = line.split(':')[1].split('\n')[0].strip(' ')
                
        #reset event frame to match index of the file
        event_frame = event_frame - first_frame

    return starData, event_frame, star_x, star_y, event_time, event_type, star_med, star_std
    
'''-----------code starts here -----------------------'''

runPar = False          #True if you want to run directories in parallel
telescope = 'Red'       #identifier for telescope
gain = 'high'           #gain level for .rcd files ('low' or 'high')
obs_date = datetime.date(2021, 8, 4)    #date observations 
process_date = datetime.date(2022, 4, 6)
base_path = pathlib.Path('/', 'home', 'rbrown', 'Documents', 'Colibri')  #path to main directory


if __name__ == '__main__':
    
    ''' prepare RickerWavelet/Mexican hat kernel to convolve with light curves'''
    
    exposure_time = 0.025    # exposure length in seconds
    expected_length = 0.15   # TODO: come back to this - related to the characteristic scale length, length of signal to boost in convolution, may need tweaking/optimizing

    kernel_frames = int(round(expected_length / exposure_time))   # width of kernel
    ricker_kernel = RickerWavelet1DKernel(kernel_frames)          # generate kernel

    kernel_set = np.loadtxt(base_path.joinpath('kernelsMar2022.txt'))
    
    #check if kernel has a detectable dip - moved out of dipdetection function RAB 031722
    noise = 0.8   #minimum kernel depth threshold RAB Mar 15 2022- detector noise levels (??) TODO: change - will depend on high/low
    for i in range(len(kernel_set)):
        
        #check if deepest dip in a kernel is greater than expected background noise 
        if min(kernel_set[i]) > noise:
            #remove kernel with dips less than expected background noise
            print('Removing kernel ', i)
            del(kernel_set[i])
            continue
    
   # refresh_rate = 2

    #create RickerWavelet/Mexican Hat kernel to convolve with light profil
   # kernel_frames = int(round(expected_length / exposure_time)) #width of kernel
   # ricker_kernel = RickerWavelet1DKernel(kernel_frames)       #generate kernel
    
    #what is this?
    #evolution_frames = int(round(refresh_rate / exposure_time))  # determines the number of frames in X seconds of data


    '''get filepaths to results directory'''
    
    #directory containing detection .txt files
    archive_dir = base_path.joinpath('ColibriArchive', telescope, str(process_date))
    
    #list of filepaths to .txt detection files
    detect_files = [f for f in archive_dir.iterdir() if 'det' in f.name]
    
    
    '''loop through each file'''
    
    diff_results = []
    geo_results = []  
    
    for filepath in detect_files:
        
        #number id of occulted star
        star_num = filepath.name.split('star')[1].split('_')[0]
        
        #read in file data
        starData, event_frame, star_x, star_y, event_time, event_type = readFile(filepath)
        
       # lightcurve_looker.plot_event(archive_dir, starData, event_frame, star_num, [star_x, star_y], event_type)
        
        if event_type == 'diffraction':
            diff_results.append((star_num, star_x, star_y, event_time, kernelDetection(list(starData['flux']), list(starData['time']), event_frame, kernel_set, star_num, archive_dir)))
       
        if event_type == 'geometric':
            geo_results.append((star_num, star_x, star_y, event_time, kernelDetection(list(starData['flux']), list(starData['time']), event_frame, kernel_set, star_num, archive_dir)))

    #save list of best matched kernels in a .txt file
        
    diff_save_file = archive_dir.joinpath(str(obs_date) + '_' + telescope + '_diffraction_kernelMatches.txt')
    geo_save_file = archive_dir.joinpath(str(obs_date) + '_' + telescope + '_geometric_kernelMatches.txt')

    with open(diff_save_file, 'w') as file:
        
        file.write('#starNumber  starX  starY  eventTime  kernelIndex  Chi2  startLocation  ObjectD   StellarD   b    shift\n')
            
        for line in diff_results:
            
            #events that didn't pass kernel matching
            if line[4] == -1:
                continue
            
            file.write('%s %f %f %s %i %f %i %f %f %f %f\n' %(line[0], line[1], line[2], line[3], 
                                                  line[4][0], line[4][1], line[4][2], 
                                                  line[4][3][2], line[4][3][3], line[4][3][4], line[4][3][5]))
    
    with open(geo_save_file, 'w') as file:
        
        file.write('#starNumber  starX  starY  eventTime  kernelIndex  Chi2  startLocation  ObjectD   StellarD   b    shift\n')
        
        for line in geo_results:
            
            #events that didn't pass kernel matching
            if line[4] == -1:
                continue
            
            file.write('%s %f %f %s %i %f %i %f %f %f %f\n' %(line[0], line[1], line[2], line[3], 
                                                  line[4][0], line[4][1], line[4][2], 
                                                  line[4][3][2], line[4][3][3], line[4][3][4], line[4][3][5]))





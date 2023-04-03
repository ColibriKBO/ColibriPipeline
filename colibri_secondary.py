"""
Created March 8, 2022 by Rachel Brown

Update: March 10, 2022 by Rachel Brown

Update: 2022-09-21 Roman A. : modified output graph and deleted astnet solution calculation, also now this code works on matched files

-secondary Colibri pipeline for matching identified events to a kernel
"""


import numpy as np
from astropy.io import fits
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import datetime
import VizieR_query
from astropy import wcs
import getRAdec
import astrometrynet_funcs
import sys, os, time
import argparse

def readRAdec(filepath):                                    #modified readFile from Colibri Pipeline
    '''read in a .txt detection file and get information from it'''
    
    #make dataframe containing image name, time, and star flux value for the star
    # starData = pd.read_csv(filepath, delim_whitespace = True, 
    #        names = ['filename', 'time', 'flux'], comment = '#')

    
    
    #get header info from file
    with filepath.open() as f:
        
        #loop through each line of the file
        for i, line in enumerate(f):
          
            #get star coords
            
                
            if i==6:
                star_coords = line.split(':')[1].split(' ')[1:3]
                star_ra = float(star_coords[0])
                star_dec = float(star_coords[1])
                
        

    #return    
    return (star_ra,star_dec)

def match_RADec(RA, DEC, gdata, SR):
    '''matches list of found stars with Gaia catalog by RA/dec to get magnitudes
    input: pandas Dataframe of detected star data {x, y, RA, dec}, dataframe of Gaia data {ra, dec, magnitudes....}, search radius [deg]
    returns: original star data frame with gaia magnitude columns added where a match was found'''
    
    #from Mike Mazur's 'MagLimitChecker.py' script--------------------
    data = pd.DataFrame()
           
    RA_SR = SR/np.cos(np.radians(DEC))     #adjustment for RA

    df = gdata[(gdata['RA_ICRS'] <= (RA+RA_SR))]      #make new data frame with rows from Gaia df that are within upper RA limit
    df = df[(df['RA_ICRS']  >= (RA-RA_SR))]            #only include rows from this new df that are within lower RA limit
    df = df[(df['DE_ICRS'] <= (DEC+SR))]          #only include rows from this new df that are within upper dec limit
    df = df[(df['DE_ICRS'] >= (DEC-SR))]          #only include rows from this new df that are withing lower dec limit
         
    #RAB - matching based on smallest distance instead of brightest magnitude
    #df['diff'] = np.sqrt(((df.RA_ICRS - RA)**2*np.cos(np.radians(df.DE_ICRS))) + (df.DE_ICRS - DEC)**2)
    #df.sort_values(by=['diff'], ascending=True, inplace = True)
       
    #df.sort_values(by=["phot_g_mean_mag"], ascending=True, inplace=True)       #sort matches by brightness (brightest at tope)
    df.sort_values(by=["Gmag"], ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    #if matches are found, add corresponsing magnitude columns from brightest Gaia match to original star dataframe
    if len(df.index)>=1:   #RAB - added >= sign
        data.loc[0, 'Colibri_RA'] = RA
        data.loc[0, 'Colibri_dec'] = DEC
        data.loc[0,'GMAG'] = df.loc[0]['Gmag']
        data.loc[0,'Gaia_RA'] = df.loc[0]['RA_ICRS']
        data.loc[0,'Gaia_dec'] = df.loc[0]['DE_ICRS']
        data.loc[0, 'Gaia_B-R'] = df.loc[0]['BP-RP']
    else:
       # print('No match found - Star ', star_num)
        data.loc[0, 'Colibri_RA'] = RA
        data.loc[0, 'Colibri_dec'] = DEC
        data.loc[0,'GMAG'] = float('NaN')
        data.loc[0,'Gaia_RA'] = float('NaN')
        data.loc[0,'Gaia_dec'] = float('NaN')
        data.loc[0, 'Gaia_B-R'] = float('NaN')
        
       #end of Mike's section -------------------------------------------

    
    return data

def getKernelParams(kernel_i):
    '''get parameters for best fitting kernel from the output .txt file'''
    
    param_filename = base_path.parent.joinpath('kernels', 'params_kernels.txt')
    kernel_params = pd.read_csv(param_filename, delim_whitespace = True)
    
    return kernel_params.iloc[kernel_i]


def plotKernel(lightcurve, eventData, kernel, start_i, starNum, directory, params, fit_line, minChi, star_df,scope,filename):
    '''make plot of the convolution of lightcurve and best fitting kernel'''
    
    fluxTimes = list(eventData[0]['time'])
    eventFrame = eventData[1]
    
    '''prepare kernel list'''
    #extend the kernel to fill the whole 2s segment
    extended_kernel = [1]*len(lightcurve)
    frame_nums = list(range(0, len(lightcurve)))
    
    #insert kernel values at the correct indices
    for i in range(len(kernel)):
        new_i = start_i + i
        extended_kernel[new_i] = kernel[i]
          
    #get time and date of the segment start for file naming
    eventDate = pathlib.Path(eventData[0]['filename'][0]).parent.name
    
    
    #get list of residuals from the kernel and the data
    residuals = lightcurve - extended_kernel
    
    #strip whitepace at beginning of fit line string
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
                
        #check if there are large time gaps in between frames
            gap_threshold = 0.3     
        
            if t - t0 - seconds[-1] > gap_threshold:
          #      print('Time jump: ', starNum)
           #     print('gap: ', t - t0 - seconds[-1])
                
                return -1
            
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
    '%s' % str(fit_line),
    'Event significance:',
    eventData[5],
    'Star SNR = %.2f' % (float(eventData[6])/float(eventData[7]))))
    
    #best matching kernel params
    textstr2 = '\n'.join((
    'Model params:',
    'Object R [m] = %.0f' % (params[2], ),
    'Star D [mas] = %.3f' % (params[3], ),
    'Impact param [m] = %.0f' % (params[4], ),
    'Shift [frames] = %.2f' % (params[5], ),
    'Chi Square: %.3f' %(minChi)))
    
    textstr3 = '\n'.join((
        'Star Coords (X, Y):',
        '(%.1f, %.1f)' %(eventData[2], eventData[3]),
        'Gaia star (RA, Dec):',
        '(%.4f, %.4f)' %(star_df['Gaia_RA'], star_df['Gaia_dec']),
        'g-band mag: %.2f' % (star_df['GMAG']),
        'B-R Colour: %.2f' % (star_df['Gaia_B-R'])))
    
    textstr4 = '\n'.join((
        'Date: ',
        '%s' %(str(obs_date)),
        'Event Time: ',
        '%s' %(eventData[4][:12])))
    
    #box to display data
    props1 = dict(boxstyle='round', facecolor='darkorange', alpha=0.5)
    props2 = dict(boxstyle='round', facecolor='cornflowerblue', alpha=0.5)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    #plot lightcurve as a function of time
    ax1.plot(fluxTimes, lightcurve)

    #place linear fit equation in box
    ax1.text(1.03, 0.65, textstr1, transform=ax1.transAxes, fontsize=15,
    verticalalignment='top', bbox=props2)
    
    #place kernel params in box
    ax1.text(1.03, 0.25, textstr2, transform=ax1.transAxes, fontsize=15,
    verticalalignment='top', bbox=props1)
    
    #place star properties in box
    ax1.text(1.03, -0.25, textstr3, transform=ax1.transAxes, fontsize=15,
             verticalalignment='top', bbox=props)
    
    #place time and date info in box
    ax1.text(1.03, -0.70, textstr4, transform=ax1.transAxes, fontsize=15,
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
   # ax1.set_title('Occultation event candidate matched with TNO occultation model', fontsize = 18)
    ax1.set_title(scope+' Star ' + starNum + ' normalized light curve matched with kernel ' + str(int(params[0])), fontsize = 18)
    ax1.set_xlabel('Time (s)', fontsize = 15)
    ax2.set_xlabel('Frame number', fontsize = 15)
    ax1.set_ylabel('Flux normalized by linear fit (counts)', fontsize = 15)
    
    #plot residuals on secondary panel
    ax3.scatter(frame_nums, residuals, s = 8)
    ax3.set_ylabel('Residuals from fit', fontsize = 15)
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
    fig.legend(loc = 'center right', fontsize = 15, facecolor = 'wheat', framealpha = 0.5, bbox_to_anchor = (1.15, 0.83), )
    #fig.legend()
    
    #plt.show()
#    plt.savefig(directory.joinpath('star' + starNum + '_' + eventDate + '_matched.png'), bbox_inches = 'tight')#, dpi=1000)
    plt.savefig(directory.joinpath(filename + '.png'), bbox_inches = 'tight')#, dpi=1000)
    plt.close()
    
    return 0
    


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


def kernelMatch(eventData, kernels, starNum, directory, starData_df,scope,filename):
    """ Matches dip event with simulated occultation pattern, checks for significance
    input: tuple of event data (including flux, time, and frame number), set of kernels [array], 
    star number, directory to save results to, dataframe with star's coordinates and Gaia data,
    scope for telescope name in the event graph, filename for the output graph name
    returns: index of best matched kernel, minimum Chi2 value, location of kernel match starting frame,
    parameters of best matched kernel, or -1 if no kernels matched with significance"""

    '''get detected event data and calculate normalized sigma'''
    fluxProfile = list(eventData[0]['flux'])        #list of star fluxes
    fluxTimes = list(eventData[0]['time'])          #list of frame times
    dipFrame = eventData[1]                         #frame where dip was detected
     
    
    #normalized sigma corresponding to LHS of Eq. 2 in Pass et al. 2018
    #sigmaNorm = np.std(trunc_profile[(FramesperMin-NumBkgndElements):FramesperMin]) / np.median(trunc_profile[(FramesperMin-NumBkgndElements):FramesperMin])
    #really not sure how this matches what is in Emily's paper.....
    sigmaNorm = np.std(fluxProfile) / np.median(fluxProfile)
    

    '''---- get Poisson noise for every value in light curve ----'''
    #TODO: use gain keyword instead of setting manually RAB - 070622
    Gain = 0.82    #For high gain images: RAB - 062922
    #Gain = 18.98     #For low gain images: RAB - 062922
      
    #array of normalized Poisson uncertainties
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
    
    '''------- Kernel matching --------'''

    StatMin = np.inf    #statistical minimum used for finding best matching kernel
        
    #loop through each kernel in set, check to see if it's a better fit
    for ind in range(0, len(kernels)):
            
        #get a statistical minimum and location of the starting point for the kernel match
        new_StatMin, loc = diffMatch(kernels[ind], fluxProfile, sigmaP)
            
        #checks if current kernel is a better match
        if new_StatMin < StatMin:
            active_kernel = ind         #best matching kernel so far
            MatchStart = loc            #index of best starting position for kernel matching
            StatMin = new_StatMin       #best matching statistical minimum
                
    #list of Poisson uncertainty values for the event
    
    #2022-12-20 Roman A. commented out unnecessary threchold check
    
    #eventSigmaP = sigmaP[MatchStart:MatchStart + len(kernels[active_kernel])]   
    #thresh = 0   #starting for sum in LHS of Eq. 2 in Pass 2018
    
    #unnecessary if we're sure kernels span 20-40% dip RAB Mar 15 2022
    #for P in eventSigmaP:
        #thresh += (abs(sigmaNorm)) / (abs(P))  # kernel match threshold - LHS of Eq. 2 in Pass 2018
    
    
    '''------------check event for significance ---------------------'''
    
    #check if dip is significant to call a candidate event
    #if thresh > StatMin:      #Eq. 2 in Pass 2018
            
      #  critFrame = np.where(fluxProfile == fluxProfile[dipFrame])[0]    #time of event
     #   critFrame = dipFrame    
     #   if len(critFrame) > 1:
     #       raise ValueError
        
        #get parameters (TNO size, stellar size, impact param, shift adjustment) for best match
    params = getKernelParams(active_kernel)
        
        #make plot with all the event information
    plot = plotKernel(fluxProfile, eventData, kernels[active_kernel], MatchStart, starNum, directory, params, bkgd_fitLine, StatMin, starData_df,scope,filename)

        #reject events with a large jump in time between exposures
    if plot < 0:
        return -1
        
        #return results
    return active_kernel, StatMin, MatchStart, params  # returns location in original time series where dip occurs
        
    #else:
        #print('Event in star %s did not pass threshold' % starNum)
        
        #params = getKernelParams(active_kernel)
        
        #plotKernel(fluxProfile, fluxTimes, kernels[active_kernel], MatchStart, dipFrame, num, directory, params, bkgd_fitLine)
        
        #return -1  # reject events that do not pass kernel matching
    
    

def readFile(filepath):
    '''read in a .txt detection file and get information from it'''
    
    #make dataframe containing image name, time, and star flux value for the star
    starData = pd.read_csv(filepath, delim_whitespace = True, 
           names = ['filename', 'time', 'flux','conv_flux'], comment = '#')

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
                print(event_time)
            
            elif i == 10:
                significance = line.split(':')[1].split('\n')[0].strip(" ")
                
            elif i == 11:
                star_std = line.split(':')[1].split('\n')[0].strip(' ')
                
            elif i == 12:
                star_med = line.split(':')[1].split('\n')[0].strip(" ")
                
            
                
        #reset event frame to match index of the file
        event_frame = event_frame - first_frame

    #return all event data from file as a tuple    
    return (starData, event_frame, star_x, star_y, event_time, significance, star_med, star_std)
    #return starData, event_frame, star_x, star_y, event_time, event_type, star_med, star_std
    
    
#def getTransform(date):
#    '''get astrometry.net transform for a given minute'''
#
#    #if transformation has already been calculated, get from dictionary
#    if date in transformations:
#        return transformations[date]
#    
#    #calculate new transformation from astrometry.net
#    else:
#        #get median combined image
#        median_image = [f for f in median_combos if date in f.name][0]
#        
#        #get name of transformation header file to save
#        transform_file = median_image.with_name(median_image.name.strip('_medstacked.fits') + '_wcs.fits')
#        
#        #check if the tranformation has already been calculated and saved
#        if transform_file.exists():
#            
#            #open file and get transformation
#            wcs_header = fits.open(transform_file)
#            transform = wcs.WCS(wcs_header[0])
#
#        #calculate new transformation
#        else:
#            #get WCS header from astrometry.net plate solution
#            wcs_header = astrometrynet_funcs.getSolution(median_image, transform_file, soln_order)
#        
#            #calculate coordinate transformation
#            transform = wcs.WCS(wcs_header)
#        
#        #add to dictionary
#        transformations[date] = transform
#        
#        return transform
    
    
'''-----------code starts here -----------------------'''

''' set up parameters for running the code '''
global telescope
telescope = os.environ['COMPUTERNAME']       #identifier for telescope
gain = 'high'           #gain level for .rcd files ('low' or 'high')
#soln_order = 4      #tweak order for astrometry.net solution
# obs_date = datetime.date(2021, 8, 4)    #date observations 
# process_date = datetime.date(2022, 4, 26)
# base_path = pathlib.Path('/', 'home', 'rbrown', 'Documents', 'Colibri', telescope)  #path to main directory

# '''get arguments - Added by MJM'''
# if len(sys.argv) > 1:
#     base_path = pathlib.Path(sys.argv[1])
#     obsYYYYMMDD = sys.argv[2]
#     obs_date = datetime.date(int(obsYYYYMMDD.split('/')[0]), int(obsYYYYMMDD.split('/')[1]), int(obsYYYYMMDD.split('/')[2]))
# else:
#     base_path = pathlib.Path('/', 'home', 'rbrown', 'Documents', 'Colibri', telescope)  #path to main directory
# obs_date = datetime.date(2021, 8, 4)    #date observations 

# procYMD = str(datetime.datetime.today().strftime('%Y/%m/%d'))
# procyear = int(datetime.datetime.today().strftime('%Y'))
# procmonth = int(datetime.datetime.today().strftime('%m'))
# procday = int(datetime.datetime.today().strftime('%d'))
# # process_date = datetime.date(procyear, procmonth, procday)
# process_date = obs_date
# print(process_date)

'''begin program'''
if __name__ == '__main__':

    ''' Argument parsing added by MJM - July 20, 2022 '''
    arg_parser = argparse.ArgumentParser(description=""" Run secondary Colibri processing
        Usage:

        """,
        formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('-b', '--basedir', help='Base directory for data (typically d:)', default='d:')
    arg_parser.add_argument('-d', '--date', help='Observation date (YYYY/MM/DD) of data to be processed.')
    # arg_parser.add_argument('-p', '--procdate', help='Processing date.', default=obs_date)
    cml_args = arg_parser.parse_args()
    obsYYYYMMDD = cml_args.date
    obs_date = datetime.date(int(obsYYYYMMDD.split('/')[0]), int(obsYYYYMMDD.split('/')[1]), int(obsYYYYMMDD.split('/')[2]))
    
    procYMD = str(datetime.datetime.today().strftime('%Y/%m/%d'))
    procyear = int(datetime.datetime.today().strftime('%Y'))
    procmonth = int(datetime.datetime.today().strftime('%m'))
    procday = int(datetime.datetime.today().strftime('%d'))
    # process_date = datetime.date(procyear, procmonth, procday)
    process_date = obs_date
    print(process_date)

    cml_args = arg_parser.parse_args()

    base_path = pathlib.Path(cml_args.basedir)
    #base_path=base_path.joinpath('/Colibri/Green')
    obsYYYYMMDD = cml_args.date
    ''' prepare occultation kernels set to match to light curves'''

    #load in pre-made kernel set
    # kernel_set = np.loadtxt(base_path.parent.joinpath('kernels', 'kernels_040622.txt')) # Commented out - MJM
    kernel_set = list(np.loadtxt(base_path.joinpath('kernels', 'kernels.txt')))
    
    #check if each kernel has a detectable dip - moved out of dipdetection function RAB 031722
    noise = 0.8   #minimum kernel depth threshold RAB Mar 15 2022- detector noise levels (??) TODO: change - will depend on high/low

    kernel_set=[k for k in kernel_set if min(k) > noise]#2023-03-04 Roman A. fixed cant delete array element

    '''get filepaths to required directories/files'''
    
    #directory containing detection .txt files
    archive_dir = base_path.joinpath('ColibriArchive', str(process_date))
    matched_dir=archive_dir.joinpath('matched')#09-21 Roman A. now this code runs on matched events
    # matched_dir=archive_dir#09-21 Roman A. now this code runs on matched events
    #list of filepaths to .txt detection files
    day_files = [f for f in matched_dir.iterdir() if str(process_date) in f.name]
    detect_files = [f for f in day_files if ('det' in f.name and 'png' not in f.name)]
    
    

    #list of median combined images
    median_combos = [f for f in archive_dir.iterdir() if 'medstacked' in f.name]
    
    matched_files = os.listdir(matched_dir)
#    transform_files = [f for f in archive_dir.iterdir() if 'new-image' in f.name]
    
    #dictionary to hold WCS transformations for each transform file
    transformations = {}
    
    diff_results = []   #list to hold results for diffraction events
    geo_results = []    #list to hold results for geometric events
    
        
    '''loop through each detection file'''
    starttime = time.time() # Added timer - MJM
    transformtime = 0 # Added timer - MJM
    gaiatime = 0 # Added timer - MJM
    occulttime = 0 # Added timer - MJM

    for filepath in detect_files:
        
        '''get event information from .txt file'''
        
        #number id of occulted star
        star_num = filepath.name.split('star')[1].split('_')[0]

        print(filepath)
        
        #read in file data as tuple containing (star lightcurve, event frame #, star x coord, star y coord, event time, event type, star med, star std)
        eventData = readFile(filepath)


        '''get transformation from image coordinates [X, Y] to ecliptic coords [RA, dec]'''
        #get corresponding WCS transformation
        # date = pathlib.Path(eventData[0]['filename'][0]).parent.name.split('_')[1]

        # tstart = time.time() # Added timer - MJM
        # transform = getTransform(date)
        
        #get star coords in RA/dec
        # star_wcs = getRAdec.getRAdecSingle(transform, (eventData[2], eventData[3]))
        
        star_RA, star_dec = readRAdec(filepath)#09-21 Roman A. now coords are in txt file
        
        
        # tend = time.time() # Added timer - MJM
        # transformtime += tend-tstart # Added timer - MJM
        
        '''Get Gaia catalog of stars in surrounding region'''
        
        #query Gaia for nearby stars
        gstart = time.time() # Added timer - MJM
        gaia_SR = 0.1         #search radius for query in degrees
        gaia = VizieR_query.makeQuery([star_RA, star_dec], gaia_SR)        #get dataframe of Gaia results {RA, dec, Gmag}


        '''Get info about detected star from Gaia'''
        
        #match Gaia star with the detection
        SR = 0.005
        star_df = match_RADec(star_RA, star_dec, gaia, SR)
        
        gend = time.time()
        gaiatime += gend-gstart # Added timer - MJM
        
        '''Find best matched occultation kernel'''
        ostart = time.time() # Added timer - MJM
        significancee = eventData[5]
        
        #21-09 Roman A. no more event types
        
        scope= filepath.name.split('_')[5].split('.')[0]
        filename=filepath.name.split('.')[0]
        diff_results.append((star_num, eventData[2], eventData[3], eventData[4], 
                              kernelMatch(eventData, np.array(kernel_set), star_num, matched_dir, star_df,scope,filename), 
                              star_df['Gaia_RA'], star_df['Gaia_dec'], star_df['GMAG'], star_df['Gaia_B-R']))
       
        # if event_type == 'geometric':
#        geo_results.append((star_num, eventData[2], eventData[3], eventData[4], 
#                            kernelMatch(eventData, kernel_set, star_num, archive_dir, star_df), 
#                            star_df['Gaia_RA'], star_df['Gaia_dec'], star_df['GMAG'], star_df['Gaia_B-R']))
        oend = time.time() # Added timer - MJM
        occulttime += oend - ostart # Added timer - MJM

    '''save results'''
    
    diff_save_file = archive_dir.joinpath(str(obs_date) + '_' + telescope + '_diffraction_kernelMatches.txt')
    #geo_save_file = archive_dir.joinpath(str(obs_date) + '_' + telescope + '_geometric_kernelMatches.txt')

    with open(diff_save_file, 'w') as file:
        print('Saving files...')
        #print(diff_results)
        file.write('#starNumber  starX  starY  eventTime  kernelIndex  Chi2  startLocation  ObjectD   StellarD   b    shift    GaiaRA    GaiaDec   GaiaGMAG    GaiaB-R\n')
            
        for line in diff_results:
            
            #events that didn't pass kernel matching
            if line[4] == -1:
                continue
            
            file.write('%s %f %f %s %i %f %i %f %f %f %f %f %f %f %f\n' %(line[0], line[1], line[2], line[3], 
                                                  line[4][0], line[4][1], line[4][2], 
                                                  line[4][3][2], line[4][3][3], line[4][3][4], line[4][3][5],
                                                  line[5], line[6], line[7], line[8]))
    
    # with open(geo_save_file, 'w') as file:
        
    #     file.write('#starNumber  starX  starY  eventTime  kernelIndex  Chi2  startLocation  ObjectD   StellarD   b    shift    GaiaRA   GaiaDec   GaiaGMAG    GaiaB-R\n')
        
    #     for line in geo_results:
            
    #         #events that didn't pass kernel matching
    #         if line[4] == -1:
    #             continue
            
    #         file.write('%s %f %f %s %i %f %i %f %f %f %f %f %f %f %f\n' %(line[0], line[1], line[2], line[3], 
    #                                               line[4][0], line[4][1], line[4][2], 
    #                                               line[4][3][2], line[4][3][3], line[4][3][4], line[4][3][5],
    #                                               line[5], line[6], line[7], line[8]))


    print('Time spent on astrometry... %f' % transformtime)
    print('Time spent on Gaia... %f' % gaiatime)
    print('Time spend on finding best kernel... %f' % occulttime)

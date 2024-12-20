"""
Filename:   colibri_photometry.py
Author(s):  Peter Quigley
Contact:    pquigley@uwo.ca
Created:    Fri Oct 21 09:23:04 2022
Updated:    Fri Oct 21 09:23:04 2022
    
Usage: import colibri_primary_filter as cpf
"""

# Module imports
import os, sys
import numpy as np
import numpy.ma as ma
import cython
import sep
import pathlib
import datetime
from copy import deepcopy
from astropy.io import fits
from astropy.convolution import convolve_fft, RickerWavelet1DKernel
from astropy.time import Time
from scipy.signal import convolve

# Custom Script Imports


# Cython-Numpy Interface
cimport numpy as np
np.import_array()

# Compile Typing Definitions
ctypedef np.uint16_t UI16
ctypedef np.float64_t F64


##############################
## Star Depection and Location
##############################

@cython.boundscheck(False)
@cython.wraparound(False)
def initialFind(np.ndarray[F64, ndim=2] img_data, float detect_thresh):
    """
    Locates the stars in an image using the sep module and completes
    preliminary photometry on the image.

    Parameters:
        img_data(arr): 2D array of image flux data
        detect_thresh (float): Detection threshold for star finding
        
    Returns:
        star_chars (arr): [x, y, half light radius] of all stars in pixels
    """

    ## Type definitions
    cdef float thresh
    cdef np.ndarray img_copy,stars,stars_profile
    cdef tuple star_chars

    ## Extract the background from the initial time slice and subtract it
    img_copy = deepcopy(img_data)
    img_bkg  = sep.Background(img_copy)
    img_bkg.subfrom(img_copy)
    
    ## Set detection threshold to (mean * detect_thresh sigma)
    thresh = detect_thresh*img_bkg.globalrms
    
    ## Identify stars in the image and approximate their light profile radius
    stars = sep.extract(img_copy,thresh)
    stars_profile = np.sqrt(stars['npix'] / np.pi) / 2
    
    ## Create tuple of (x,y,r) positions of each star
    star_chars = tuple(zip(stars['x'],stars['y'],stars_profile))
    
    return star_chars


@cython.wraparound(False)
def refineCentroid(np.ndarray[F64, ndim=2] img_data,
                   str time, #TODO: remove this
                   np.ndarray[F64, ndim=2] star_coords,
                   float sigma):
    """
    Refines the centroid for each star for an image based on previous coords
    
    Parameters:
        img_data (arr): 2D array of flux data for a single image
        time (str): Header time of image
        star_coords (list): Coordinates of stars in previous image
        sigma (float): Guassian sigma weighting
            
    Return:
        coords (arr): 2D array of star coordinates (column1=x,column2=y)
        time (str): Header time of image
    """
    
    ## Type definitions
    cdef list x_initial,y_initial
    cdef np.ndarray new_pos

    ## Generate initial (x,y) coordinates from star_coords
    x_initial = [pos[0] for pos in star_coords]
    y_initial = [pos[1] for pos in star_coords]
    
    ## Calculate more accurate object centroids using windowed algorithm
    new_pos = np.array(sep.winpos(img_data, x_initial, y_initial, sigma, subpix=5))[0:2, :]

    ## Extract improved (x,y) coordinates as lists and zip them together
    #x = new_pos[:][0].tolist()
    #y = new_pos[:][1].tolist()
    #return tuple(zip(x,y)), time
    
    return new_pos.transpose(),time


##############################
## Image Manipulation Tools
##############################

def sumFlux(np.ndarray[F64, ndim=2] img_data,
            np.ndarray[F64, ndim=2] star_coords,
            int l):
    '''
    Function to sum up flux in square aperture of size.
    Depreciated for sep.sum_circle.
    
    Parameters:
        data (arr): 2D image flux array
        x_coords (arr/list): x-coordinates of the stars
        y_coords (arr/list): y-coordinates of the stars
        l (int): "Radius" of the square aperture
        
    Returns:
        star_fluxes (list): Fluxes of each star
    '''
    
    ## Type definitions
    cdef list star_flux_list,star_fluxes
    
    ## Extract flux data for each detected star flagged by star_coords
    star_flux_list = [[img_data[y][x]
                       for x in range(int(star_coords[star,0] - l),
                                      int(star_coords[star,0] + l + 1))
                       for y in range(int(star_coords[star,1] - l),
                                      int(star_coords[star,1] + l + 1))]
                       for star in range(0,len(star_coords))]
    
    ## Obtain an integrated flux for each star over this minute directory
    star_fluxes = [sum(fluxlist) for fluxlist in star_flux_list]
    
    return star_fluxes


@cython.wraparound(False)
def clipCutStars(np.ndarray[F64, ndim=1] x, 
                 np.ndarray[F64, ndim=1] y,
                 int x_length,
                 int y_length):
    """
    Sets flux measurements too close to the field of view boundary to zero in
    order to prevent fadeout. To be used for a single image.
    
    Parameters:
        x (arr): 1D array containing the x-coordinates of stars
        y (arr): 1D array containing the y-coordinates of stars
        x_length (int): Length of image in the x-direction
        y_length (int): Length of image in the y-direction

    Returns:
        bad_ind (arr): Indices of stars deemed too near to the image edge
    """
    
    ## Get list of indices where the stars are too near to the edge
    cdef int pixel_buffer = 20
    cdef np.ndarray bad_ind = np.append(np.where((x < pixel_buffer) | \
                                                 (x > x_length - pixel_buffer))[0], \
                                        np.where((y < pixel_buffer) | \
                                                 (y > y_length - pixel_buffer))[0])
        
    return bad_ind


@cython.wraparound(False)
def clipCutStars3D(np.ndarray[F64, ndim=2] x, 
                   np.ndarray[F64, ndim=2] y,
                   int x_length,
                   int y_length):
    """
    Sets flux measurements too close to the field of view boundary to zero in
    order to prevent fadeout. To be used with stacked arrays.
    
    Parameters:
        x (arr): 2D array containing the x-coordinates of stars
        y (arr): 2D array containing the y-coordinates of stars
        x_length (int): Length of image in the x-direction
        y_length (int): Length of image in the y-direction

    Returns:
        bad_ind (arr): Indices of stars deemed too near to the image edge
    """
    
    ## Get list of indices where the stars are too near to the edge
    cdef int pixel_buffer = 20
    cdef np.ndarray bad_ind = np.append(np.where((x < pixel_buffer) | \
                                                 (x > x_length - pixel_buffer))[0], \
                                        np.where((y < pixel_buffer) | \
                                                 (y > y_length - pixel_buffer))[0])
        
    return bad_ind


##############################
## Correct Star Drift
##############################

@cython.wraparound(False)
def averageDrift(np.ndarray[F64, ndim=2] star_coords1,
                 np.ndarray[F64, ndim=2] star_coords2,
                 F64 time1,
                 F64 time2):
    """
    Determines the median x/y drift rates of all stars in a minute (first to
    last image)
    
        Parameters:
            star_coords1 (arr): 2D array of star positions for first frame
            star_coords2 (arr): 2D array of star positions for last frame
            times (list): Header times of each position in [star1,star2] order
            
        Returns: 
            x_drift_rate (arr): Median x drift rate [px/star]
            y_drift_rate (arr): Median y drift rate [px/star]
    """
    
    ## Type definitions
    cdef float time_interval,x_drift_rate,y_drift_rate
    cdef np.ndarray x_drifts,y_drifts
    
    ## Find the time difference between the two frames
    time_interval = np.subtract(time2,time1)
    
    ## Determine the x- and y-drift of each star between frames (in px)
    x_drifts = np.subtract(star_coords2[:,0], star_coords1[:,0])
    y_drifts = np.subtract(star_coords2[:,1], star_coords1[:,1])
    
    ## Calculate median drift rate across all stars (in px/s)
    x_drift_rate = np.median(x_drifts/time_interval)
    y_drift_rate = np.median(y_drifts/time_interval)
    
    return x_drift_rate,y_drift_rate


@cython.wraparound(False)
def timeEvolve(np.ndarray[F64, ndim=2] curr_img,
               np.ndarray[F64, ndim=2] prev_img,
               str img_time,
               int r,
               int num_stars,
               tuple pix_length,
               tuple pix_drift=(0,0)):
    """
    Adjusts aperture based on star drift and calculates flux in aperture
    
        Parameters:
            curr_img (arr): 2D image flux array of current image
            prev_img (arr): 2D image flux array of previous sequential image
            img_time (float): Header image time
            r (int): Aperture radius to sum flux (in pixels)
            numStars (int): Number of stars in image
            pix_length (tuple): Image pixel length as (x_length,y_length)
            pix_drift (tuple): Image drift rate as (x_drift,y_drift) (in px/s)
            
        Returns:
            star_data (tuple): New star coordinates, image flux, time as tuple
     """
    
    ## Type definitions
    cdef float drift_time
    cdef double curr_time
    cdef list fluxes
    cdef tuple star_data
    cdef np.ndarray x,y,stars,x_clipped,y_clipped
    cdef int inner_annulus = 5
    cdef int outer_annulus = 8
    cdef int Edge_buffer = 10
    
    ## Calculate time between prev_img and curr_img
    curr_time  = Time(img_time,scale='utc',precision=9).unix
    # print(img_time, curr_time,Time(img_time,format='isot',scale='utc',precision=9).unix)
    drift_time = curr_time - prev_img[0,3]
    
    ## Incorporate drift into each star's coords based on time since last frame
    x = np.array([prev_img[ind, 0] + pix_drift[0]*drift_time for ind in range(0, num_stars)])
    y = np.array([prev_img[ind, 1] + pix_drift[1]*drift_time for ind in range(0, num_stars)])
    
    #2023-01-16 Roman A. commented out removing stars that are drifting closer to the edge
    
    ## Eliminate stars too near the field of view boundary
    # edge_stars = clipCutStars(x, y, *pix_length)
    # edge_stars = np.sort(np.unique(edge_stars))
    # x_clipped  = np.delete(x,edge_stars)
    # y_clipped  = np.delete(y,edge_stars)
    
    ## Assign integrated flux to each star, and then insert 0 for edge stars
    #TODO: be clever about doing this with numpy arrays
    # fluxes = (sep.sum_circle(curr_img, 
    #                          x_clipped, y_clipped, 
    #                          r, 
    #                          bkgann = (r + 6., r + 11.))[0]).tolist()
    fluxes = (sep.sum_circle(curr_img, 
                             x, y, 
                             r, 
                             bkgann = (inner_annulus, outer_annulus))[0]).tolist()
    # for i in edge_stars:
    #     fluxes.insert(i,0)
        
    ## Return star data as layered tuple as (x,y,integrated flux,array of curr_time)
    star_data = tuple(zip(x, y, fluxes, np.full(len(fluxes), curr_time)))
    return star_data


@cython.wraparound(False)
def timeEvolve3D(np.ndarray[F64, ndim=3] img_stack,
                 np.ndarray[F64, ndim=2] first_img,
                 list img_times,
                 int r,
                 int num_stars,
                 tuple pix_length,
                 tuple pix_drift=(0,0)):
    """
    Adjusts aperture based on star drift and calculates flux in aperture.
    !!!Currently does not work!!!
    
        Parameters:
            img_stack (arr): 3D image flux array of stacked minute directory
            img_times (list): List of header image times (not formatted)
            r (int): Aperture radius to sum flux (in pixels)
            numStars (int): Number of stars in image
            pix_length (tuple): Image pixel length as (x_length,y_length)
            pix_drift (tuple): Image drift rate as (x_drift,y_drift) (in px/s)
            
        Returns:
            star_data (tuple): New star coordinates, image flux, time as tuple
     """
    
# =============================================================================
#     ## Type definitions
#     cdef int num_frames,ind,frame,pixel_buffer
#     cdef list fluxes
#     cdef tuple star_data
#     cdef np.ndarray unix_time,drift_time,x,y,stars,x_clipped,y_clipped
#     cdef np.ndarray xmask,ymask,xymask,fluxarr
#     
#     ## Calculate time between subsequent frames
#     num_frames = len(img_times)
#     unix_time  = Time(img_times,precision=9).unix
#     drift_time = unix_time -  first_img[1,3]
#     
#     ## Incorporate drift into each star's coords based on time since last frame
#     ## Returns a 2D array with each 0-axis slice being a frame
#     x = np.array([[first_img[ind, 0] + pix_drift[0]*drift_time[frame] for ind in range(0, num_stars)]
#                    for frame in range(0,num_frames)])
#     y = np.array([[first_img[ind, 1] + pix_drift[1]*drift_time[frame] for ind in range(0, num_stars)]
#                    for frame in range(0,num_frames)])
#     
#     ## Eliminate stars too near the field of view boundary
#     edge_stars = clipCutStars(x, y, *pix_length)
#     edge_stars = np.sort(np.unique(edge_stars))
#     x_clipped  = np.delete(x,edge_stars)
#     y_clipped  = np.delete(y,edge_stars)
#     
#     ## Assign integrated flux to each star, and then insert 0 for edge stars
#     #TODO: be clever about doing this with numpy arrays
#     fluxes = (sep.sum_circle(curr_img, 
#                              x_clipped, y_clipped, 
#                              r, 
#                              bkgann = (r + 6., r + 11.))[0]).tolist()
#     for i in edge_stars:
#         fluxes.insert(i,0)
#         
#     ## Return star data as layered tuple as (x,y,integrated flux,array of curr_time)
#     star_data = tuple(zip(x, y, fluxes, np.full(len(fluxes), curr_time)))
#     return star_data
#     
#     ## Mask the out-of-bounds stars
#     pixel_buffer = 20
#     xmask  = ma.masked_outside(x, pixel_buffer,pix_length[0]-pixel_buffer)[1]
#     ymask  = ma.masked_outside(y, pixel_buffer,pix_length[1]-pixel_buffer)[1]
#     xymask = ma.array(xmask, mask=ymask)[1]
#     
#     ## Generate fluxes for each star and then mask the invalid ones
#     fluxes  = [sep.sum_circle(img_stack[frame],x[frame],y[frame],r,bkgann = (r + 6., r + 11.))[0]
#                for frame in range(0,num_frames)]
#     fluxarr = ma.array(fluxes, mask=xymask, fill_value=0.0)
#     
#     ## Return star data as layered tuple as (x,y,integrated flux,array of curr_time)
#     star_data = tuple(zip(x, y, fluxes, np.full((num_frames,num_stars), np.vstack(unix_time))))
#     return np.transpose(star_data,axes=[0,2,1]) # rotate axes to get correct shape
# =============================================================================

    #TODO: finish this method
    raise NotImplementedError("Do not use this method! It is not working correctly.")
    pass
    

@cython.boundscheck(False)
@cython.wraparound(False)
def getStationaryFlux(np.ndarray[F64, ndim=3] img_stack,
                      np.ndarray[F64, ndim=2] prev_img,
                      list img_times,
                      int r,
                      int num_stars,
                      tuple pix_length):
    """
    Gets the position of each star and finds the star flux at each frame,
    assuming no drift of the centroid.
    
        Parameters:
            img_stack (arr): 3D image flux array of stacked minute directory
            img_times (list): List of header image times (not formatted)
            r (int): Aperture radius to sum flux (in pixels)
            numStars (int): Number of stars in image
            pix_length (tuple): Image pixel length as (x_length,y_length)
            pix_drift (tuple): Image drift rate as (x_drift,y_drift) (in px/s)
            
        Returns:
            star_data (tuple): New star coordinates, image flux, time as tuple
     """
    
    ## Type definitions
    cdef np.ndarray unix_time,x,y,
    cdef np.ndarray edge_stars,x_clipped,y_clipped
    cdef np.ndarray fluxes,star_data
    cdef int inner_annulus = 5
    cdef int outer_annulus = 8
    cdef int Edge_buffer = 10
    
    ## Calculate time between prev_img and curr_img
    unix_time  = Time(img_times,precision=9).unix
    
    ## Incorporate drift into each star's coords based on time since last frame
    x = np.array([prev_img[ind, 0] for ind in range(0, num_stars)])
    y = np.array([prev_img[ind, 1] for ind in range(0, num_stars)])
    
    ## Eliminate stars too near the field of view boundary
    
    #2023-01-16 Roman A. commented out removing stars that are drifting closer to the edge
    
    # edge_stars = clipCutStars(x, y, *pix_length)
    # edge_stars = np.sort(np.unique(edge_stars))
    # x_clipped  = np.delete(x,edge_stars)
    # y_clipped  = np.delete(y,edge_stars)
    
    ## Assign integrated flux to each star, and then insert 0 for edge stars
    #TODO: be clever about doing this with numpy arrays
    
    # fluxes = np.array([sep.sum_circle(img_stack[frame],
    #                                   x_clipped,y_clipped,
    #                                   r, 
    #                                   bkgann=(r + 6., r + 11.))[0]
    #                    for frame in range(len(img_stack))])
    
    fluxes = np.array([sep.sum_circle(img_stack[frame],
                                      x,y,
                                      r, 
                                      bkgann=(inner_annulus, outer_annulus))[0]
                       for frame in range(len(img_stack))])

    # for bad_ind in edge_stars:
    #     fluxes = np.insert(fluxes, bad_ind, 0, axis=1)
        
        
    ## Return star data as layered tuple as (x,y,integrated flux,array of curr_time)
    star_data = np.array([np.stack((x, y, fluxes[frame], np.full(len(x), unix_time[frame])),axis=1)
                          for frame in range(len(img_stack))])
    return star_data


##############################
## Dip Detection
##############################
#TODO: Cythonize this function

def dipDetection(fluxProfile, kernel, num, sigma_threshold):
    """
    Checks for geometric dip, and detects dimming using Ricker Wavelet kernel
    
    Parameters:
        fluxProfile (arr): Light curve of star (array of fluxes in each image)
        kernel (arr): Ricker wavelet kernel
        num (int): Current star number
        sigma_threshold (float): sigma_threshold for determining stars
        
    Returns:
        frameNum (int): Frame number of detected event (-1 for no detection
                        or -2 if data unusable)
        lc_arr (arr): Light curve as an array (empty list if no event
                      detected)
        event_type (str): Keyword indicating event type (empty string if
                          no event detected)
    """
    

    '''' Prunes profiles'''
    light_curve = np.trim_zeros(fluxProfile)
    
    if len(light_curve) == 0:
        print(f"Empty profile: {num}")
        return -2, [], [], np.nan, np.nan, np.nan, np.nan, -2, np.nan  # reject empty profiles
   
    
    '''perform checks on data before proceeding'''

    
    FramesperMin = 2400      #ideal number of frames in a directory (1 minute)
    minSNR = 0               #median/stddev limit to look for detections
    minLightcurveLen = FramesperMin/4    #minimum length of lightcurve
    
    # reject lightcurves that dip below 500
    # Designed to remove bright satellites
    if np.min(light_curve) < 500:
        print(f"Negative flux: star {num}")
        return -2, [], [], np.nan, np.nan, np.nan, np.nan, -2, np.nan

    # reject stars with SNR too low
    # if np.median(light_curve)/np.std(light_curve) < minSNR:
    #     print(f"Signal to Noise too low: star {num}")
    #     return -2, [], [], np.nan, np.nan, np.nan, np.nan, -2, np.nan  

    # reject stars that go out of frame to rapidly
    if len(light_curve) < minLightcurveLen:
        print(f"Light curve too short: star {num}")
        return -2, [], [], np.nan, np.nan, np.nan, np.nan, -2, np.nan  



    '''convolve light curve with ricker wavelet kernel'''
    #will throw error if try to normalize (sum of kernel too close to 0)
    conv = convolve(light_curve, kernel, mode='valid')    #convolution of light curve with Ricker wavelet
    minLoc = np.argmin(conv)    #index of minimum value of convolution
    minVal = np.min(conv)          #minimum of convolution



    #bkgZone is the convolution without the area where the dip is. 
    # Define the number of frames to exclude around the dip
    exclusion_zone = 5

    # Create the background zone by excluding the dip region
    start_exclusion = max(0, minLoc - exclusion_zone)
    end_exclusion = min(len(conv), minLoc + exclusion_zone + 1)
    bkgZone = np.concatenate((conv[:start_exclusion], conv[end_exclusion:]))
    
    lightcurve_std=np.std(light_curve)
    
    conv_bkg_mean=np.mean(bkgZone)
    
    significance=(conv_bkg_mean-minVal)/np.std(bkgZone) #significance of the event x*sigma
    
    # Calculate the number of zeros to add on each side
    padding_length = (len(light_curve) - len(conv)) // 2

    # Create arrays of for padding
    padding = np.ones(padding_length) * conv_bkg_mean

    # Concatenate zeros to the beginning and end of the convolution result
    conv_padded = np.concatenate((padding, conv, padding))

    # Ensure the length of the convolution matches the length of the light curve
    if len(conv_padded) < len(light_curve):
        conv_padded = np.concatenate((conv_padded, [0]))

    # Update the return statements to use conv_padded
    if significance >= sigma_threshold:
        critFrame = minLoc + (len(kernel.array) // 2)
        print(f"Found significant dip in star: {num} at frame: {critFrame}")
        return critFrame, light_curve, conv_padded, lightcurve_std, np.mean(light_curve), np.std(bkgZone), conv_bkg_mean, minVal, significance
    else:
        return -1, light_curve, conv_padded, np.nan, np.nan, np.nan, np.nan, np.nan, significance

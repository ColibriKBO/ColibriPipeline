#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename:   colibri_primary_filter
Author(s):  Peter Quigley
Contact:    pquigley@uwo.ca
Created:    Fri Oct 21 09:23:04 2022
Updated:    Fri Oct 21 09:23:04 2022
    
Usage: import colibri_primary_filter as cpf
Includes: dipDetection, timeEvolve
"""

# Module imports
import os, sys
import numpy as np
import cython as cy
import sep
import pathlib
import datetime
from astropy.io import fits
from astropy.convolution import convolve_fft, RickerWavelet1DKernel
from astropy.time import Time

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
    cdef np.ndarray img_copy,img_bkg,stars,stars_profile
    cdef tuple star_chars

    ## Extract the background from the initial time slice and subtract it
    img_copy = img_data.copy()
    img_bkg  = sep.Background(img_copy)
    img_bkg.subfrom(img_copy)
    
    ## Set detection threshold to (mean * detect_thresh sigma)
    thresh = detect_thresh*img_bkg.globalrms
    
    ## Identify stars in the image and approximate their light profile radius
    stars = sep.extract(img_copy,thresh)
    stars_profile = np.sqrt(stars['npix'] / np.pi) / 2
    
    ## Create tuple of (x,y,r) positions of each star
    star_chars = zip(stars['x'],stars['y'],stars_profile)
    
    return star_chars


def refineCentroid(np.ndarray[F64, ndim=2] img_data,
                   str time, #TODO: remove this
                   list star_coords,
                   float sigma)
    """
    Refines the centroid for each star for an image based on previous coords
    
    Parameters:
        img_data (arr): 2D array of flux data for a single image
        time (str): Header time of image
        coords (list): Coordinates of stars in previous image
        sigma (float): Guassian sigma weighting
            
    Return:
        Coords (tuple): two zipped lists of star coordinates
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
    
    return newpos[:][:2],time


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
    star_flux_list = [[data[y][x]
                       for x in range(int(star_coords[star,0] - l),
                                      int(star_coords[star,0] + l + 1)),
                       for y in range(int(star_coords[star,1] - l),
                                      int(star_coords[star,1] + l + 1))]
                       for star in range(0,len(star_coords))]
    
    ## Obtain an integrated flux for each star over this minute directory
    star_fluxes = [sum(fluxlist) for fluxlist in star_flux_list]
    
    return star_fluxes


def clipCutStars(np.ndarray[F64, ndim=1] x, 
                 np.ndarray[F64, ndim=1] y,
                 int x_length,
                 int y_length):
    """
    Sets flux measurements too close to the field of view boundary to zero in
    order to prevent fadeout
    
    Parameters:
        x (arr): 1D array containing the x-coordinates of stars
        y (arr): 1D array containing the y-coordinates of stars
        x_length (int/float): Length of image in the x-direction
        y_length (int/float): Length of image in the y-direction

    Returns:
        ind (arr): Indices of stars deemed too near to the image edge
    """
    
    ## Get list of indices where the stars are too near to the edge
    cdef int pixel_buffer = 20
    cdef np.ndarray ind   = np.append(np.where((x < edgeThresh) | \
                                               (x > x_length - edgeThresh))[0], \
                                      np.where((y < edgeThresh) | \
                                               (y > y_length - edgeThresh))[0])
        
    return ind


##############################
## Correct Star Drift
##############################

def averageDrift(np.ndarray[F64, ndim=2] star_coords1,
                 np.ndarray[F64, ndim=2] star_coords2,
                 list times):
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
    times = Time(times, precision=9).unix
    time_interval = np.subtract(times[1],times[0],dtype=np.float64)
    
    ## Determine the x- and y-drift of each star between frames (in px)
    x_drifts = np.subtract(star_coords2[:,0], star_coords1[:,0])
    y_drifts = np.subtract(star_coords2[:,1], star_coords1[:,1])
    
    ## Calculate median drift rate across all stars (in px/s)
    x_drift_rate = np.median(x_drifts/time_interval)
    y_drift_rate = np.median(y_drifts/time_interval)
    
    return x_drifts


def timeEvolve(np.ndarray[F64, ndim=2] curr_img,
               np.ndarray[F64, ndim=2] prev_img,
               float img_time,
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
    cdef float curr_time,drift_time
    cdef list fluxes
    cdef tuple star_data
    cdef np.ndarray x,y,stars,x_clipped,y_clipped
    
    ## Calculate time between prev_img and curr_img
    curr_time  = Time(img_time,precision=9).unix
    drift_time = frame_time - prev_img[1,3]
    
    ## Incorporate drift into each star's coords based on time since last frame
    x = np.array([prev_img[ind, 0] + pix_drift[0]*drift_time for ind in range(0, num_stars)])
    y = np.array([prev_img[ind, 1] + pix_drift[1]*drift_time for ind in range(0, num_stars)])
    
    ## Eliminate stars too near the field of view boundary
    edge_stars = clipCutStars(x, y, *pix_length)
    edge_stars = np.sort(np.unique(edge_stars))
    x_clipped  = np.delete(x,edge_stars)
    y_clipped  = np.delete(y,edge_stars)
    
    ## Assign integrated flux to each star, and then insert 0 for edge stars
    #TODO: be clever about doing this with numpy arrays
    fluxes = (sep.sum_circle(curr_img, 
                             x_clipped, y_clipped, 
                             r, 
                             bkgann = (r + 6., r + 11.))[0]).tolist()
    for i in edge_stars:
        fluxes.insert(i,0)
        
    ## Return star data as layered tuple as (x,y,integrated flux,array of curr_time)
    star_data = tuple(zip(x, y, fluxes, np.full(len(fluxes), curr_time)))
    return star_data    
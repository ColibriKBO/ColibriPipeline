"""
Filename:   colibri_primary_filter
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
from time import time
from astropy.io import fits
from astropy.convolution import convolve_fft, RickerWavelet1DKernel
from astropy.time import Time

# Custom Script Imports
from bitconverter import conv_12to16

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
    cdef float curr_time,drift_time
    cdef list fluxes
    cdef tuple star_data
    cdef np.ndarray x,y,stars,x_clipped,y_clipped
    
    ## Calculate time between prev_img and curr_img
    curr_time  = Time(img_time,precision=9).unix
    drift_time = curr_time - prev_img[1,3]
    
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
    
    ## Calculate time between prev_img and curr_img
    unix_time  = Time(img_times,precision=9).unix
    
    ## Incorporate drift into each star's coords based on time since last frame
    x = np.array([prev_img[ind, 0] for ind in range(0, num_stars)])
    y = np.array([prev_img[ind, 1] for ind in range(0, num_stars)])
    
    ## Eliminate stars too near the field of view boundary
    edge_stars = clipCutStars(x, y, *pix_length)
    edge_stars = np.sort(np.unique(edge_stars))
    x_clipped  = np.delete(x,edge_stars)
    y_clipped  = np.delete(y,edge_stars)
    
    ## Assign integrated flux to each star, and then insert 0 for edge stars
    #TODO: be clever about doing this with numpy arrays
    
    fluxes = np.array([sep.sum_circle(img_stack[frame],
                                      x_clipped,y_clipped,
                                      r, 
                                      bkgann=(r + 6., r + 11.))[0]
                       for frame in range(len(img_stack))])

    for bad_ind in edge_stars:
        fluxes = np.insert(fluxes, bad_ind, 0, axis=1)
        
        
    ## Return star data as layered tuple as (x,y,integrated flux,array of curr_time)
    star_data = np.array([np.stack((x, y, fluxes[frame], np.full(len(x), unix_time[frame])),axis=1)
                          for frame in range(len(img_stack))])
    return star_data


##############################
## Bitreading Functions
##############################

@cython.wraparound(False)
def noDriftMask(np.ndarray[UI16, ndim=2] star_ind,
                int box_dim=7,
                bint gain_high=True):
    """
    Create an array for reading in relevant bits (containing stars) for this
    minute directory. Must pre-eliminate stars too close to the edge

    Args:
        star_ind (list): Pixel coordinates of the star centrodis to be analyzed.
        box_dim (int, optional): Width of integration box (in px). Must be odd
        gain_high (bool, optional): Analyze high gain image over the low gain
                                    image. Defaults to True.

    Returns:
        seek_ind (arr): Array of bits to seek from the RCD file.
        
    """
    
    ## Specific container definitions
    cdef int half_box = box_dim//2
    cdef np.ndarray seek_ind = np.empty((len(star_ind)*box_dim,2))

    ## Index definitions
    cdef int i,j
    cdef np.ndarray star


    ## Loop to read in and sum the pixel box.
    ## Uses two cases for half-box being even and odd
    if half_box%2 == 0: # even half-box case
        for i,star in enumerate(star_ind):
            if star[1]%2 == 0: # even pixel case
                for j in range(box_dim):
                    seek_ind[i*box_dim + j,0] = 384 + 3072*(gain_high + 2*(star[0] + j - half_box)) + 3*(star[1] - half_box)
                    seek_ind[i*box_dim + j,1] = i
                    
            else: # odd pixel case
                for j in range(box_dim):
                    seek_ind[i*box_dim + j,0] = 384 + 3072*(gain_high + 2*(star[0] + j - half_box)) + 3*(star[1] - half_box - 1)
                    seek_ind[i*box_dim + j,1] = i
                
            
    else: # odd half-box case
        for i,star in enumerate(star_ind):
            if star[1]%2 == 1: # even pixel case
                for j in range(box_dim):
                    seek_ind[i*box_dim + j,0] = 384 + 3072*(gain_high + 2*(star[0] + j - half_box)) + 3*(star[1] - half_box)
                    seek_ind[i*box_dim + j,1] = i
                    
            else: # odd pixel case
                for j in range(box_dim):
                    seek_ind[i*box_dim + j,0] = 384 + 3072*(gain_high + 2*(star[0] + j - half_box)) + 3*(star[1] - half_box - 1)
                    seek_ind[i*box_dim + j,1] = i
                

    ## Sort seek_inds to eliminate backtracking and return the inds
    seek_ind = seek_ind[seek_ind[:,0].argsort()]
    ind_loc  = np.array([np.where(seek_ind[:,1] == i) for i in range(len(star_ind))])
    return seek_ind[:,0],ind_loc


@cython.wraparound(False)
def fluxBitString(list imgdir,
                  np.ndarray[UI16, ndim=2] star_coords,
                  int box_dim=7,
                  int l=2048,
                  int pixel_buffer=20,
                  bint gain_high=True):
    """
    Obtain the flux of stars for a list of images in a given minute. Uses a 
    square flux aperature, eliminates stars near the boundaries, and uses a
    selective bit reading method for I/O.

    Args:
        imgdir (list): List of image paths to read in.
        star_coords (list): Pixel coordinates of the star centrodis to be analyzed.
        box_dim (int, optional): Width of integration box (in px). Must be odd
                                 to be symmetric.
        l (int, optional): Dimension of the square image.
        pixel_buffer (int, optional): Buffer width from the image edge that
                                      a star must be to be analyzed. 
        timestamp (bool, optional): Return the image timestamp. Defaults to True.
        gain_high (bool, optional): Analyze high gain image over the low gain
                                    image. Defaults to True.

    Returns:
        None.

    """

    ## Type definitions
    cdef int half_box,ints_to_read
    cdef int frame,i,ind
    cdef np.ndarray clipped_ind,seek_ind,identifier,bit_buffer,star,reshape16b,partial_flux,flux,imgtimes

    ## Integration variables
    half_box = box_dim//2
    ints_to_read = (box_dim + 1)*3//2
    #print(ints_to_read)
    
    
    ## Eliminate stars too close to the border
    t0 = time()
    clipped_ind = star_coords[np.all(star_coords > pixel_buffer, axis=1) & \
                           np.all(star_coords < l - pixel_buffer, axis=1)]
    
    ## Get indexes of the stars to sum and create the bit buffer for tmp storage
    seek_ind,identifier = noDriftMask(clipped_ind,box_dim,gain_high)
    print("Star, seek: ",np.shape(clipped_ind),np.shape(seek_ind))
    bit_buffer = np.empty(len(seek_ind)*ints_to_read,dtype=np.uint8)
    flux = np.empty((len(imgdir),len(clipped_ind)))
    
    ## For each image in the directory, read the timestamp and then seek
    ## indices and read in the relevant bits for all stars. Then convert
    ## to uint16 type. Group the relevant integers and sum the fluxes.
    cdef list timestamps = []
    for frame,path in enumerate(imgdir):
        #print(path)
        with open(path,'rb') as fid:
            #fid.seek(-1,2)
            #print(fid.tell(), np.max(seek_ind))
            
            # Get frame timestamp
            fid.seek(152,0)
            timestamps.append(fid.read(29).decode('utf-8'))
            
            # Get bitstring
            for i,ind in enumerate(seek_ind):
                fid.seek(ind,0)
                #print(ind,fid.tell())
                #print(ind,identifier[i])
                #bit_buffer[i*ints_to_read:(i+1)*ints_to_read] = np.fromfile(fid, dtype=np.uint8, count=ints_to_read)
                bit_buffer[i*ints_to_read:(i+1)*ints_to_read] = np.fromfile(fid, dtype=np.uint8, count=ints_to_read)
            
            # Convert 8-bit imposter ints to 16-bit proper ints, reshape, and sum
            #print("Bitbuffer",np.shape(bit_buffer))
            reshape16b = (conv_12to16(bit_buffer)).reshape((len(seek_ind),box_dim+1))
            #print("Reshape16b", np.shape(reshape16b))
            partial_flux = np.sum(reshape16b,axis=1).transpose()
            #print(partial_flux)
            for i,star in enumerate(identifier):
                flux[frame][i] = np.sum(partial_flux[star])

    
    # Convert time to unix and zip
    imgtimes = Time(timestamps,precision=9).unix
    return clipped_ind,flux,imgtimes


@cython.wraparound(False)
def fluxFromBits(filename,
                 list star_ind,
                 int box_dim=7,
                 bint timestamp=True,
                 bint gain_high=True):
    """
    Read specific stars from their pixel coordinates and returns an integrated
    flux calculated using a box method. Only works with 2048x2048 RCD images.

    Args:
        filename (str/Path): Path or pathlib object to the frame to be analyzed.
        star_ind (list): Pixel coordinates of the stars to be analyzed.
        box_dim (int, optional): Width of integration box (in px). Must be odd
                                 to be symmetric.
        timestamp (bool, optional): Return the image timestamp. Defaults to True.
        gain_high (bool, optional): Analyze high gain image over the low gain
                                    image. Defaults to True.

    Returns:
        flux (arr): Flux corresponding to the star_ind stars integrated over a
                    square area.
        img_time (str, optional): Image timestamp. Returned if timestamp=True.
        
    """
    
    raise DeprecationWarning
    
    ## Type definitions
    cdef int half_box,ints_to_read
    cdef str img_time
    cdef int i,j
    cdef np.ndarray bit_buffer, flux
    
    ## Assert that the integration box be symmetric
    #assert box_dim%2 == 1
    
    ## Integration variables
    half_box = box_dim//2
    ints_to_read = (box_dim + 1)*(3/2)
    
    bit_buffer = np.empty((box_dim, ints_to_read),dtype=np.uint8)
    flux = np.zeros(len(star_ind))
    
    
    ## Loop to read in and sum the pixel box.
    ## Uses two cases for half-box being even and odd
    if half_box%2 == 0: # even half-box case
        with open(filename, 'rb') as fid:
            for i,star in enumerate(star_ind):
                if star[1]%2 == 0: # even pixel case
                    for j in range(box_dim):
                        fid.seek(384 + 24576*(gain_high + 2*(star[0] + j - half_box)) + 12*(star[1] - half_box), 0)
                        bit_buffer[j] = np.fromfile(fid, dtype=np.uint8, count=ints_to_read)
                        
                    flux[i] = np.sum(conv_12to16(bit_buffer.flatten())[:box_dim-1])
                
                else: # odd pixel case
                    for j in range(box_dim):
                        fid.seek(384 + 24576*(gain_high + 2*(star[0] + j - half_box)) + 12*(star[1] - half_box - 1), 0)
                        bit_buffer[j] = np.fromfile(fid, dtype=np.uint8, count=ints_to_read)
                        
                    flux[i] = np.sum(conv_12to16(bit_buffer.flatten())[1:])
                    
            if timestamp:
                fid.seek(152,0)
                img_time = fid.read(29).decode('utf-8')
                return flux,img_time

                    
                    
    else: # odd half-box case
        with open(filename, 'rb') as fid:
            for i,star in enumerate(star_ind):
                if star[1]%2 == 1: # odd pixel case
                    for j in range(box_dim):
                        fid.seek(384 + 24576*(gain_high + 2*(star[0] + j - half_box)) + 12*(star[1] - half_box), 0)
                        bit_buffer[j] = np.fromfile(fid, dtype=np.uint8, count=ints_to_read)
                        
                    flux[i] = np.sum(conv_12to16(bit_buffer.flatten())[:box_dim-1])
                
                else: # even pixel case
                    for j in range(box_dim):
                        fid.seek(384 + 24576*(gain_high + 2*(star[0] + j - half_box)) + 12*(star[1] - half_box - 1), 0)
                        bit_buffer[j] = np.fromfile(fid, dtype=np.uint8, count=ints_to_read)
                        
                    flux[i] = np.sum(conv_12to16(bit_buffer.flatten())[1:])
                    
            if timestamp:
                fid.seek(152,0)
                img_time = fid.read(29).decode('utf-8')
                return flux,img_time

    return flux
            

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
    minSNR = 5               #median/stddev limit to look for detections
    minLightcurveLen = FramesperMin/4    #minimum length of lightcurve
    
    # reject stars that go out of frame to rapidly
    if len(light_curve) < minLightcurveLen:
        print(f"Light curve too short: star {num}")
        return -2, [], [], np.nan, np.nan, np.nan, np.nan, -2, np.nan  
    
    #TODO: what should the limits actually be?
    # reject tracking failures
    if abs(np.mean(light_curve[:FramesperMin]) - np.mean(light_curve[-FramesperMin:])) > np.std(light_curve[:FramesperMin]):
        print(f"Tracking failure: star {num}")
        return -2, [], [], np.nan, np.nan, np.nan, np.nan, -2, np.nan 
    
    # reject stars with SNR too low
    if np.median(light_curve)/np.std(light_curve) < minSNR:
        print(f"Signal to Noise too low: star {num}")
        return -2, [], [], np.nan, np.nan, np.nan, np.nan, -2, np.nan  

    #uncomment to save light curve of each star (doesn't look for dips)
    #return num, light_curve


    '''convolve light curve with ricker wavelet kernel'''
    #will throw error if try to normalize (sum of kernel too close to 0)
    conv = convolve_fft(light_curve, kernel, normalize_kernel=False)    #convolution of light curve with Ricker wavelet
    minLoc = np.argmin(conv)    #index of minimum value of convolution
    minVal = np.min(conv)          #minimum of convolution
    #TODO: The problems with Rocker wavelet smoothing, as currently implemented, are that:
    #1.The wavelet-smoothed light curve has correlated data points, so the original statistics are lost. 
    #In reality, the scatter has been diminished by about the square root of the number  of data points in the width of the wavelet 
    #(which is 6 or 7 data points, if I recall correctly)
    #2.The mean level has been scaled and/or shifted in a way that we (I, at least) don’t currently understand.
    
    #Sep 2022 we don't look for geometric dips any more - Roman A.
    
    # '''geometric dip detection (greater than 40%)'''
    # geoDip = 0.6    #threshold for geometric dip
    # norm_trunc_profile = light_curve/np.median(light_curve)  #normalize light curve
 
    # #if normalized profile at min value of convolution is less than the geometric dip threshold
    # if norm_trunc_profile[minLoc] < geoDip:
        
    #     #get frame number of dip
    #     critFrame = np.where(fluxProfile == light_curve[minLoc])[0]
    #     print (datetime.datetime.now(), "Detected >40% dip: frame", str(critFrame) + ", star", num)
        
    #     return critFrame[0], light_curve, 'geometric'


    '''look for diffraction dip'''
    KernelLength = len(kernel.array)    #number of elements in kernel array
    
    #check if dip is at least one kernel length from edge
    if KernelLength <= minLoc < len(light_curve) - KernelLength:
        
        edgeBuffer = 10     #how many elements from beginning/end of the convolution to exclude
        bkgZone = conv[edgeBuffer: -edgeBuffer]        #background
    
        #dipdetection = 3.75  #dip detection threshold ; Sep 2022 now it's an input parameter - Roman A.
        
    else:
        print(f"Event cutoff star: {num}")
        return -2, [], [], np.nan, np.nan, np.nan, np.nan, -2, np.nan  # reject events that are cut off at the start/end of time series

    #if minimum < background - 3.75*sigma
    # if minVal < np.mean(bkgZone) - dipdetection * np.std(bkgZone):  

    #     #get frame number of dip
    #     critFrame = np.where(fluxProfile == light_curve[minLoc])[0]
    #     print('found significant dip in star: ', num, ' at frame: ', critFrame[0])
        
    #     return critFrame[0], light_curve, 'diffraction'
        
    # else:
    #     return -1, [], ''  # reject events that do not pass dip detection
    
    lightcurve_std=np.std(light_curve)
    
    # event_std=np.std(conv)
    conv_bkg_mean=np.mean(bkgZone)
    #event_mean=np.mean(conv)
    
    significance=(conv_bkg_mean-minVal)/np.std(bkgZone) #significance of the event x*sigma
    
    
    if significance>=sigma_threshold:
        #get frame number of dip
        critFrame = np.where(fluxProfile == light_curve[minLoc])[0]
        print(f"Found significant dip in star: {num} at frame: {critFrame[0]}")
        
        return critFrame[0], light_curve, conv, lightcurve_std, np.mean(light_curve), np.std(bkgZone),conv_bkg_mean,minVal,significance
        
    else:
        return -1, light_curve, conv, np.nan, np.nan, np.nan, np.nan, np.nan, significance  # reject events that do not pass dip detection


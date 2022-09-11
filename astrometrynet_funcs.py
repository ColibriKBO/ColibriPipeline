#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:29:02 2022
Updated: Thurs. June 23, 2022

@author: Rachel A. Brown

Script to make use of astrometry.net API - see online docs
"""

from astroquery.astrometry_net import AstrometryNet

def getSolution(image_file, save_file, order):
    '''send request to solve image from astrometry.net
    input: path to the image file to submit, filepath to save the WCS solution header to, order of soln
    returns: WCS solution header'''

    #astrometry.net API
    ast = AstrometryNet()
    
    #key for astrometry.net account
    ast.api_key = 'vbeenheneoixdbpb'    #key for Rachel Brown's account (040822)
    wcs_header = ast.solve_from_image(image_file, crpix_center = True, tweak_order = order, force_image_upload=True)

    #save solution to file
    if not save_file.exists():
            wcs_header.tofile(save_file)
            
    return wcs_header

# def getLocalSolution(image_file, save_file, order):
#     # ast = AstrometryNet()

#     wcs_header = 

#     if not save_file.exists():
#         wcs_header.tofile(save_file)
#file = pathlib.Path('..', 'ColibriArchive', 'Red', '2022-04-06', 'high20210804_04.49.06.823_medstacked.fits')
#savefile = file.parent.joinpath('testsave.fits')

#getSolution(file, savefile, 3)



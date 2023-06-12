# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:29:02 2022
Updated: Thurs. June 23, 2022

@author: Rachel A. Brown

Script to make use of astrometry.net API - see online docs
"""

import time, subprocess, os
from astropy.io.fits import Header

#--------------------------------functions------------------------------------#

def getSolution(image_file, save_file, order):
    '''send request to solve image from astrometry.net
    input: path to the image file to submit, filepath to save the WCS solution header to, order of soln
    returns: WCS solution header'''
    from astroquery.astrometry_net import AstrometryNet
    #astrometry.net API
    ast = AstrometryNet()
    
    #key for astrometry.net account
    ast.api_key = 'vbeenheneoixdbpb'    #key for Rachel Brown's account (040822)
    wcs_header = ast.solve_from_image(image_file, crpix_center = True, tweak_order = order, force_image_upload=True)

    #save solution to file
    if not save_file.exists():
            wcs_header.tofile(save_file)
            
    return wcs_header

def getLocalSolution(image_file, save_file, order):
    """
    Astrometry.net must be installed locally to use this function. It installs under WSL.
    To use the local solution, you'll need to modify call to the function somewhat.
    This function will write the new fits file w/ plate solution to a file with the name save_file in the
    tmp directory on the d: drive.
    The function will return wcs_header. Alternatively, you could comment out those lines and read it from
    the pipeline.
    """
    try:
        # -D to specify write directory, -o to specify output base name, -N new-fits-filename
        print(image_file)
        # print(save_file.split(".")[0])
        print(save_file.split(".fits")[0])

        cwd = os.getcwd()
        os.chdir('d:\\')

        #p = subprocess.run('wsl time solve-field --no-plots -D /mnt/d/tmp -O -o ' + save_file.split(".")[0] + ' -N ' + save_file + ' -t ' + str(order) + ' --scale-units arcsecperpix --scale-low 2.2 --scale-high 2.6 ' + image_file)
        p = subprocess.run('wsl time solve-field --no-plots -D /mnt/d/tmp -O -o ' + save_file.split(".fits")[0] + ' -N ' + save_file + ' -t ' + str(order) + ' --scale-units arcsecperpix --scale-low 2.2 --scale-high 2.6 ' + image_file)
        
        os.chdir(cwd)
        print(os.getcwd())

        #wcs_header = Header.fromtextfile('d:\\tmp\\' + save_file.split('.')[0] + '.wcs')

        wcs_header = Header.fromfile('d:\\tmp\\' + save_file.split(".fits")[0] + '.wcs')

    except:
        pass

    return wcs_header


#-----------------------------------main--------------------------------------#

if __name__ == '__main__':
    wcs = getLocalSolution('/mnt/d/testmedstack.fits', 'test-newer.fits', 3)
    print(wcs)



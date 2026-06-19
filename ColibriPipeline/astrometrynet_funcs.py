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
    Obtain a WCS plate solution for a median-stacked FITS image.

    Strategy (in order):
      1. On Windows: delegate to WSL (solve-field installed inside WSL).
      2. On Linux/Mac: call solve-field natively if index files are present
         (detected by a successful solve run).
      3. Fallback: submit to the astrometry.net web API (requires internet).

    Args:
        image_file: path to the median-stacked FITS image (string).
        save_file:  basename of the output file (e.g. 'myfield_4th_wcs.fits').
        order:      SIP polynomial order (int).

    Returns:
        astropy.io.fits.Header with WCS keywords, or None on failure.
    """
    import platform

    wcs_header = None
    base_name = save_file.split(".fits")[0]

    print(image_file)
    print(base_name)

    # --- attempt 1: local solve-field ---
    try:
        if platform.system() == 'Windows':
            cwd = os.getcwd()
            os.chdir('d:\\')
            subprocess.run(
                'wsl time solve-field --no-plots -D /mnt/d/tmp -O'
                ' -o ' + base_name +
                ' -N ' + save_file +
                ' -t ' + str(order) +
                ' --scale-units arcsecperpix --scale-low 2.2 --scale-high 2.6 ' +
                image_file,
                shell=True
            )
            os.chdir(cwd)
            wcs_header = Header.fromfile('d:\\tmp\\' + base_name + '.wcs')

        else:
            # Linux / Mac: call solve-field directly
            tmp_dir = '/tmp/colibri_wcs'
            os.makedirs(tmp_dir, exist_ok=True)
            result = subprocess.run(
                [
                    'solve-field', '--no-plots',
                    '-D', tmp_dir, '-O',
                    '-o', base_name,
                    '-N', os.path.join(tmp_dir, save_file),
                    '-t', str(order),
                    '--scale-units', 'arcsecperpix',
                    '--scale-low', '2.2',
                    '--scale-high', '2.6',
                    image_file,
                ],
                timeout=300
            )
            wcs_file = os.path.join(tmp_dir, base_name + '.wcs')
            if os.path.exists(wcs_file):
                wcs_header = Header.fromfile(wcs_file)
            else:
                raise FileNotFoundError(f"solve-field did not produce {wcs_file} "
                                        "(index files may not be installed)")

    except Exception as e:
        print(f"WARNING: Local solve-field failed: {e}")
        print("Falling back to astrometry.net web API...")

    # --- attempt 2: web API fallback ---
    if wcs_header is None:
        # astroquery runs in this (possibly Windows) Python process and cannot
        # open a WSL mount path; translate /mnt/d/foo/bar -> D:\foo\bar.
        web_path = image_file
        if isinstance(web_path, str) and web_path.startswith("/mnt/") and len(web_path) > 6:
            web_path = web_path[5].upper() + ":\\" + web_path[7:].replace("/", "\\")
        try:
            from astroquery.astrometry_net import AstrometryNet
            ast = AstrometryNet()
            ast.api_key = 'vbeenheneoixdbpb'
            wcs_header = ast.solve_from_image(
                web_path,
                crpix_center=True,
                tweak_order=order,
                force_image_upload=True,
                scale_units='arcsecperpix',
                scale_lower=2.2,
                scale_upper=2.6,
            )
        except Exception as e:
            print(f"WARNING: Web API WCS solve also failed: {e}")

    return wcs_header


#-----------------------------------main--------------------------------------#

if __name__ == '__main__':
    wcs = getLocalSolution('/mnt/d/testmedstack.fits', 'test-newer.fits', 3)
    print(wcs)



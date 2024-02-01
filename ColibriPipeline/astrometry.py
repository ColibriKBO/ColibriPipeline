#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 14:14:16 2021
look at the output files from astrometry.net, compare star finding
@author: Rachel A. Brown
"""
#imports
import pathlib
from astropy.io import fits
import astropy.stats
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import getRAdec


def get_results(filename_base):
    '''read in files form astrometry.net output and put data into variables, also read in xy-radec file
    input: base name of the file that was analyzed
    returns: headers and data for astrometry.net files'''
    

    axy = fits.open(sorted(save_path.glob('*' + image_index + '_axy_' + order + '.fits'))[0])
    corr = fits.open(sorted(save_path.glob('*' + image_index + '_corr_' + order + '.fits'))[0])
    new_image = fits.open(sorted(save_path.glob('*' + image_index + '_new-image_' + order + '.fits'))[0])
    wcs = fits.open(sorted(save_path.glob('*' + image_index + '_wcs_' + order + '.fits'))[0])
    rdls = fits.open(sorted(save_path.glob('*' + image_index + '_rdls_' + order + '.fits'))[0])

    #axy.info()
    #corr.info()
    #new_image.info()
    #corr.info()
    #rdls.info()

    #WCS solution
    wcs_header = wcs[0].header
    
    #new image (with RA, dec info)
    newimage_header = new_image[0].header
    newimage_data = new_image[0].data

    #X, Y of extracted sources in image
    axy_header = axy[1].header
    axy_data = axy[1].data

    #RA, dec of extracted sources in image
    rdls_header = rdls[1].header
    rdls_data = rdls[1].data

    corr_header = corr[1].header
    corr_data = corr[1].data
    

    return wcs_header, newimage_header, newimage_data, axy_header, axy_data, rdls_header, rdls_data, corr_header, corr_data

def tableMatch(df1, df2, key1A, key1B, key2A, key2B, tolA, tolB):
    '''matches two data frames based on 2 sets of column label keys
    input: data frame to match to, data frame to match from, first col name to match from 1st df, 
    2nd col name to match from 1st df, 1st col name to match from 2nd df, 2nd col name from 2nd df,
    match tolerance for 1st col, match tolerance for 2nd col
    returns: dataframe based off the 2nd df with all rows that match within tolerance'''
    
    #list of matching indices
    matching_inds = []
    
    #loop through each row of 1st df
    for i in range(df1.shape[0]):
        
        #get values (A, B) to match to
        A = df1.loc[i, key1A]
        B = df1.loc[i, key1B]
        
        #get rows that match in 1st col between dfs within tolerance
        df = df2[(df2[key2A] < (A + tolA))]
        df = df[(df[key2A] > (A - tolA))]
        
        #get rows that match in 2nd col between dfs within tolerance
        df = df[(df[key2B] < (B + tolB))]
        df = df[(df[key2B] > (B - tolB))]
        
        matching_inds.append(df.index)
        
    return matching_inds
        
        
def match_XY(ast, colibri, SR):
    '''matches two data frames based on X, Y 
    input: pandas dataframe of star data with Gaia magnitudes {x, y, RA, dec, 3 magnitudes} 
    dataframe of star data with SNR {x, y, med, std, snr}
    returns: original star data frame with SNR data added where a match was found'''
    
    match_counter = 0
    
    #modified from Mike Mazur's code in 'MagLimitChecker.py'------------------
    
    #astrometry.net has bottom left corner at (1, 1) instead of (0, 0)
    colibri['X_shift'] = colibri['#X'] + 1
    colibri['Y_shift'] = colibri['Y'] + 1
    
    #get indices of matching rows
    match_inds = tableMatch(ast, colibri, 'field_x', 'field_y', 'X_shift', 'Y_shift', SR, SR)

    #loop through each row in astrometry table
    for i in range(ast.shape[0]):
        
        #make data frame of matching indices
        df = colibri.iloc[match_inds[i]]
        
        #sort matches by x difference
        df['diff_x'] = df['#X'] - ast.loc[i, 'field_x']
        
        if len(df.index) >= 1:
            df.sort_values(by='diff_x', ascending = True, inplace = True)
            df.reset_index(drop=True, inplace=True)
            
            #add new columns to astrometry table
            ast.loc[i,'colibri_X'] = df.loc[0]['#X']
            ast.loc[i,'colibri_Y'] = df.loc[0]['Y']
            ast.loc[i,'colibri_RA'] = df.loc[0]['RA']
            ast.loc[i, 'colibri_dec'] = df.loc[0]['Dec']
    
    #calculate differences
    ast['ra_diff'] = (ast['colibri_RA'] - ast['index_ra'])*np.cos(np.radians(ast['colibri_dec']))
    ast['dec_diff'] = ast['colibri_dec'] - ast['index_dec']
    ast['x_diff'] = ast['colibri_X'] + 1 - ast['index_x']
    ast['y_diff'] = ast['colibri_Y'] + 1 - ast['index_y']
             
    print('Number of SNR matches: ', match_counter)
    return ast
    

def plotAstVSColibriXY_diff(matched):
    '''makes plot of X difference vs Y differences for Colibri and ast.net transformed index
    input: dataframe containing columns 'x_diff' and 'y_diff'
    returns: displays and saves plot of xdifferences and y differences, prints out mean and stddev'''
    
    #calculate sigma clipped mean difference for X and Y
    clip_sigma = 2
    clipped_xdiff = astropy.stats.sigma_clip(matched['x_diff'], sigma = clip_sigma, cenfunc = 'mean')
    clipped_ydiff = astropy.stats.sigma_clip(matched['y_diff'], sigma = clip_sigma, cenfunc = 'mean')
    x_mean = np.mean(clipped_xdiff)
    x_std = np.std(clipped_xdiff)
    y_mean = np.mean(clipped_ydiff)
    y_std = np.std(clipped_ydiff)

    
    #make figure
    plt.figure(figsize=(10, 10), dpi=100)
    
    plt.scatter(matched['x_diff'], matched['y_diff'], s = 8)
    plt.scatter(0, 0, color = 'black', marker = '+', s = 50)
    
    plt.text(-6, 6, 'X mean difference: %.3f +/- %.3f px' %(x_mean, x_std), fontsize = 18)
    plt.text(-6, 5, 'Y mean difference: %.3f +/- %.3f px' %(y_mean, y_std), fontsize = 18)
    plt.text(-6, 4, '(%.0f$\sigma$ clipped)' %(clip_sigma), fontsize = 18)

    plt.title(order + ' order, ' + telescope + ', ' + field_name + ', ' + str(obs_date) + ', ' + str(obs_time), fontsize = 20)

    plt.xlim(-7, 7)
    plt.ylim(-7, 7)
    
    plt.xlabel('(SEP Measured X - Expected Index X) [px]', fontsize = 18)
    plt.ylabel('(SEP Measured Y - Expected Index Y) [px]', fontsize = 18)
    plt.yticks(fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.savefig(save_path.joinpath('astVSColibriXY_diff_' + order + '_' + telescope + '_' + field_name + '_' + str(obs_date) + '_' + str(obs_time) + '.png'))
    plt.show()
    plt.close()
    
def plotAstFieldvsIndexXY_diff(corr):
    '''makes plot of (X, Y) differences for ast.net measured ('field') and expected ('index') coords, calculates mean and stddev
    input: output correlation table from ast.net containing both field and index star columns
    returns: displays and saves plot of x differences and y differences, prints out mean and stddev'''
    
    #calculate difference ('field' = measured, 'index' = expected from transformed catalog)
    corr_Xdiff = corr['field_x'] - corr['index_x']
    corr_Ydiff = corr['field_y'] - corr['index_y']
    
    #calculate mean/stddev of both
    x_mean = np.mean(corr_Xdiff)
    x_std = np.std(corr_Xdiff)
    y_mean = np.mean(corr_Ydiff)
    y_std = np.std(corr_Ydiff)

    #plot figure
    plt.figure(figsize=(10, 10), dpi=100)
    
    plt.scatter(corr_Xdiff, corr_Ydiff, s = 8)
    plt.scatter(0, 0, color = 'black', marker = '+', s = 50)
    
    plt.text(-6, 6, 'X mean difference: %.3f +/- %.3f px' %(x_mean, x_std), fontsize = 18)
    plt.text(-6, 5, 'Y mean difference: %.3f +/- %.3f px' %(y_mean, y_std), fontsize = 18)

    plt.title('Astrometry.net ' + order + ' order, ' + telescope + ', ' + field_name + ', ' + str(obs_date) + ', ' + str(obs_time), fontsize = 20)

    plt.xlim(-7, 7)
    plt.ylim(-7, 7)
    
    plt.xlabel('(Measured X - Expected X) [px]', fontsize = 18)
    plt.ylabel('(Measured Y - Expected Y) [px]', fontsize = 18)
    plt.yticks(fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.savefig(save_path.joinpath('ast_X-Y_diff_' + order + '_' + telescope + '_' + field_name + '_' + str(obs_date) + '_' + str(obs_time) + '.png'))
    plt.show()
    plt.close()
    
def plotAstFieldVSColibriXY_diff(matched):
    '''makes plot of X difference vs Y differences, calculates mean and stddev
    input: dataframe containing columns 'x_diff' and 'y_diff'
    returns: displays and saves plot of xdifferences and y differences, prints out mean and stddev'''
    
   # x_mean = np.mean(matched['x_diff'])
    Xdiff = matched['field_x'] - (matched['colibri_X'] + 1)
    Ydiff = matched['field_y'] - (matched['colibri_Y'] + 1)
    
    x_mean = np.mean(Xdiff)
    x_std = np.std(Xdiff)
    y_mean = np.mean(Ydiff)
    y_std = np.std(Ydiff)

    plt.figure(figsize=(10, 10), dpi=100)
    
    plt.scatter(Xdiff, Ydiff, s = 8)
    plt.scatter(0, 0, color = 'black', marker = '+', s = 50)
    
    plt.text(-6, 6, 'X mean difference: %.3f +/- %.3f px' %(x_mean, x_std), fontsize = 18)
    plt.text(-6, 5, 'Y mean difference: %.3f +/- %.3f px' %(y_mean, y_std), fontsize = 18)

    plt.title('Detection comparison ' + order + ' order, ' + telescope + ', ' + field_name + ', ' + str(obs_date) + ', ' + str(obs_time), fontsize = 20)

    plt.xlim(-7, 7)
    plt.ylim(-7, 7)
    
    plt.xlabel('(Ast.net Measured X - Colibri SEP X) [px]', fontsize = 18)
    plt.ylabel('(Ast.net Measured Y - Colibri SEP Y) [px]', fontsize = 18)
    plt.yticks(fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.savefig(save_path.joinpath('sep_ast_X-Y_diff_' + order + '_' + telescope + '_' + field_name + '_' + str(obs_date) + '_' + str(obs_time) + '.png'))
    plt.show()
    plt.close()

def plotAstVSColibriRAdec_diff(matched):
    '''makes plot of RA difference vs dec differences for Colibri transformed and ast.net index
    input: dataframe containing columns 'ra_diff' and 'dec_diff'
    returns: displays and saves plot of xdifferences and y differences, prints out mean and stddev'''
    
    #calculate sigma clipped mean difference for RA and dec
    clip_sigma = 2
    clipped_radiff = astropy.stats.sigma_clip(matched['ra_diff'], sigma = clip_sigma, cenfunc = 'mean')
    clipped_decdiff = astropy.stats.sigma_clip(matched['dec_diff'], sigma = clip_sigma, cenfunc = 'mean')
    
    ra_mean = np.mean(clipped_radiff)*3600
    ra_std = np.std(clipped_radiff)*3600
    dec_mean = np.mean(clipped_decdiff)*3600
    dec_std = np.std(clipped_decdiff)*3600

    #make figure
    plt.figure(figsize=(10, 10), dpi=100)
    
    plt.scatter(matched['ra_diff'].multiply(3600), matched['dec_diff'].multiply(3600), s = 8)
    plt.scatter(0, 0, color = 'black', marker = '+', s = 50)
    
    plt.text(-14, 14, 'RA mean difference: %.3f +/- %.3f "' %(ra_mean, ra_std), fontsize = 18)
    plt.text(-14, 12, 'Dec mean difference: %.3f +/- %.3f "' %(dec_mean, dec_std), fontsize = 18)
    plt.text(-14, 10, '(%.0f$\sigma$ clipped)' %(clip_sigma), fontsize = 18)

    #plt.xlim(-0.004*3600, 0.004*3600)
    plt.xlim(-7*2.4, 7*2.4)
    #plt.ylim(-0.004*3600, 0.004*3600)
    plt.ylim(-7*2.4, 7*2.4)

    plt.title(order + ' order, ' + telescope + ', ' + field_name + ', ' + str(obs_date) + ', ' + str(obs_time), fontsize = 20)
    plt.xlabel('(SEP Transformed RA - Catalog RA) [arcsec]', fontsize = 18)
    plt.ylabel('(SEP Transformed Dec - Catalog Dec) [arcsec]', fontsize = 18)
    plt.yticks(fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.savefig(save_path.joinpath('astVSColibriRAdec_diff_' + order + '_' + telescope + '_' + field_name + '_' + str(obs_date) + '_' + str(obs_time) + '.png'))
    plt.show()
    plt.close()
    
def plotAstFieldvsIndexRAdec_diff(corr):
    '''makes plot of (RA, dec) differences for ast.net transformed measured ('field') and catalog ('index') coords, calculates mean and stddev
    input: output correlation table from ast.net containing both field and index star columns
    returns: displays and saves plot of RA differences and dec differences, prints out mean and stddev'''
    
    #calculate difference ('field' = transformed measured, 'index' = expected from catalog)
    corr_RAdiff = corr['field_ra'] - corr['index_ra']
    corr_Decdiff = corr['field_dec'] - corr['index_dec']
    
    #calcaulte mean/stddev in arcsecs
    ra_mean = np.mean(corr_RAdiff)*3600
    ra_std = np.std(corr_RAdiff)*3600
    dec_mean = np.mean(corr_Decdiff)*3600
    dec_std = np.std(corr_Decdiff)*3600

    #make figure 
    plt.figure(figsize=(10, 10), dpi=100)
    
    plt.scatter(corr_RAdiff.multiply(3600), corr_Decdiff.multiply(3600), s = 8)
    plt.scatter(0, 0, color = 'black', marker = '+', s = 50)
    
    plt.text(-14, 14, 'RA mean difference: %.3f +/- %.3f "' %(ra_mean, ra_std), fontsize = 18)
    plt.text(-14, 12, 'Dec mean difference: %.3f +/- %.3f "' %(dec_mean, dec_std), fontsize = 18)

    #plt.xlim(-0.004*3600, 0.004*3600)
    plt.xlim(-7*2.4, 7*2.4)
    #plt.ylim(-0.004*3600, 0.004*3600)
    plt.ylim(-7*2.4, 7*2.4)

    plt.title('Astrometry.net ' + order + ' order, ' + telescope + ', ' + field_name + ', ' + str(obs_date) + ', ' + str(obs_time), fontsize = 20)
    plt.xlabel('(Measured RA - Catalog RA) [arcsec]', fontsize = 18)
    plt.ylabel('(Measured Dec - Catalog Dec) [arcsec]', fontsize = 18)
    plt.yticks(fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.savefig(save_path.joinpath('ast_RA-dec_diff_' + order + '_' + telescope + '_' + field_name + '_' + str(obs_date) + '_' + str(obs_time) + '.png'))
    plt.show()
    plt.close()
    


def plotXYArrows(matched):
    '''makes plot of (X, Y) differences between Colibri SEP and ast.net index as arrows on XY plane
    input: dataframe containing columns of X, Y corrds, X, Y differences
    returns: displays and saves arrow plot'''
    
    arrow_scale = 0.04
    plt.figure(figsize=(10, 10), dpi=100)
    
    plt.quiver(matched['colibri_X'], matched['colibri_Y'], matched['x_diff'], matched['y_diff'], 
               units = 'xy', scale = arrow_scale, scale_units = 'xy')

    plt.xlim(-100, 2100)
    plt.ylim(-100, 2100)
    
    plt.text(-50, 2000, 'Arrow scale: %.2f ' %(arrow_scale), fontsize = 18)

    plt.title(order + ' order, ' + telescope + ', ' + field_name + ', ' + str(obs_date) + ', ' + str(obs_time), fontsize = 20)

    plt.xlabel('SEP Measured X [px]', fontsize = 18)
    plt.ylabel('SEP Measured Y [px]', fontsize = 18)
    plt.yticks(fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.savefig(save_path.joinpath('X-Y_diff_arrows_' + order + '_' + telescope + '_' + field_name + '_' + str(obs_date) + '_' + str(obs_time) + '.png'))
    plt.show()
    plt.close()
    

def plotRAdecArrows(matched):
    '''makes plot of (RA, dec) differences between Colibri transformed SEP and ast.net index as arrows on celestial plane
    input: dataframe containing columns of RA, dec coords, RA, dec differences
    returns: displays and saves arrow plot'''
    
    arrow_scale = 0.04
    plt.figure(figsize=(10, 10), dpi=100)
        
    plt.quiver(matched['colibri_RA'], matched['colibri_dec'], matched['ra_diff'], matched['dec_diff'], 
               units = 'xy', scale = arrow_scale, scale_units = 'xy')
    
    plt.text(273.08, -19.25, 'Arrow scale: %.2f ' %(arrow_scale), fontsize = 18)

    plt.xlim(273.05, 274.115)
    plt.ylim(-18.221, -19.286)

    plt.title(order + ' order, ' + telescope + ', ' + field_name + ', ' + str(obs_date) + ', ' + str(obs_time), fontsize = 20)
    plt.xlabel('SEP Measured RA [degrees]', fontsize = 18)
    plt.ylabel('SEP Measured Dec [degrees]', fontsize = 18)
    plt.yticks(fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.savefig(save_path.joinpath('RA-dec_diff_arrows_' + order + '_' + telescope + '_' + field_name + '_' + str(obs_date) + '_' + str(obs_time) + '.png'))
    plt.show()
    plt.close()
    
def plotImage(image_data):
    '''make plot of colibri image data in grayscale with colorbar, including vectors
    input: filename
    returns: saves plot to disk'''
        
    arrow_scale = 0.04
        
    plt.figure(figsize = (10, 10), dpi = 100)
    plt.imshow(image_data, vmin = 0, vmax = 20)
        
    plt.gca().invert_yaxis()
    plt.colorbar().set_label('Pixel Value', fontsize = 18)
        
    plt.quiver(matched['colibri_X'], matched['colibri_Y'], matched['x_diff'], matched['y_diff'], 
                   units = 'xy', scale = arrow_scale, scale_units = 'xy')
        
    plt.text(15, 20, 'Arrow scale: %.2f ' %(arrow_scale), fontsize = 18)

    plt.title(order + ' order, ' + telescope + ', ' + field_name + ', ' + str(obs_date) + ', ' + str(obs_time), fontsize = 20)

    plt.xlabel('SEP Measured X [px]', fontsize = 18)
    plt.ylabel('SEP Measured Y [px]', fontsize = 18)
    plt.yticks(fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.gca().set_aspect('equal', adjustable='box')
        
    plt.savefig(save_path.joinpath('XY_arrows-image_' + order + '_' + telescope + '_' + field_name + '_' + str(obs_date) + '_' + str(obs_time) + '.png'))
    plt.show()
    plt.close()   
    

#def compareFlux()
#translate .npy file into .txt for plotting in ds9
def npy2text():
    col_savefile = save_path.joinpath('colibri_stars.txt')
    with open(col_savefile, 'w') as filehandle:
    
        for index, line in colibri_stars.iterrows():
            filehandle.write('%s %f %f %f\n' %('circle', line[0], line[1], 6.))
            #filehandle.write('%f %f\n' %(line[0], line[1]))
        
        
        corr_savefile = save_path.joinpath('corr_stars.txt')
        with open(corr_savefile, 'w') as filehandle:
    
            for i in range(len(corr_data)):
                filehandle.write('%s %f %f %f\n' %('circle', corr_data['index_x'][i], corr_data['index_y'][i], 6.))
                #filehandle.write('%f %f\n' %(line[0], line[1]))
        
        axy_savefile = save_path.joinpath('axy_stars.txt')
        with open(axy_savefile, 'w') as filehandle:
    
            for i in range(len(axy_data)):
                filehandle.write('%s %f %f %f\n' %('circle', axy_data['X'][i], axy_data['Y'][i], 6.))
                #filehandle.write('%f %f\n' %(line[0], line[1]))

def matchColibriAstnet(matched, starTable, SR):
    '''matches astrometry table with colibri star table
    input: pandas dataframe of star data with Gaia magnitudes {x, y, RA, dec, 3 magnitudes} 
    dataframe of star data with SNR {x, y, med, std, snr}
    returns: original star data frame with SNR data added where a match was found'''
    
    match_counter = 0
    
    #modified from Mike Mazur's code in 'MagLimitChecker.py'------------------
    
    #get indices of matching rows
    match_inds = tableMatch(matched, starTable, 'colibri_X', 'colibri_Y', 'X', 'Y', SR, SR)

    #loop through each row in astrometry table
    for i in range(matched.shape[0]):
        
        #make data frame of matching indices
        df = starTable.iloc[match_inds[i]]
        
        #sort matches by x difference
        df['diff_x'] = df['X'] - matched.loc[i, 'colibri_X']
        
        if len(df.index) >= 1:
            df.sort_values(by='diff_x', ascending = True, inplace = True)
            df.reset_index(drop=True, inplace=True)
            
            #add new columns to astrometry table
            matched.loc[i,'Gaia_RA'] = df.loc[0]['Gaia_RA']
            matched.loc[i,'Gaia_dec'] = df.loc[0]['Gaia_dec']
            matched.loc[i,'colibri_med'] = df.loc[0]['med']
            matched.loc[i,'colibri_std'] = df.loc[0]['std']
            matched.loc[i,'colibri_SNR'] = df.loc[0]['SNR']
    
    #calculate differences
    matched['Gaia_ra_diff'] = (matched['Gaia_RA'] - matched['index_ra'])*np.cos(np.radians(matched['Gaia_dec']))
    matched['Gaia_dec_diff'] = matched['Gaia_dec'] - matched['index_dec']

    print('Number of SNR matches: ', match_counter)
    return matched

def plotGaiaVSindexCoords(matched):
    '''makes plot of Gaia catalog coords vs ast.net catalog coords
    input: dataframe containing columns of Gaia RA/dec and ast.net index RA/dec
    returns: displays and saves plot of RA differences and dec differences'''
    
    #get RA & dec differences
    RAdiff = (matched['Gaia_RA'] - matched['index_ra'])*np.cos(np.radians(matched['Gaia_dec']))
    Decdiff = matched['Gaia_dec'] - matched['index_dec']
    
    #calculate means and stddev for each [in arcsec]
    ra_mean = np.mean(RAdiff)*3600
    ra_std = np.std(RAdiff)*3600
    dec_mean = np.mean(Decdiff)*3600
    dec_std = np.std(Decdiff)*3600

    #plot figure
    plt.figure(figsize=(10, 10), dpi=100)
    
    plt.scatter(RAdiff.multiply(3600), Decdiff.multiply(3600), s = 8)
    plt.scatter(0, 0, color = 'black', marker = '+', s = 50)
    
    plt.text(-14, 14, 'RA mean difference: %.3f +/- %.3f "' %(ra_mean, ra_std), fontsize = 18)
    plt.text(-14, 12, 'Dec mean difference: %.3f +/- %.3f "' %(dec_mean, dec_std), fontsize = 18)

    #plt.xlim(-0.004*3600, 0.004*3600)
    plt.xlim(-7*2.4, 7*2.4)
    #plt.ylim(-0.004*3600, 0.004*3600)
    plt.ylim(-7*2.4, 7*2.4)

    plt.title('Gaia vs Ast.net ' + order + ' order, ' + telescope + ', ' + field_name + ', ' + str(obs_date) + ', ' + str(obs_time), fontsize = 20)
    plt.xlabel('(Gaia RA - Catalog RA) [arcsec]', fontsize = 18)
    plt.ylabel('(Gaia Dec - Catalog Dec) [arcsec]', fontsize = 18)
    plt.yticks(fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.savefig(save_path.joinpath('gaiaVSindex_diff_' + order + '_' + telescope + '_' + field_name + '_' + str(obs_date) + '_' + str(obs_time) + '.png'))
    plt.show()
    plt.close()  
    
def plotAstVSColibriSNR(matched):
    '''make plot comparing SNR from ast.net and SEP
    input: dataframe containing columns of Flux, Background from ast.net, and SNR from Colibri
    returns: saves and displays plot of ast.net SNR vs Colibri SNR'''
    
    #make figure
    plt.figure(figsize = (10,10), dpi = 100)
    
    plt.scatter(matched['FLUX']/matched['BACKGROUND'], matched['colibri_SNR'], s = 8)
    
    plt.title('SNR Compare ' + order + ' order, ' + telescope + ', ' + field_name + ', ' + str(obs_date) + ', ' + str(obs_time), fontsize = 20)
    plt.xlabel('Astrometry.net measured SNR (flux/background)', fontsize = 18)
    plt.ylabel('Colibri SEP measured SNR (median/stddev)', fontsize = 18)
    
    plt.yticks(fontsize = 15)
    plt.xticks(fontsize=15)
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.savefig(save_path.joinpath('flux_diff_' + order + '_' + telescope + '_' + field_name + '_' + str(obs_date) + '_' + str(obs_time) + '.png'))
    plt.show()
    plt.close()

'''----------------------------------start script------------------------------------------'''
'''----------------------------------------------------------------------------------------'''
'''----------------------------------------------------------------------------------------'''

if __name__ == '__main__':

    '''--------------observation & solution info----------------'''
    obs_date = datetime.date(2022, 5, 18)           #date of observation
    obs_time = datetime.time(5, 41, 54)             #time of observation (to the second)
    image_index = 'medstacked'                      #index of image to use
    order = '4th'                                   #tweak order of solution polynomial

    telescope = 'Green'                             #telescope identifier
    field_name = 'field1'                           #name of field observed


    '''-------set up paths to files----------------------------'''

    #path to base directory
    base_path = pathlib.Path('/', 'home', 'rbrown', 'Documents', 'Colibri')

    #path to directory that holds images
    #data_path = base_path.joinpath(telescope, 'ColibriData', str(obs_date).replace('-', ''))    #path that holds data
    data_path = base_path.joinpath(telescope, 'Elginfield' + telescope, str(obs_date).replace('-','') + '_diagnostics', 'Sensitivity') 

    #get exact name of desired minute directory
    subdirs = [f.name for f in data_path.iterdir() if f.is_dir()]                   #all minutes in night directory
    minute_dir = [f for f in subdirs if str(obs_time).replace(':', '.') in f][0]    #minute we're interested in

    #path to save output files to
    #save_path = base_path.joinpath(telescope, 'ColibriArchive', str(obs_date), minute_dir)    #path to save outputs in
    save_path = data_path.joinpath(minute_dir)

    matchTol = 0.1            #matching tolerance [px]


    '''-----------make star transform file---------------------'''

    #get coordinate solution from astrometry.net and apply to list of X,Y coords
    coords = getRAdec.getRAdecfromFile(sorted(save_path.glob('*' + image_index + '_new-image_' + order + '.fits'))[0],                            
                            sorted(save_path.glob('*.npy'))[0], 
                            save_path.joinpath(minute_dir + '_' + order + '_xy_RD.txt'))

    #read in list of converted coordinates
    colibri_stars = pd.read_csv(save_path.joinpath(minute_dir + '_' + order + '_xy_RD.txt'), delim_whitespace = True, header = 4)


    '''----------read in astrometry.net results---------------'''

    #output files from astrometry.net
    wcs_header, newimage_header, newimage_data, axy_header, axy_data, rdls_header, rdls_data, corr_header, corr_data = get_results(image_index)

    #make dataframe of correlating stars between astrometry.net's finds and catalogs
    corr_df = pd.DataFrame(np.array(corr_data).byteswap().newbyteorder())


    '''------match astrometry stars with colibri stars--------'''

    matched = match_XY(corr_df, colibri_stars, 1)


    '''------read in star table from sensitivity tests to get Gaia data -------'''
    #comment out below if sensitivity tests haven't been done

    #table with detected stars and their sky coords + Gaia magnitudes
    starTableFile = sorted(save_path.glob('starTable*'+ order + '*.txt'))[0]
    starTable = pd.read_csv(starTableFile, delim_whitespace = True)         #load into dataframe

    #match this table with astrometry.net results
    matched = matchColibriAstnet(matched, starTable, matchTol)


    '''-------make plots----------------'''

    #test astrometry.net solution with itself
    plotAstFieldvsIndexXY_diff(corr_df)     #plot (X, Y) differences between ast.net measured coords and expected coords
    plotAstFieldvsIndexRAdec_diff(corr_df)  #plot (RA, dec) differences between ast.net transformed measured coords and catalog coords

    #test astrometry.net solution with Colibri SEP
    plotAstFieldVSColibriXY_diff(matched)       #plot diff between ast.net measured coords and Colibri SEP measured coords (XY)
    plotAstVSColibriSNR(matched)            #plot ast.net SNR vs Colibri SNR
    plotAstVSColibriXY_diff(matched)        #plot diff between ast.net index and Colibri (X, Y)
    plotAstVSColibriRAdec_diff(matched)     #plot diff between ast.net index and Colibri (RA, dec)
    plotXYArrows(matched)                   #plot star (X, Y) coords with vectors showing magnitude and direction of difference between Colibri and ast.index
    plotRAdecArrows(matched)                #plot star (RA, dec) coords with vectors showing magnitude and direction of difference between Colibri and ast.index

    #test astrometry.net solution with Gaia catalog
    plotGaiaVSindexCoords(matched)          #plot diff between Gaia and ast.net index (RA, dec)

    #plot image with colourbar
    plotImage(newimage_data)                #plot image with colourbar, showing X,Y differences as vectors

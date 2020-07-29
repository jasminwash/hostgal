#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 10:14:32 2018

@author: Dartoon

For analysis the data
"""
import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from matplotlib.colors import LogNorm
import matplotlib
import copy
import sys
sys.path.insert(0, '../../../py_tools/')
from est_bkg import est_bkg
from cut_image import cut_image
my_cmap = copy.copy(matplotlib.cm.get_cmap('gist_heat')) # copy the default cmap
my_cmap.set_bad('black')
import glob

def gen_fit_id(folder_name, ra, dec, file_name,cut_frame=120, subbkl='n'):
    '''
    Generate the ingredients for the fitting, including 
        1. The images for QSO, errormap and PSF.
            Note: QSO and errormap is cut out based on it's loc in file 'sdss_hsc_match_zgt0.2_lt0.5.asc'
        2. Calculate the pixel scale
        3. From FLUXMAG0 calulate the zeropoint = 2.5log(FLUXMAG0)
    
    Parameter
    --------
        seq: The seq of the ID, seq=1 means the image file is "2-*.fits"
        subbkl: if test and sub the background light, 'y' means yes.
    Return
    --------
        QSO_im, err_map, PSF, pix_scale, zp, qso_ID, qso_fr_center
    '''
    fitsFile = pyfits.open(folder_name+file_name)
    sci_data= fitsFile[1].data
    err_data= fitsFile[3].data ** 0.5
    subbkl = subbkl #raw_input("do you want to test and sub the background light for the image?:\ninpt y to esti and sub bkg_light, else no.\n")
    if subbkl == 'y':
        bkg = est_bkg(sci_data)
        sci_data = sci_data - bkg
    
    file_header0 = fitsFile[0].header
#    file_header1 = fitsFile[1].header
    #==============================================================================
    # Cut out the QSO image based on the RA DEC position
    #==============================================================================
    wcs = WCS(fitsFile[1].header)    
#    data_info_file= '../sdss_qsos.asc'
#    with open('{0}'.format(data_info_file)) as f:
#        content = f.readlines()
#    lines = [x.strip() for x in content] 
#    inf_seq = [i for i in range(len(lines)) if lines[i].split( )[0]==file_name.split('_')[0]][0]
#    data_info = lines[inf_seq].split( )
#    print 'inf_seq', inf_seq
#   qso_ID = data_info[0]
#    qso_RA = float(data_info[1])
#    qso_DEC = float(data_info[2])

    qso_RA=float(ra)
    qso_DEC=float(dec)
    QSO_x, QSO_y = wcs.all_world2pix([qso_RA],[qso_DEC],1)  #Read the corresponding pixel
    load_RA, load_DEC = wcs.all_pix2world([QSO_x],[QSO_y],1)
#    print "the pixel of the QSO:", QSO_x, QSO_y 
    center = np.array([int(QSO_x), int(QSO_y)])
    QSO_im = cut_image(sci_data, center, cut_frame)
    err_map = cut_image(err_data, center, cut_frame)
#    print "PSF_name", PSF_name
    PSF = pyfits.getdata(folder_name+file_name.split('.fits')[0]+'_psf.fits')
    #==============================================================================
    # Derive the pixel scale and zeropoint
    #==============================================================================
    diff_RA_DEC = wcs.all_pix2world([100,100],[100,200],1)
    diff_scale = np.sqrt((diff_RA_DEC[0][1]-diff_RA_DEC[0][0])**2 + (diff_RA_DEC[1][1]-diff_RA_DEC[1][0])**2)
    pix_scale = diff_scale/(100.) * 3600
    FLUXMAG0 = file_header0['FLUXMAG0']
    zp =  2.5 * np.log10(FLUXMAG0)   # This is something Xuheng can't make sure.
    return QSO_im, err_map, PSF, pix_scale, zp, center, [load_RA, load_DEC]

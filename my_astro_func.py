#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:27:49 2019

@author: Dartoon

my function to faster analysis the pyfits
"""
import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from matplotlib.colors import LogNorm
from mask_objects import find_loc_max
from cut_image import cut_image

def read_pixel_scale(filename,frame=0):
    fitsFile = pyfits.open(filename)
    wcs = WCS(fitsFile[frame].header)
    diff_RA_DEC = wcs.all_pix2world([0,0],[0,1],1)
    diff_scale = np.sqrt((diff_RA_DEC[0][1]-diff_RA_DEC[0][0])**2 + (diff_RA_DEC[1][1]-diff_RA_DEC[1][0])**2)
    pix_scale = diff_scale * 3600
    return pix_scale

def read_fits_exp(filename,frame=0):
    fitsFile = pyfits.open(filename)
    file_header0 = fitsFile[frame].header
    return file_header0['EXPTIME']

def plt_fits(img):
    plt.imshow(img, norm=LogNorm(),origin='low')   
    plt.colorbar()
    plt.show()
    
def auto_cut_PSFs(img, radius=100, view=True):
    PSFx, PSFy =find_loc_max(img)
    if view ==True:
        for i in range(len(PSFx)):
            cut_img = cut_image(img, [PSFx[i], PSFy[i]], radius=radius)
            if True not in np.isnan(cut_img) and np.max(cut_img) == cut_img[len(cut_img)/2][len(cut_img)/2]:
                print("plot for position: [{0}, {1}]".format(PSFx[i], PSFy[i]), "idx:", i)
                cut_img[np.isnan(cut_img)] = 0
                print("total flux:", cut_img.sum())
                plt_fits(cut_img)
                print("================")
    return  PSFx, PSFy  
    
    
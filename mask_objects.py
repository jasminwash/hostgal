#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 15:01:17 2018

@author: Dartoon
"""
import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from photutils import detect_threshold
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from photutils import detect_sources,deblend_sources
from matplotlib.colors import LogNorm
from photutils import source_properties

import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters

def detect_obj(img, snr=2.8, exp_sz= 1.2, pltshow=1):
    threshold = detect_threshold(img, nsigma=snr)
    center_img = len(img)/2
    sigma = 3.0 * gaussian_fwhm_to_sigma# FWHM = 3.
    kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
    kernel.normalize()
    npixels = 15
    segm = detect_sources(img, threshold, npixels=npixels, filter_kernel=kernel)
#    print(segm)
    if segm==None: return "bad!",1
    segm_deblend = deblend_sources(img, segm, npixels=npixels,
                                    filter_kernel=kernel, nlevels=25,
                                    contrast=0.001)
    #Number of objects segm_deblend.data.max()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 10))
    import copy, matplotlib
    my_cmap = copy.copy(matplotlib.cm.get_cmap('gist_heat')) # copy the default cmap
    my_cmap.set_bad('black')
    vmin = 1.0e-3
    vmax = 2.1 
    ax1.imshow(img, origin='lower', cmap=my_cmap, norm=LogNorm(), vmin=vmin, vmax=vmax)
    ax1.set_title('Data')
    ax2.imshow(segm_deblend, origin='lower', cmap=segm_deblend.cmap(random_state=12345))
    ax2.set_title('Segmentation Image')
    if pltshow == 0:
        plt.close()
    else:
        plt.show()
    
    columns = ['id', 'xcentroid', 'ycentroid', 'source_sum', 'area']
    cat = source_properties(img, segm_deblend)
    tbl = cat.to_table(columns=columns)
    tbl['xcentroid'].info.format = '.2f'  # optional format
    tbl['ycentroid'].info.format = '.2f'
    print(tbl)
    cat = source_properties(img, segm_deblend)
    objs = []
    for obj in cat:
        position = (obj.xcentroid.value-center_img, obj.ycentroid.value-center_img)
        a_o = obj.semimajor_axis_sigma.value
        b_o = obj.semiminor_axis_sigma.value
        Re = (a_o + b_o) /2.
        q = 1 - obj.ellipticity.to_value()
        objs.append((position,Re,q))
    dis_sq = [np.sqrt((objs[i][0][0])**2+(objs[i][0][1])**2) for i in range(len(objs))]
    dis_sq = np.array(dis_sq)
    c_index= np.where(dis_sq == dis_sq.min())[0][0]
    return objs, c_index
    

def mask_obj(img, snr=3.0, exp_sz= 1.2, pltshow = 0, npixels=20, return_deblend = False):
    threshold = detect_threshold(img, nsigma=snr)
    sigma = 3.0 * gaussian_fwhm_to_sigma# FWHM = 3.
    kernel = Gaussian2DKernel(sigma, x_size=int(npixels/4), y_size=int(npixels/4))
    kernel.normalize()
    segm = detect_sources(img, threshold, npixels=npixels, filter_kernel=kernel)
    npixels = npixels
    segm_deblend = deblend_sources(img, segm, npixels=npixels,
                                    filter_kernel=kernel, nlevels=15,
                                    contrast=0.001)
    columns = ['id', 'xcentroid', 'ycentroid', 'source_sum', 'area']
    cat = source_properties(img, segm_deblend)
    tbl = cat.to_table(columns=columns)
    tbl['xcentroid'].info.format = '.2f'  # optional format
    tbl['ycentroid'].info.format = '.2f'
    print(tbl)
    #Number of objects segm_deblend.data.max()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 10))
    import copy, matplotlib
    my_cmap = copy.copy(matplotlib.cm.get_cmap('gist_heat')) # copy the default cmap
    my_cmap.set_bad('black')
    vmin = 1.e-3
    vmax = 2.1 
    ax1.imshow(img, origin='lower', cmap=my_cmap, norm=LogNorm(), vmin=vmin, vmax=vmax)
    ax1.set_title('Data')
    ax2.imshow(segm_deblend, origin='lower', cmap=segm_deblend.cmap(random_state=12345))
    for obj in cat:
        ax2.text(obj.xcentroid.value, obj.ycentroid.value,'ID{0}'.format(obj.id-1),fontsize=20, color='white')
    ax2.set_title('Segmentation Image')
    if pltshow == 0 or return_deblend == True:
        plt.close()
    else:
        plt.show()
    
    from photutils import EllipticalAperture
    segm_deblend_size = segm_deblend.areas
    apertures = []
    for obj in cat:
        print(len(segm_deblend_size), obj.id)
        size = segm_deblend_size[obj.id-1]
        position = (obj.xcentroid.value, obj.ycentroid.value)
        a_o = obj.semimajor_axis_sigma.value
        b_o = obj.semiminor_axis_sigma.value
        size_o = np.pi * a_o * b_o
        r = np.sqrt(size/size_o)*exp_sz
        a, b = a_o*r, b_o*r
        theta = obj.orientation.value
        apertures.append(EllipticalAperture(position, a, b, theta=theta))
    center_img = len(img)/2
    dis_sq = [np.sqrt((apertures[i].positions[0]-center_img)**2+(apertures[i].positions[1]-center_img)**2) for i in range(len(apertures))]
    dis_sq = np.asarray(dis_sq)
    c_index= np.where(dis_sq == dis_sq.min())[0][0]
    #from astropy.visualization.mpl_normalize import ImageNormalize
    #norm = ImageNormalize(stretch=SqrtStretch())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 10))
    ax1.imshow(img, origin='lower', cmap=my_cmap, norm=LogNorm(), vmin=vmin, vmax=vmax)
    ax1.set_title('Data')
    ax2.imshow(segm_deblend, origin='lower',
               cmap=segm_deblend.cmap(random_state=12345))
    ax2.set_title('Segmentation Image')
    for i in range(len(apertures)):
        aperture = apertures[i]
        aperture.plot(color='white', lw=1.5, ax=ax1)
        aperture.plot(color='white', lw=1.5, ax=ax2)
    if pltshow == 0 or return_deblend == True:
        plt.close()
    else:
        plt.show()
    from regions import PixCoord, EllipsePixelRegion
    from astropy.coordinates import Angle
    obj_masks = []  # In the script, the objects are 1, emptys are 0.
    for i in range(len(apertures)):
        aperture = apertures[i]
        x, y = aperture.positions
        center = PixCoord(x=x, y=y)
        theta = Angle(aperture.theta/np.pi*180.,'deg')
        reg = EllipsePixelRegion(center=center, width=aperture.a*2, height=aperture.b*2, angle=theta)
        patch = reg.as_artist(facecolor='none', edgecolor='red', lw=2)
        fig, axi = plt.subplots(1, 1, figsize=(10, 12.5))
        axi.add_patch(patch)
        mask_set = reg.to_mask(mode='center')
        mask = mask_set.to_image((len(img),len(img)))
        axi.imshow(mask, origin='lower')
        plt.close()
        if i != c_index:
            obj_masks.append(mask)
        elif i == c_index:
            target_mask = mask
    if obj_masks == []:
        obj_masks.append(np.zeros_like(img))
    if return_deblend == False :
        return target_mask, obj_masks
    else:
        return target_mask, obj_masks, segm_deblend

def find_loc_max(data, neighborhood_size = 4, threshold = 5):
    neighborhood_size = neighborhood_size
    threshold = threshold
    
    data_max = filters.maximum_filter(data, neighborhood_size) 
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1)/2    
        y.append(y_center)
    return x, y

def measure_FWHM(image, measure_radius = 10):
    seed_num = 2*measure_radius+1
    frm = len(image)
    q_frm = int(frm/4)
    x_center = np.where(image == image[q_frm:-q_frm,q_frm:-q_frm].max())[1][0]
    y_center = np.where(image == image[q_frm:-q_frm,q_frm:-q_frm].max())[0][0]
    
    x_n = np.asarray([image[y_center][x_center+i] for i in range(-measure_radius, measure_radius+1)]) # The x value, vertcial 
    y_n = np.asarray([image[y_center+i][x_center] for i in range(-measure_radius, measure_radius+1)]) # The y value, horizontal 
    xy_n = np.asarray([image[y_center+i][x_center+i] for i in range(-measure_radius, measure_radius+1)]) # The up right value, horizontal
    xy__n =  np.asarray([image[y_center-i][x_center+i] for i in range(-measure_radius, measure_radius+1)]) # The up right value, horizontal
#    print xy_n
#    print xy__n
    #popt, pcov = curve_fit(func, x, yn)
    from astropy.modeling import models, fitting
    g_init = models.Gaussian1D(amplitude=y_n.max(), mean=measure_radius, stddev=1.5)
    fit_g = fitting.LevMarLSQFitter()
    g_x = fit_g(g_init, range(seed_num), x_n)
    g_y = fit_g(g_init, range(seed_num), y_n)
    g_xy = fit_g(g_init, range(seed_num), xy_n)
    g_xy_ = fit_g(g_init, range(seed_num), xy__n)
    
    FWHM_ver = g_x.stddev.value * 2.355  # The FWHM = 2*np.sqrt(2*np.log(2)) * stdd = 2.355*stdd
    FWHM_hor = g_y.stddev.value * 2.355
    FWHM_xy = g_xy.stddev.value * 2.355 * np.sqrt(2.)
    FWHM_xy_ = g_xy_.stddev.value * 2.355 * np.sqrt(2.)
#    if (FWHM_hor-FWHM_ver)/FWHM_ver > 0.20:
#        print "Warning, the {0} have inconsistent FWHM".format(count), FWHM_ver, FWHM_hor
#    fig = plt.figure()
#    p_range = np.linspace(0, seed_num,50)
#    ax = fig.add_subplot(111)
#    ax.plot(p_range, g_x(p_range), c='b', label='x_Gaussian')
#    ax.plot(p_range, g_y(p_range), c='r', label='y_Gaussian')
#    ax.legend()
#    ax.scatter(range(seed_num), xy_n, color = 'b')
#    ax.scatter(range(seed_num), xy__n, color = 'r')
#    plt.show()
    return FWHM_ver, FWHM_hor, FWHM_xy, FWHM_xy_

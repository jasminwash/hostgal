#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 10:57:14 2018

@author: Dartoon

Doing the PSFs average given a list of PSF
"""

import numpy as np
from flux_profile import text_in_string_list,cr_mask
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm

def psf_ave(psfs_list, not_count=(), mode = 'CI',  mask_list=['default.reg'], mask_img_list=None):
    '''
    Produce the average for a list of psfs.
    
    
    Parameter
    --------
        psfs_list:
            A raw list of psfs array. 
        mode: the way to do the average.
            'direcity':
                directly do the average of the scaled PSF 
            'CI':
                Consider center (1/3 region) Intensity of the PSF. (weighted by
                the root-square of the intensity),---as the noise of PSF is most
                related to the Poission noise.The SNR of the image is related to
                root-square of the image.
            'CI_tot'    
                Silimar to CI, but use the total flux.
            'mid'
                Select the median value. But, four PSFs were recommed to load.
        no_count:
            The serial No. of psf which not considered.
        mask_list: 
            A list include all the mask.reg names
    Return
    --------
        A averaged and scaled PSF.
    '''
    ### masked PSF give the region.
    psf_NO = len(psfs_list)
    for i in range (psf_NO):
        if psfs_list[i] is None:
            psfs_list[i] = np.zeros_like([x for x in psfs_list if x is not None][0])
    psfs_l_msk = np.ones_like(psfs_list)  # To load the PSF plus masked area 
    for i in range(psf_NO):
        if i in not_count:
            print("The PSF{0} is not count".format(i))
            psfs_l_msk[i] = np.zeros_like(psfs_list[i])
        else:
            if mask_img_list is None:
                msk_counts, mask_lists = text_in_string_list("PSF{0}_".format(i), mask_list)
                mask = np.ones(psfs_list[i].shape)
                if msk_counts != 0:
                    for j in range(msk_counts):
                        mask *= cr_mask(image=psfs_list[i], filename=mask_lists[j])
            else:
                mask = mask_img_list[i]
            psfs_l_msk[i] = psfs_list[i] * mask
#    for i in range(psf_NO):
#            print("plot psfs_list", i)
#            plt.matshow(psfs_l_msk[i], origin= 'low', norm=LogNorm())
#            plt.colorbar()
#            plt.show()  
    ### Doing the average.
    if mode =='direct':
        for i in range(psf_NO):
            if psfs_l_msk[i].sum() != 0:
                psfs_l_msk[i] /= psfs_l_msk[i].sum()  # scale the image to a same level
#        print(np.where(np.isclose(psfs_l_msk,0)))
        psf_ave = np.nanmean(np.where(np.isclose(psfs_l_msk,0), np.nan, psfs_l_msk), axis=0)
#        print(psf_ave.sum())
        psf_std = np.nanstd(np.where(np.isclose(psfs_l_msk,0), np.nan, psfs_l_msk), axis=0)
        psf_std /= psf_ave.sum()
        psf_ave /= psf_ave.sum()
    elif mode == 'CI':
        weights = np.zeros(psf_NO)
        for i in range(psf_NO):
            box_c = len(psfs_l_msk[i])/2
            box_r = len(psfs_l_msk[i])/6
            if psfs_l_msk[i].sum() != 0:
                weights[i] = np.sqrt(np.sum(psfs_l_msk[i][box_c-box_r:box_c+box_r,box_c-box_r:box_c+box_r]))  # set weight based on their intensity (SNR)
                print("Sum flux for PSF center",i , ":", psfs_l_msk[i][box_c-box_r:box_c+box_r,box_c-box_r:box_c+box_r].sum())
                psfs_l_msk[i] /= weights[i] **2  # scale the image to a same level
        print("The final weights for doing the average:\n", weights)
#        print(abs(psfs_l_msk[3]).min())
        psfs_msk2nan=np.where(np.isclose(psfs_l_msk,0, rtol=1e-10, atol=1e-09), np.nan, psfs_l_msk)
        cleaned_psfs = np.ma.masked_array(psfs_msk2nan,np.isnan(psfs_msk2nan))
#        plt.imshow(cleaned_psfs.mask[5],origin = 'low')
#        plt.show()
        psf_ave = np.ma.average(cleaned_psfs,axis=0,weights=weights)
        diffs = (psfs_l_msk-psf_ave)**2
        diffs_msk2nan=np.where(np.isclose(psfs_l_msk,0, rtol=1e-10, atol=1e-09), np.nan, diffs)
        cleaned_diffs = np.ma.masked_array(diffs_msk2nan,np.isnan(diffs_msk2nan))
        psf_variance = np.ma.average(cleaned_diffs, weights=weights,axis=0)
        psf_std = np.sqrt(psf_variance)
        psf_std /= psf_ave.sum()
        psf_ave /= psf_ave.sum()
        psf_std = psf_std.data
        psf_ave = psf_ave.data
    elif mode == 'CI_tot':
        weights = np.zeros(psf_NO)
        for i in range(psf_NO):
            if psfs_l_msk[i].sum() != 0:
                weights[i] = np.sqrt(np.sum(psfs_l_msk[i]))  # set weight based on their intensity (SNR)
                print("Sum flux for PSF",i , ":", psfs_l_msk[i].sum())
                psfs_l_msk[i] /= psfs_l_msk[i].sum()  # scale the image to a same level
        print("The final weights for doing the average:\n", weights)
#        print(abs(psfs_l_msk[3]).min())
        psfs_msk2nan=np.where(np.isclose(psfs_l_msk,0, rtol=1e-10, atol=1e-09), np.nan, psfs_l_msk)
        cleaned_psfs = np.ma.masked_array(psfs_msk2nan,np.isnan(psfs_msk2nan))
#        plt.imshow(cleaned_psfs.mask[5],origin = 'low')
#        plt.show()
        psf_ave = np.ma.average(cleaned_psfs,axis=0,weights=weights)
        diffs = (psfs_l_msk-psf_ave)**2
        diffs_msk2nan=np.where(np.isclose(psfs_l_msk,0, rtol=1e-10, atol=1e-09), np.nan, diffs)
        cleaned_diffs = np.ma.masked_array(diffs_msk2nan,np.isnan(diffs_msk2nan))
        psf_variance = np.ma.average(cleaned_diffs, weights=weights,axis=0)
        psf_std = np.sqrt(psf_variance)
        psf_std /= psf_ave.sum()
        psf_ave /= psf_ave.sum()
        psf_std = psf_std.data
        psf_ave = psf_ave.data
    elif mode == 'mid':
        weights = np.zeros(psf_NO)        
        for i in range(psf_NO):
            box_c = len(psfs_l_msk[i])/2
            box_r = len(psfs_l_msk[i])/6
            if psfs_l_msk[i].sum() != 0:
                weights[i] = np.sqrt(np.sum(psfs_l_msk[i][box_c-box_r:box_c+box_r,box_c-box_r:box_c+box_r]))  # set weight based on their intensity (SNR)
                psfs_l_msk[i] /= weights[i] **2  # scale the image to a same level
        sz = len(psfs_l_msk[0])
        psf_ave = np.zeros_like(psfs_l_msk[0])
        for i in range(sz):
            for j in range(sz):
                psf_cell = psfs_l_msk[:,i,j]
                psf_cell = psf_cell[psf_cell!=0.]  # Delete the non-zeros.
                psf_ave[i,j] = median(psf_cell)
        psf_ave /= psf_ave.sum()        
    #### The PSF are found not very necessary to be shiftted. !!!! Note the high_CI is not ready --- high_res. mask is not OK.
#    if mode == 'high_CI':
#        psfs_high_list = np.empty([psf_NO, psfs_list[0].shape[0]*scale, psfs_list[0].shape[1]*scale])
#        #Creat a mask_high_list:
#        mask_high_list = np.ones_like(psfs_high_list)
#        for i in range(psf_NO):
#            psfs_high_list[i] = im_2_high_res(psfs_list[i], scale=scale)  # scale the image to a same level
#            psfs_high_list[i] *= mask_high_list[i]
#            psfs_high_list[i] /= np.sqrt(np.sum(psfs_high_list[0]))
#        psf_high_total = np.sum(psfs_high_list, axis=0)
#        sum_4ave = np.sum(mask_high_list, axis=0)
#        print(sum_4ave)
#        psf_high_final = psf_high_total/sum_4ave
#        psf_final = rebin(psf_high_final, scale = scale)
    else:
        raise ValueError("mode is not defined")
    if mode == 'mid':
        return psf_ave
    else:
        return psf_ave, psf_std

    
def median(l):
    l = sorted(l)
    if len(l)%2==1:  
        return l[len(l)/2];  
    else:  
        return (l[len(l)/2-1]+l[len(l)/2])/2.0;     

def psf_shift_ave(psfs_list, not_count=None, mode = 'direct',  mask_list=['default.reg'], count_psf_std = True, num_iter=1):
    '''
    Shifted the PSF to the center by fitting with the init_ave_PSF--- So that the final averaged PSF could be in the center (sharper).
    Parameter
    --------
        Similar to psf_ave()
        count_psf_std: Whether consider the psf_std when doing the PSF fitting.
        num_iter: is the numbers for doing the interation.
        
    Return
    --------
        A averaged PSF.
    '''
    from fit_psf_pos import fit_psf_pos
    from lenstronomy.Util.kernel_util import de_shift_kernel
    psf_init_ave, psf_std=psf_ave(psfs_list,mode = mode, not_count=not_count,
                  mask_list=mask_list)
    psf_final, psf_final_std = psf_init_ave, psf_std
    for iters in range(num_iter):
#        print("!!!!!iters is ", iters)
        shifted_psf_list = np.zeros_like(psfs_list)
        for i in range(len(psfs_list)):
            fitted_PSF = psfs_list[i]
            print("fiting PSF", i)
            if count_psf_std == True:
                ra_image, dec_image = fit_psf_pos(fitted_PSF, psf_final, psf_final_std)
            else:
                ra_image, dec_image = fit_psf_pos(fitted_PSF, psf_final)
            print(ra_image, dec_image)
            if abs(ra_image)>0.3 or abs(dec_image)>0.3:
                print("Warning, the fitted ra_image, dec_image for psf", i ,'is too large!!!:', ra_image, dec_image )
            shifted_psf_list[i] = de_shift_kernel(fitted_PSF, -ra_image, -dec_image)
            plt.imshow(shifted_psf_list[i], norm = LogNorm(),origin='low')
            plt.show()
        psf_final, psf_final_std=psf_ave(shifted_psf_list,mode = mode, not_count=not_count,
                mask_list=mask_list)
    return psf_final, psf_final_std


def rebin(image, scale=3):         
    '''
    Rebin a image to lower resolution 
    
    Parameter
    --------
        image:
        scale:
    Return
    --------
    '''
    shape = (len(image)/scale, len(image)/scale)
    sh = shape[0], scale, shape[1], scale
    return image.reshape(sh).mean(-1).mean(1)*scale**2


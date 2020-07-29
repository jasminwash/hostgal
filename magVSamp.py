#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 09:57:50 2017

@author: dxh
"""
import numpy as np
#from lenstronomy.SimulationAPI.simulations import Simulation
#SimAPI = Simulation()
##import sys
##sys.path.insert(0,'./parameters')
##from gene_para import gene_para
##seed=17#input("The seed for simulation:\n")
##para=gene_para(seed=seed)
##SERSIC_in_mag = para.lens_light()
##import matplotlib.pyplot as plt
##import astropy.io.fits as pyfits
##sigma_bkg = 2.  #  background noise per pixel
##exp_time = 1.  #  exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
##numPix = 240  #  pixel size
##deltaPix = 0.13/4  #  pixel size in arcsec (area per pixel = deltaPix**2)
##kwargs_data=SimAPI.data_configure(numPix, deltaPix, exp_time, sigma_bkg)
##zp= 25.9463
##
##psf = pyfits.open('../material/PSF/f160w/psf_sub4.fits')
##psf_pixel_high_res = psf[0].data
##psf.close()
##psf_type_tin = 'pixel'  # 'gaussian', 'pixel', 'NONE'
##kwargs_psf_tin_high_res = SimAPI.psf_configure(psf_type=psf_type_tin, kernelsize=len(psf_pixel_high_res), kernel=psf_pixel_high_res)
##kwargs_data = SimAPI.data_configure(numPix, deltaPix, exp_time, sigma_bkg)
#
#def getAmp(kwargs_data,light,zp,kwargs_psf):
#    '''
#    Input:
#        getAmp(kwargs_data,light,zp,kwargs_psf):
#    Goal:
#        get the sersic Amp by given the sereic mag.
#    How: 
#        First simulate the init image, then calculate the correspond Amp by flux_should/flux_init
#    Example:
#        lens_amp=mva.getAmp(kwargs_data=kwargs_data_high_res,light=lens_light_para, zp=zp, kwargs_psf=kwargs_psf_tin_high_res)
#    '''
#    flux_should=10**((light['mag_sersic']-zp)/(-2.5))
#    kwargs_options = {'lens_model_list': ['NONE'], 
#                      'lens_light_model_list': ['SERSIC_ELLIPSE'],
#                      'source_light_model_list': ['NONE'],
#                      'psf_type': 'NONE',    # 'gaussian', 'pixel', 'NONE'
#                      'foreground_shear': False,
#                      'point_source': False  # if True, simulates point source at source position of 'sourcePos_xy' in kwargs_else
#                      }
#    lens_light_para=light
#    kwargs_init_sersic = {'I0_sersic': 1, 'R_sersic': lens_light_para['R_sersic'],'n_sersic': lens_light_para['n_sersic'],
#                          'center_x': 0.0, 'center_y': 0.0, 'phi_G': lens_light_para['phi_G'], 'q': lens_light_para['q']} 
#    sersic_init_image = SimAPI.im_sim(kwargs_options, \
#                                      kwargs_data, kwargs_psf, [{}], [{}], [kwargs_init_sersic], [{}], no_noise=True)
#    flux_init=np.sum(sersic_init_image)
#    #mag_init=-2.5*np.log10(flux_init)+zp
#    amp_should=flux_should/flux_init
##    kwargs_sersic=kwargs_init_sersic
##    kwargs_sersic['I0_sersic']=amp_should
##    sersic_image = SimAPI.im_sim(kwargs_options, \
##                                 kwargs_data, kwargs_psf_tin_high_res, [{}], [{}], [kwargs_init_sersic], [{}], no_noise=True)
##    flux=np.sum(sersic_image)
##    sersic_mag=-2.5*np.log10(flux)+zp
##    plt.matshow(np.log10(sersic_image),origin='lower')
##    plt.colorbar()
##    plt.show()    
#    return amp_should


#Tried to get the amp for lenstronomy code, but find the generation is 0.2 mag fainter.
from scipy.special import gamma
from math import exp,pi,log10
def getMag(amp,SERSIC_in_mag,zp=None,deltaPix=None):
    n=SERSIC_in_mag['n_sersic']
    re = SERSIC_in_mag['R_sersic']/deltaPix
    k = 2.*n-1./3+4./(405.*n)+46/(25515.*n**2)
    cnts = (re**2)*amp*exp(k)*n*(k**(-2*n))*gamma(2*n)*2*pi
#    print 'conts_getmag', cnts
    mag=-2.5*log10(cnts) + zp
    return mag
#print getMag(0.422389413882, SERSIC_in_mag,zp= 25.9463, deltaPix=0.13/4 )

def getAmp(SERSIC_in_mag,zp=None,deltaPix=None):
    mag=SERSIC_in_mag['mag_sersic']
    n=SERSIC_in_mag['n_sersic']
    re = SERSIC_in_mag['R_sersic']/deltaPix
    k = 2.*n-1./3+4./(405.*n)+46/(25515.*n**2)
    cnts= 10.**(-0.4*(mag-zp))
#    print 'conts_getamp', cnts
    amp= cnts/((re**2)*exp(k)*n*(k**(-2*n))*gamma(2*n)*2*pi)
    return amp
#print getAmp(SERSIC_in_mag,zp=25.9463, deltaPix=0.13/4)

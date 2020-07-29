#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 20:23:36 2018

@author: Dartoon

A func for fitting the QSO with a input PSF. The PSF std is optional.

Update based on Lenstronomy version 0.7.0
Imporve the way for to pickle.dump

Update based on Lenstronomy version 0.8.1
Including:
    The lenstronomy package be universal to overall function     
To do:
    1. PSF std should be put to the PSF class.

Update based on Lenstronomy version 0.8.2
Including the input for kwargs_psf
Debug the missing of lightModel for None-sersic case.

Update based on Lenstronomy version 0.9.1
Notes:
    package dynesty would be needed. 
    
Update based on Lenstronomy version 1.2.0
Notes: 
    Require schwimmbad and multiprocess packages.
    Require emcee at >= 3.0.2 version.
"""

from matplotlib.pylab import plt
import numpy as np
import corner
import pickle
import time

import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.ImSim.image_linear_solve import ImageLinearFit
from lenstronomy.Plots.model_plot import ModelPlot
from lenstronomy.Plots import chain_plot
from lenstronomy.Sampling.parameters import Param

def fit_qso(QSO_im, psf_ave, psf_std=None, source_params=None,ps_param=None, background_rms=0.04, pix_sz = 0.168,
            exp_time = 300., fix_n=None, image_plot = True, corner_plot=True, supersampling_factor = 2, 
            flux_ratio_plot=False, deep_seed = False, fixcenter = False, QSO_msk=None, QSO_std=None,
            tag = None, no_MCMC= False, pltshow = 1, return_Chisq = False, dump_result = False, pso_diag=False,band='I'):
    '''
    A quick fit for the QSO image with (so far) single sersice + one PSF. The input psf noise is optional.
    
    Parameter
    --------
        QSO_im: An array of the QSO image.
        psf_ave: The psf image.
        psf_std: The psf noise, optional.
        source_params: The prior for the source. Default is given. If [], means no Sersic light.
        background_rms: default as 0.04
        exp_time: default at 2400.
        deep_seed: if Ture, more mcmc steps will be performed.
        tag: The name tag for save the plot
            
    Return
    --------
        Will output the fitted image (Set image_plot = True), the corner_plot and the flux_ratio_plot.
        source_result, ps_result, image_ps, image_host
    
    To do
    --------
        
    '''
    # data specifics need to set up based on the data situation
    background_rms = background_rms  #  background noise per pixel (Gaussian)
    exp_time = exp_time  #  exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
    numPix = len(QSO_im)  #  cutout pixel size
    deltaPix = pix_sz
    psf_type = 'PIXEL'  # 'gaussian', 'pixel', 'NONE'
    kernel = psf_ave

    kwargs_numerics = {'supersampling_factor': supersampling_factor, 'supersampling_convolution': False} 
    
    if source_params is None:
        # here are the options for the host galaxy fitting
        fixed_source = []
        kwargs_source_init = []
        kwargs_source_sigma = []
        kwargs_lower_source = []
        kwargs_upper_source = []
        
        # Disk component, as modelled by an elliptical Sersic profile
        if fix_n == None:
            fixed_source.append({})  # we fix the Sersic index to n=1 (exponential)
            kwargs_source_init.append({'R_sersic': 0.3, 'n_sersic': 2., 'e1': 0., 'e2': 0., 'center_x': 0., 'center_y': 0.})
            kwargs_source_sigma.append({'n_sersic': 0.5, 'R_sersic': 0.5, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1})
            kwargs_lower_source.append({'e1': -0.5, 'e2': -0.5, 'R_sersic': 0.1, 'n_sersic': 0.3, 'center_x': -10, 'center_y': -10})
            kwargs_upper_source.append({'e1': 0.5, 'e2': 0.5, 'R_sersic': 3., 'n_sersic': 7., 'center_x': 10, 'center_y': 10})
        elif fix_n is not None:
            fixed_source.append({'n_sersic': fix_n})
            kwargs_source_init.append({'R_sersic': 0.3, 'n_sersic': fix_n, 'e1': 0., 'e2': 0., 'center_x': 0., 'center_y': 0.})
            kwargs_source_sigma.append({'n_sersic': 0.001, 'R_sersic': 0.5, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1})
            kwargs_lower_source.append({'e1': -0.5, 'e2': -0.5, 'R_sersic': 0.1, 'n_sersic': fix_n, 'center_x': -10, 'center_y': -10})
            kwargs_upper_source.append({'e1': 0.5, 'e2': 0.5, 'R_sersic': 3, 'n_sersic': fix_n, 'center_x': 10, 'center_y': 10})
        source_params = [kwargs_source_init, kwargs_source_sigma, fixed_source, kwargs_lower_source, kwargs_upper_source]
    else:
        source_params = source_params
    
    if ps_param is None:
        center_x = 0.0
        center_y = 0.0
        point_amp = QSO_im.sum()/2.
        fixed_ps = [{}]
        kwargs_ps = [{'ra_image': [center_x], 'dec_image': [center_y], 'point_amp': [point_amp]}]
        kwargs_ps_init = kwargs_ps
        kwargs_ps_sigma = [{'ra_image': [0.05], 'dec_image': [0.05]}]
        kwargs_lower_ps = [{'ra_image': [-0.6], 'dec_image': [-0.6]}]
        kwargs_upper_ps = [{'ra_image': [0.6], 'dec_image': [0.6]}]
        ps_param = [kwargs_ps_init, kwargs_ps_sigma, fixed_ps, kwargs_lower_ps, kwargs_upper_ps]
    else:
        ps_param = ps_param
    
    #==============================================================================
    #Doing the QSO fitting 
    #==============================================================================
    kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, exp_time, background_rms, inverse=True)
    data_class = ImageData(**kwargs_data)
    kwargs_psf = {'psf_type': psf_type, 'kernel_point_source': kernel}
    psf_class = PSF(**kwargs_psf)
    data_class.update_data(QSO_im)
    
    point_source_list = ['UNLENSED'] * len(ps_param[0])
    pointSource = PointSource(point_source_type_list=point_source_list)
    
    if fixcenter == False:
        kwargs_constraints = {'num_point_source_list': [1] * len(ps_param[0])
                              }
    elif fixcenter == True:
        kwargs_constraints = {'joint_source_with_point_source': [[i, i] for i in range(len(ps_param[0]))],
                              'num_point_source_list': [1] * len(ps_param[0])
                              }
    
    
    if source_params == []:   #fitting image as Point source only.
        kwargs_params = {'point_source_model': ps_param}
        lightModel = None
        kwargs_model = {'point_source_model_list': point_source_list }
        imageModel = ImageModel(data_class, psf_class, point_source_class=pointSource, kwargs_numerics=kwargs_numerics)
        kwargs_likelihood = {'check_bounds': True,  #Set the bonds, if exceed, reutrn "penalty"
                             'image_likelihood_mask_list': [QSO_msk]
                     }
    elif source_params != []:
        kwargs_params = {'source_model': source_params,
                 'point_source_model': ps_param}

        light_model_list = ['SERSIC_ELLIPSE'] * len(source_params[0])
        lightModel = LightModel(light_model_list=light_model_list)
        kwargs_model = { 'source_light_model_list': light_model_list,
                        'point_source_model_list': point_source_list
                        }
        imageModel = ImageModel(data_class, psf_class, source_model_class=lightModel,
                                point_source_class=pointSource, kwargs_numerics=kwargs_numerics)
        # numerical options and fitting sequences
        kwargs_likelihood = {'check_bounds': True,  #Set the bonds, if exceed, reutrn "penalty"
                             'source_marg': False,  #In likelihood_module.LikelihoodModule -- whether to fully invert the covariance matrix for marginalization
                              'check_positive_flux': True, 
                              'image_likelihood_mask_list': [QSO_msk]
                             }
    
    kwargs_data['image_data'] = QSO_im
    if QSO_std is not None:
        kwargs_data['noise_map'] = QSO_std
    
    if psf_std is not None:
        kwargs_psf['psf_error_map'] = psf_std
    image_band = [kwargs_data, kwargs_psf, kwargs_numerics]
    multi_band_list = [image_band]

    kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}  # 'single-band', 'multi-linear', 'joint-linear'
    fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params)
    
    if deep_seed == False:
        fitting_kwargs_list = [
             ['PSO', {'sigma_scale': 0.8, 'n_particles': 100, 'n_iterations': 60}],
             ['MCMC', {'n_burn': 10, 'n_run': 10, 'walkerRatio': 50, 'sigma_scale': .1}]
            ]
    elif deep_seed == True:
         fitting_kwargs_list = [
             ['PSO', {'sigma_scale': 0.8, 'n_particles': 250, 'n_iterations': 250}],
             ['MCMC', {'n_burn': 100, 'n_run': 200, 'walkerRatio': 10, 'sigma_scale': .1}]
            ]
    if no_MCMC == True:
        fitting_kwargs_list = [fitting_kwargs_list[0],
                               ]        

    start_time = time.time()
    chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
    kwargs_result = fitting_seq.best_fit()
    ps_result = kwargs_result['kwargs_ps']
    source_result = kwargs_result['kwargs_source']
    if no_MCMC == False:
        sampler_type, samples_mcmc, param_mcmc, dist_mcmc  = chain_list[1]    
    
    end_time = time.time()
    print(end_time - start_time, 'total time needed for computation')
    print('============ CONGRATULATION, YOUR JOB WAS SUCCESSFUL ================ ')
    imageLinearFit = ImageLinearFit(data_class=data_class, psf_class=psf_class,
                                    source_model_class=lightModel,
                                    point_source_class=pointSource, 
                                    kwargs_numerics=kwargs_numerics)    
    image_reconstructed, error_map, _, _ = imageLinearFit.image_linear_solve(kwargs_source=source_result, kwargs_ps=ps_result)
    # this is the linear inversion. The kwargs will be updated afterwards
    modelPlot = ModelPlot(multi_band_list, kwargs_model, kwargs_result,
                          arrow_size=0.02, cmap_string="gist_heat", likelihood_mask_list=[QSO_msk])
    image_host = []  #!!! The linear_solver before and after LensModelPlot could have different result for very faint sources.
    for i in range(len(source_result)):
        image_host.append(imageModel.source_surface_brightness(source_result, de_lensed=True,unconvolved=False,k=i))
    
    image_ps = []
#    image_ps = np.array([])
    for i in range(len(ps_result)):
        image_ps.append(imageModel.point_source(ps_result, k = i))
#        print("i")
#        print(i)
#        image_ps=imageModel.point_source(ps_result, k = i)
#        help(imageModel.point_source)
#        print("piss")
#        print(ps_result)
#        print(np.unique(image_ps))
#        image_ps=np.append(image_ps,imageModel.point_source(ps_result, k = i))

    if pso_diag == True:
        f, axes = chain_plot.plot_chain_list(chain_list,0)
        if pltshow == 0:
            plt.close()
        else:
            plt.show()

    # let's plot the output of the PSO minimizer
    reduced_Chisq =  imageLinearFit.reduced_chi2(image_reconstructed, error_map)
    if image_plot:
        f, axes = plt.subplots(3, 3, figsize=(16, 16), sharex=False, sharey=False)
        modelPlot.data_plot(ax=axes[0,0], text="Data")
        modelPlot.model_plot(ax=axes[0,1])
        modelPlot.normalized_residual_plot(ax=axes[0,2], v_min=-6, v_max=6)
        
        modelPlot.decomposition_plot(ax=axes[1,0], text='Host galaxy', source_add=True, unconvolved=True)
        modelPlot.decomposition_plot(ax=axes[1,1], text='Host galaxy convolved', source_add=True)
        modelPlot.decomposition_plot(ax=axes[1,2], text='All components convolved', source_add=True, lens_light_add=True, point_source_add=True)
        
        modelPlot.subtract_from_data_plot(ax=axes[2,0], text='Data - Point Source', point_source_add=True)
        modelPlot.subtract_from_data_plot(ax=axes[2,1], text='Data - host galaxy', source_add=True)
        modelPlot.subtract_from_data_plot(ax=axes[2,2], text='Data - host galaxy - Point Source', source_add=True, point_source_add=True)
        
        f.tight_layout()
        #f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
        if tag is not None:
            f.savefig('{0}_fitted_image_'.format(tag)+band+'.pdf')
        if pltshow == 0:
            plt.close()
        else:
            plt.show()
        
    if corner_plot==True and no_MCMC==False:
        # here the (non-converged) MCMC chain of the non-linear parameters
        if not samples_mcmc == []:
           n, num_param = np.shape(samples_mcmc)
           plot = corner.corner(samples_mcmc, labels=param_mcmc, show_titles=True)
           if tag is not None:
               plot.savefig('{0}_para_corner.pdf'.format(tag))
           plt.close()               
           # if pltshow == 0:
           #     plt.close()
           # else:
           #     plt.show()
        
    if flux_ratio_plot==True and no_MCMC==False:
        param = Param(kwargs_model, kwargs_fixed_source=source_params[2], kwargs_fixed_ps=ps_param[2], **kwargs_constraints)
        mcmc_new_list = []
        if len(ps_param[2]) == 1:
            labels_new = ["Quasar flux"] +  ["host{0} flux".format(i) for i in range(len(source_params[0]))]
        else:
            labels_new = ["Quasar{0} flux".format(i) for i in range(len(ps_param[2]))] +  ["host{0} flux".format(i) for i in range(len(source_params[0]))]
        if len(samples_mcmc) > 10000:
            trans_steps = [len(samples_mcmc)-10000, len(samples_mcmc)]
        else:
            trans_steps = [0, len(samples_mcmc)]
        for i in range(trans_steps[0], trans_steps[1]):
            kwargs_out = param.args2kwargs(samples_mcmc[i])
            kwargs_light_source_out = kwargs_out['kwargs_source']
            kwargs_ps_out =  kwargs_out['kwargs_ps']
            image_reconstructed, _, _, _ = imageLinearFit.image_linear_solve(kwargs_source=kwargs_light_source_out, kwargs_ps=kwargs_ps_out)
            flux_quasar = []
            if len(ps_param[0]) == 1:
                image_ps_j = imageModel.point_source(kwargs_ps_out)
                flux_quasar.append(np.sum(image_ps_j))  
            else:    
                for j in range(len(ps_param[0])):
                    image_ps_j = imageModel.point_source(kwargs_ps_out, k=j)
                    flux_quasar.append(np.sum(image_ps_j))
            fluxs = []
            for j in range(len(source_params[0])):
                image_j = imageModel.source_surface_brightness(kwargs_light_source_out,unconvolved= False, k=j)
                fluxs.append(np.sum(image_j))
            mcmc_new_list.append(flux_quasar + fluxs )
            if int(i/1000) > int((i-1)/1000) :
                print(len(samples_mcmc), "MCMC samplers in total, finished translate:", i )
        plot = corner.corner(mcmc_new_list, labels=labels_new, show_titles=True)
        if tag is not None:
            plot.savefig('{0}_HOSTvsQSO_corner.pdf'.format(tag))
        if pltshow == 0:
            plt.close()
        else:
            plt.show()
    if QSO_std is None:
        noise_map = np.sqrt(data_class.C_D+np.abs(error_map))
    else:
        noise_map = np.sqrt(QSO_std**2+np.abs(error_map))
    if dump_result == True:
        if flux_ratio_plot==True and no_MCMC==False:
            trans_paras = [mcmc_new_list, labels_new, 'mcmc_new_list, labels_new']
        else:
            trans_paras = []
        picklename= tag + '.pkl'
        best_fit = [source_result, image_host, ps_result, image_ps,'source_result, image_host, ps_result, image_ps']
        chain_list_result = [chain_list, 'chain_list']
        kwargs_fixed_source=source_params[2]
        kwargs_fixed_ps=ps_param[2]
        classes = data_class, psf_class, lightModel, pointSource
        material = multi_band_list, kwargs_model, kwargs_result, QSO_msk, kwargs_fixed_source, kwargs_fixed_ps, kwargs_constraints, kwargs_numerics, classes
        pickle.dump([best_fit, chain_list_result, trans_paras, material], open(picklename, 'wb'))
    if return_Chisq == False:
        return source_result, ps_result, image_ps, image_host, noise_map
    elif return_Chisq == True:
        return source_result, ps_result, image_ps, image_host, noise_map, reduced_Chisq

def fit_galaxy(galaxy_im, psf_ave, psf_std=None, source_params=None, background_rms=0.04, pix_sz = 0.08,
            exp_time = 300., fix_n=None, image_plot = True, corner_plot=True,
            deep_seed = False, galaxy_msk=None, galaxy_std=None, flux_corner_plot = False,
            tag = None, no_MCMC= False, pltshow = 1, return_Chisq = False, dump_result = False, pso_diag=False):
    '''
    A quick fit for the QSO image with (so far) single sersice + one PSF. The input psf noise is optional.
    
    Parameter
    --------
        galaxy_im: An array of the QSO image.
        psf_ave: The psf image.
        psf_std: The psf noise, optional.
        source_params: The prior for the source. Default is given.
        background_rms: default as 0.04
        exp_time: default at 2400.
        deep_seed: if Ture, more mcmc steps will be performed.
        tag: The name tag for save the plot
            
    Return
    --------
        Will output the fitted image (Set image_plot = True), the corner_plot and the flux_ratio_plot.
        source_result, ps_result, image_ps, image_host
    
    To do
    --------
        
    '''
    # data specifics need to set up based on the data situation
    background_rms = background_rms  #  background noise per pixel (Gaussian)
    exp_time = exp_time  #  exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
    numPix = len(galaxy_im)  #  cutout pixel size
    deltaPix = pix_sz
    if psf_ave is not None:
        psf_type = 'PIXEL'  # 'gaussian', 'pixel', 'NONE'
        kernel = psf_ave
    
#    if psf_std is not None:
#        kwargs_numerics = {'subgrid_res': 1, 'psf_error_map': True}     #Turn on the PSF error map
#    else: 
    kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}
        
    if source_params is None:
        # here are the options for the host galaxy fitting
        fixed_source = []
        kwargs_source_init = []
        kwargs_source_sigma = []
        kwargs_lower_source = []
        kwargs_upper_source = []
        # Disk component, as modelled by an elliptical Sersic profile
        if fix_n == None:
            fixed_source.append({})  # we fix the Sersic index to n=1 (exponential)
            kwargs_source_init.append({'R_sersic': 0.3, 'n_sersic': 2., 'e1': 0., 'e2': 0., 'center_x': 0., 'center_y': 0.})
            kwargs_source_sigma.append({'n_sersic': 0.5, 'R_sersic': 0.1, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1})
            kwargs_lower_source.append({'e1': -0.5, 'e2': -0.5, 'R_sersic': 0.01, 'n_sersic': 0.3, 'center_x': -10, 'center_y': -10})
            kwargs_upper_source.append({'e1': 0.5, 'e2': 0.5, 'R_sersic': 3., 'n_sersic': 7., 'center_x': 10, 'center_y': 10})
        elif fix_n is not None:
            fixed_source.append({'n_sersic': fix_n})
            kwargs_source_init.append({'R_sersic': 0.3, 'n_sersic': fix_n, 'e1': 0., 'e2': 0., 'center_x': 0., 'center_y': 0.})
            kwargs_source_sigma.append({'n_sersic': 0.001, 'R_sersic': 0.1, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1})
            kwargs_lower_source.append({'e1': -0.5, 'e2': -0.5, 'R_sersic': 0.01, 'n_sersic': fix_n, 'center_x': -10, 'center_y': -10})
            kwargs_upper_source.append({'e1': 0.5, 'e2': 0.5, 'R_sersic': 3, 'n_sersic': fix_n, 'center_x': 10, 'center_y': 10})
        source_params = [kwargs_source_init, kwargs_source_sigma, fixed_source, kwargs_lower_source, kwargs_upper_source]
    else:
        source_params = source_params
    kwargs_params = {'source_model': source_params}
    
    #==============================================================================
    #Doing the QSO fitting 
    #==============================================================================
    kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, exp_time, background_rms, inverse=True)
    data_class = ImageData(**kwargs_data)
    if psf_ave is not None:
        kwargs_psf = {'psf_type': psf_type, 'kernel_point_source': kernel}
    else:
        kwargs_psf =  {'psf_type': 'NONE'}
    
    psf_class = PSF(**kwargs_psf)
    data_class.update_data(galaxy_im)
    
    light_model_list = ['SERSIC_ELLIPSE'] * len(source_params[0])
    lightModel = LightModel(light_model_list=light_model_list)
    
    kwargs_model = { 'source_light_model_list': light_model_list}
    # numerical options and fitting sequences
    kwargs_constraints = {}
    
    kwargs_likelihood = {'check_bounds': True,  #Set the bonds, if exceed, reutrn "penalty"
                         'source_marg': False,  #In likelihood_module.LikelihoodModule -- whether to fully invert the covariance matrix for marginalization
                          'check_positive_flux': True,       
                          'image_likelihood_mask_list': [galaxy_msk]
                         }
    kwargs_data['image_data'] = galaxy_im
    if galaxy_std is not None:
        kwargs_data['noise_map'] = galaxy_std
    if psf_std is not None:
        kwargs_psf['psf_error_map'] = psf_std
                  
    image_band = [kwargs_data, kwargs_psf, kwargs_numerics]
    multi_band_list = [image_band]
    
    kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}  # 'single-band', 'multi-linear', 'joint-linear'
    fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params)
    
    if deep_seed == False:
        fitting_kwargs_list = [
            ['PSO', {'sigma_scale': 0.8, 'n_particles': 50, 'n_iterations': 50}],
            ['MCMC', {'n_burn': 10, 'n_run': 10, 'walkerRatio': 50, 'sigma_scale': .1}]
            ]            
    elif deep_seed == True:
         fitting_kwargs_list = [
            ['PSO', {'sigma_scale': 0.8, 'n_particles': 100, 'n_iterations': 80}],
            ['MCMC', {'n_burn': 10, 'n_run': 15, 'walkerRatio': 50, 'sigma_scale': .1}]
            ]
    elif deep_seed == 'very_deep':
         fitting_kwargs_list = [
            ['PSO', {'sigma_scale': 0.8, 'n_particles': 150, 'n_iterations': 150}],
            ['MCMC', {'n_burn': 10, 'n_run': 20, 'walkerRatio': 50, 'sigma_scale': .1}]
            ]
    if no_MCMC == True:
        fitting_kwargs_list = [fitting_kwargs_list[0],
                               ]        
    
    start_time = time.time()
    chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
    kwargs_result = fitting_seq.best_fit()
    ps_result = kwargs_result['kwargs_ps']
    source_result = kwargs_result['kwargs_source']
    
    if no_MCMC == False:
        sampler_type, samples_mcmc, param_mcmc, dist_mcmc  = chain_list[1]      
    
#    chain_list, param_list, samples_mcmc, param_mcmc, dist_mcmc = fitting_seq.fit_sequence(fitting_kwargs_list)
#    lens_result, source_result, lens_light_result, ps_result, cosmo_temp = fitting_seq.best_fit()
    end_time = time.time()
    print(end_time - start_time, 'total time needed for computation')
    print('============ CONGRATULATION, YOUR JOB WAS SUCCESSFUL ================ ')
    # this is the linear inversion. The kwargs will be updated afterwards
    imageModel = ImageModel(data_class, psf_class, source_model_class=lightModel,kwargs_numerics=kwargs_numerics)
    imageLinearFit = ImageLinearFit(data_class=data_class, psf_class=psf_class,
                                       source_model_class=lightModel,
                                       kwargs_numerics=kwargs_numerics)    
    image_reconstructed, error_map, _, _ = imageLinearFit.image_linear_solve(kwargs_source=source_result, kwargs_ps=ps_result)
#    image_host = []   #!!! The linear_solver before and after could have different result for very faint sources.
#    for i in range(len(source_result)):
#        image_host_i = imageModel.source_surface_brightness(source_result,de_lensed=True,unconvolved=False, k=i)
#        print("image_host_i", source_result[i])
#        print("total flux", image_host_i.sum())
#        image_host.append(image_host_i)  
        
    # let's plot the output of the PSO minimizer
    modelPlot = ModelPlot(multi_band_list, kwargs_model, kwargs_result,
                          arrow_size=0.02, cmap_string="gist_heat", likelihood_mask_list=[galaxy_msk])  
    
    if pso_diag == True:
        f, axes = chain_plot.plot_chain_list(chain_list,0)
        if pltshow == 0:
            plt.close()
        else:
            plt.show()
                
    reduced_Chisq =  imageLinearFit.reduced_chi2(image_reconstructed, error_map)
    if image_plot:
        f, axes = plt.subplots(1, 3, figsize=(16, 16), sharex=False, sharey=False)
        modelPlot.data_plot(ax=axes[0])
        modelPlot.model_plot(ax=axes[1])
        modelPlot.normalized_residual_plot(ax=axes[2], v_min=-6, v_max=6)
        f.tight_layout()
        #f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
        if tag is not None:
            f.savefig('{0}_fitted_image.pdf'.format(tag))
        if pltshow == 0:
            plt.close()
        else:
            plt.show()
    image_host = []    
    for i in range(len(source_result)):
        image_host_i = imageModel.source_surface_brightness(source_result,de_lensed=True,unconvolved=False, k=i)
#        print("image_host_i", source_result[i])
#        print("total flux", image_host_i.sum())
        image_host.append(image_host_i)  
        
    if corner_plot==True and no_MCMC==False:
        # here the (non-converged) MCMC chain of the non-linear parameters
        if not samples_mcmc == []:
           n, num_param = np.shape(samples_mcmc)
           plot = corner.corner(samples_mcmc, labels=param_mcmc, show_titles=True)
           if tag is not None:
               plot.savefig('{0}_para_corner.pdf'.format(tag))
           if pltshow == 0:
               plt.close()
           else:
               plt.show()
    if flux_corner_plot ==True and no_MCMC==False:
        param = Param(kwargs_model, kwargs_fixed_source=source_params[2], **kwargs_constraints)
        mcmc_new_list = []
        labels_new = ["host{0} flux".format(i) for i in range(len(source_params[0]))]
        for i in range(len(samples_mcmc)):
            kwargs_out = param.args2kwargs(samples_mcmc[i])
            kwargs_light_source_out = kwargs_out['kwargs_source']
            kwargs_ps_out =  kwargs_out['kwargs_ps']
            image_reconstructed, _, _, _ = imageLinearFit.image_linear_solve(kwargs_source=kwargs_light_source_out, kwargs_ps=kwargs_ps_out)
            fluxs = []
            for j in range(len(source_params[0])):
                image_j = imageModel.source_surface_brightness(kwargs_light_source_out,unconvolved= False, k=j)
                fluxs.append(np.sum(image_j))
            mcmc_new_list.append( fluxs )
            if int(i/1000) > int((i-1)/1000) :
                print(len(samples_mcmc), "MCMC samplers in total, finished translate:", i    )
        plot = corner.corner(mcmc_new_list, labels=labels_new, show_titles=True)
        if tag is not None:
            plot.savefig('{0}_HOSTvsQSO_corner.pdf'.format(tag))
        if pltshow == 0:
            plt.close()
        else:
            plt.show() 

    if galaxy_std is None:
        noise_map = np.sqrt(data_class.C_D+np.abs(error_map))
    else:
        noise_map = np.sqrt(galaxy_std**2+np.abs(error_map))   
        
    if dump_result == True:
        if flux_corner_plot==True and no_MCMC==False:
            trans_paras = [source_params[2], mcmc_new_list, labels_new, 'source_params[2], mcmc_new_list, labels_new']
        else:
            trans_paras = []
        picklename= tag + '.pkl'
        best_fit = [source_result, image_host, 'source_result, image_host']
#        pso_fit = [chain_list, param_list, 'chain_list, param_list']
#        mcmc_fit = [samples_mcmc, param_mcmc, dist_mcmc, 'samples_mcmc, param_mcmc, dist_mcmc']
        chain_list_result = [chain_list, 'chain_list']
        pickle.dump([best_fit, chain_list_result, trans_paras], open(picklename, 'wb'))
        
    if return_Chisq == False:
        return source_result, image_host, noise_map
    elif return_Chisq == True:
        return source_result, image_host, noise_map, reduced_Chisq


def fit_qso_multiband(QSO_im_list, psf_ave_list, psf_std_list=None, source_params=None,ps_param=None,
                      background_rms_list=[0.04]*5, pix_sz = 0.168,
                      exp_time = 300., fix_n=None, image_plot = True, corner_plot=True,
                      flux_ratio_plot=True, deep_seed = False, fixcenter = False, QSO_msk_list=None,
                      QSO_std_list=None, tag = None, no_MCMC= False, pltshow = 1, new_band_seq=None):
    '''
    A quick fit for the QSO image with (so far) single sersice + one PSF. The input psf noise is optional.
    
    Parameter
    --------
        QSO_im: An array of the QSO image.
        psf_ave: The psf image.
        psf_std: The psf noise, optional.
        source_params: The prior for the source. Default is given.
        background_rms: default as 0.04
        exp_time: default at 2400.
        deep_seed: if Ture, more mcmc steps will be performed.
        tag: The name tag for save the plot
            
    Return
    --------
        Will output the fitted image (Set image_plot = True), the corner_plot and the flux_ratio_plot.
        source_result, ps_result, image_ps, image_host
    
    To do
    --------
        
    '''
    # data specifics need to set up based on the data situation
    background_rms_list = background_rms_list  #  background noise per pixel (Gaussian)
    exp_time = exp_time  #  exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
    numPix = len(QSO_im_list[0])  #  cutout pixel size
    deltaPix = pix_sz
    psf_type = 'PIXEL'  # 'gaussian', 'pixel', 'NONE'
    kernel_list = psf_ave_list
    if new_band_seq == None:
        new_band_seq= range(len(QSO_im_list))
    
#    if psf_std_list is not None:
#        kwargs_numerics_list = [{'subgrid_res': 1, 'psf_subgrid': False, 'psf_error_map': True}] * len(QSO_im_list)     #Turn on the PSF error map
#    else: 
    kwargs_numerics_list = [{'supersampling_factor': 1, 'supersampling_convolution': False}] * len(QSO_im_list)
    
    if source_params is None:
        # here are the options for the host galaxy fitting
        fixed_source = []
        kwargs_source_init = []
        kwargs_source_sigma = []
        kwargs_lower_source = []
        kwargs_upper_source = []
        
        # Disk component, as modelled by an elliptical Sersic profile
        if fix_n == None:
            fixed_source.append({})  # we fix the Sersic index to n=1 (exponential)
            kwargs_source_init.append({'R_sersic': 0.3, 'n_sersic': 2., 'e1': 0., 'e2': 0., 'center_x': 0., 'center_y': 0.})
            kwargs_source_sigma.append({'n_sersic': 0.5, 'R_sersic': 0.5, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1})
            kwargs_lower_source.append({'e1': -0.5, 'e2': -0.5, 'R_sersic': 0.1, 'n_sersic': 0.3, 'center_x': -10, 'center_y': -10})
            kwargs_upper_source.append({'e1': 0.5, 'e2': 0.5, 'R_sersic': 3., 'n_sersic': 7., 'center_x': 10, 'center_y': 10})
        elif fix_n is not None:
            fixed_source.append({'n_sersic': fix_n})
            kwargs_source_init.append({'R_sersic': 0.3, 'n_sersic': fix_n, 'e1': 0., 'e2': 0., 'center_x': 0., 'center_y': 0.})
            kwargs_source_sigma.append({'n_sersic': 0.001, 'R_sersic': 0.5, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1})
            kwargs_lower_source.append({'e1': -0.5, 'e2': -0.5, 'R_sersic': 0.1, 'n_sersic': fix_n, 'center_x': -10, 'center_y': -10})
            kwargs_upper_source.append({'e1': 0.5, 'e2': 0.5, 'R_sersic': 3, 'n_sersic': fix_n, 'center_x': 10, 'center_y': 10})
        source_params = [kwargs_source_init, kwargs_source_sigma, fixed_source, kwargs_lower_source, kwargs_upper_source]
    else:
        source_params = source_params
    
    if ps_param is None:
        center_x = 0.0
        center_y = 0.0
        point_amp = QSO_im_list[0].sum()/2.
        fixed_ps = [{}]
        kwargs_ps = [{'ra_image': [center_x], 'dec_image': [center_y], 'point_amp': [point_amp]}]
        kwargs_ps_init = kwargs_ps
        kwargs_ps_sigma = [{'ra_image': [0.01], 'dec_image': [0.01]}]
        kwargs_lower_ps = [{'ra_image': [-10], 'dec_image': [-10]}]
        kwargs_upper_ps = [{'ra_image': [10], 'dec_image': [10]}]
        ps_param = [kwargs_ps_init, kwargs_ps_sigma, fixed_ps, kwargs_lower_ps, kwargs_upper_ps]
    else:
        ps_param = ps_param
    
    kwargs_params = {'source_model': source_params,
                     'point_source_model': ps_param}
    
    #==============================================================================
    #Doing the QSO fitting 
    #==============================================================================
    kwargs_data_list, data_class_list = [], []
    for i in range(len(QSO_im_list)):
        kwargs_data_i = sim_util.data_configure_simple(numPix, deltaPix, exp_time, background_rms_list[i], inverse=True)
        kwargs_data_list.append(kwargs_data_i)
        data_class_list.append(ImageData(**kwargs_data_i))
    kwargs_psf_list = []
    psf_class_list = []
    for i in range(len(QSO_im_list)):
        kwargs_psf_i = {'psf_type': psf_type, 'kernel_point_source': kernel_list[i]}
        kwargs_psf_list.append(kwargs_psf_i)
        psf_class_list.append(PSF(**kwargs_psf_i))
        data_class_list[i].update_data(QSO_im_list[i])
    
    light_model_list = ['SERSIC_ELLIPSE'] * len(source_params[0])
    lightModel = LightModel(light_model_list=light_model_list)
    point_source_list = ['UNLENSED']
    pointSource = PointSource(point_source_type_list=point_source_list)
    
    imageModel_list = []
    for i in range(len(QSO_im_list)):
        kwargs_data_list[i]['image_data'] = QSO_im_list[i]
#        if QSO_msk_list is not None:
#            kwargs_numerics_list[i]['mask'] = QSO_msk_list[i]
        if QSO_std_list is not None:
            kwargs_data_list[i]['noise_map'] = QSO_std_list[i]
#        if psf_std_list is not None:
#            kwargs_psf_list[i]['psf_error_map'] = psf_std_list[i]
    
    image_band_list = []
    for i in range(len(QSO_im_list)):
        imageModel_list.append(ImageModel(data_class_list[i], psf_class_list[i], source_model_class=lightModel,
                                        point_source_class=pointSource, kwargs_numerics=kwargs_numerics_list[i]))
                  
        
        image_band_list.append([kwargs_data_list[i], kwargs_psf_list[i], kwargs_numerics_list[i]])
    multi_band_list = [image_band_list[i] for i in range(len(QSO_im_list))]
    
    # numerical options and fitting sequences
    
    kwargs_model = { 'source_light_model_list': light_model_list,
                    'point_source_model_list': point_source_list
                    }
    
    if fixcenter == False:
        kwargs_constraints = {'num_point_source_list': [1]
                              }
    elif fixcenter == True:
        kwargs_constraints = {'joint_source_with_point_source': [[0, 0]],
                              'num_point_source_list': [1]
                              }
    
    kwargs_likelihood = {'check_bounds': True,  #Set the bonds, if exceed, reutrn "penalty"
                         'source_marg': False,  #In likelihood_module.LikelihoodModule -- whether to fully invert the covariance matrix for marginalization
                          'check_positive_flux': True,       
                          'image_likelihood_mask_list': [QSO_msk_list]
                         }
    
#    mpi = False  # MPI possible, but not supported through that notebook.
    # The Params for the fitting. kwargs_init: initial input. kwargs_sigma: The parameter uncertainty. kwargs_fixed: fixed parameters;
    #kwargs_lower,kwargs_upper: Lower and upper limits.
    kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}  # 'single-band', 'multi-linear', 'joint-linear'
    fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params)
    
    if deep_seed == False:
        fitting_kwargs_list = [
            ['PSO', {'sigma_scale': 0.8, 'n_particles': 80, 'n_iterations': 60, 'compute_bands': [True]+[False]*(len(QSO_im_list)-1)}],
            ['align_images', {'n_particles': 10, 'n_iterations': 10, 'compute_bands': [False]+[True]*(len(QSO_im_list)-1)}],
            ['PSO', {'sigma_scale': 0.8, 'n_particles': 100, 'n_iterations': 200, 'compute_bands': [True]*len(QSO_im_list)}],
            ['MCMC', {'n_burn': 10, 'n_run': 20, 'walkerRatio': 50, 'sigma_scale': .1}]              
            ]
    elif deep_seed == True:
         fitting_kwargs_list = [
            ['PSO', {'sigma_scale': 0.8, 'n_particles': 150, 'n_iterations': 60, 'compute_bands': [True]+[False]*(len(QSO_im_list)-1)}],
            ['align_images', {'n_particles': 20, 'n_iterations': 20, 'compute_bands': [False]+[True]*(len(QSO_im_list)-1)}],
            ['PSO', {'sigma_scale': 0.8, 'n_particles': 150, 'n_iterations': 200, 'compute_bands': [True]*len(QSO_im_list)}],
            ['MCMC', {'n_burn': 20, 'n_run': 40, 'walkerRatio': 50, 'sigma_scale': .1}]                 
            ]
    if no_MCMC == True:
        del fitting_kwargs_list[-1]
    
    start_time = time.time()
#    lens_result, source_result, lens_light_result, ps_result, cosmo_temp, chain_list, param_list, samples_mcmc, param_mcmc, dist_mcmc = fitting_seq.fit_sequence(fitting_kwargs_list)
    chain_list, param_list, samples_mcmc, param_mcmc, dist_mcmc = fitting_seq.fit_sequence(fitting_kwargs_list)
    lens_result, source_result, lens_light_result, ps_result, cosmo_temp = fitting_seq.best_fit()    
    end_time = time.time()
    print(end_time - start_time, 'total time needed for computation')
    print('============ CONGRATULATION, YOUR JOB WAS SUCCESSFUL ================ ')
    source_result_list, ps_result_list = [], []
    image_reconstructed_list, error_map_list, image_ps_list, image_host_list, shift_RADEC_list=[], [], [], [],[]
    imageLinearFit_list = []
    for k in range(len(QSO_im_list)):
    # this is the linear inversion. The kwargs will be updated afterwards
        imageLinearFit_k = ImageLinearFit(data_class_list[k], psf_class_list[k], source_model_class=lightModel,
                                        point_source_class=pointSource, kwargs_numerics=kwargs_numerics_list[k])  
        image_reconstructed_k, error_map_k, _, _ = imageLinearFit_k.image_linear_solve(kwargs_source=source_result, kwargs_ps=ps_result)
        imageLinearFit_list.append(imageLinearFit_k) 
        
        [kwargs_data_k, kwargs_psf_k, kwargs_numerics_k] = fitting_seq.multi_band_list[k]
#        data_class_k = data_class_list[k] #ImageData(**kwargs_data_k)
#        psf_class_k = psf_class_list[k] #PSF(**kwargs_psf_k)
#        imageModel_k = ImageModel(data_class_k, psf_class_k, source_model_class=lightModel,
#                                point_source_class=pointSource, kwargs_numerics=kwargs_numerics_list[k])
        imageModel_k = imageModel_list[k]
        modelPlot = ModelPlot(multi_band_list[k], kwargs_model, lens_result, source_result,
                                 lens_light_result, ps_result, arrow_size=0.02, cmap_string="gist_heat", likelihood_mask=QSO_im_list[k])
        print("source_result", 'for', "k", source_result)
        image_host_k = []
        for i in range(len(source_result)):
            image_host_k.append(imageModel_list[k].source_surface_brightness(source_result,de_lensed=True,unconvolved=False, k=i))
        image_ps_k = imageModel_k.point_source(ps_result)
        # let's plot the output of the PSO minimizer
        
        image_reconstructed_list.append(image_reconstructed_k)
        source_result_list.append(source_result)
        ps_result_list.append(ps_result)
        error_map_list.append(error_map_k)
        image_ps_list.append(image_ps_k)
        image_host_list.append(image_host_k)
        if 'ra_shift' in fitting_seq.multi_band_list[k][0].keys():
            shift_RADEC_list.append([fitting_seq.multi_band_list[k][0]['ra_shift'], fitting_seq.multi_band_list[k][0]['dec_shift']])
        else:
            shift_RADEC_list.append([0,0])
        if image_plot:
            f, axes = plt.subplots(3, 3, figsize=(16, 16), sharex=False, sharey=False)
            modelPlot.data_plot(ax=axes[0,0], text="Data")
            modelPlot.model_plot(ax=axes[0,1])
            modelPlot.normalized_residual_plot(ax=axes[0,2], v_min=-6, v_max=6)
            
            modelPlot.decomposition_plot(ax=axes[1,0], text='Host galaxy', source_add=True, unconvolved=True)
            modelPlot.decomposition_plot(ax=axes[1,1], text='Host galaxy convolved', source_add=True)
            modelPlot.decomposition_plot(ax=axes[1,2], text='All components convolved', source_add=True, lens_light_add=True, point_source_add=True)
            
            modelPlot.subtract_from_data_plot(ax=axes[2,0], text='Data - Point Source', point_source_add=True)
            modelPlot.subtract_from_data_plot(ax=axes[2,1], text='Data - host galaxy', source_add=True)
            modelPlot.subtract_from_data_plot(ax=axes[2,2], text='Data - host galaxy - Point Source', source_add=True, point_source_add=True)
            f.tight_layout()
            if tag is not None:
                f.savefig('{0}_fitted_image_band{1}.pdf'.format(tag,new_band_seq[k]))
            if pltshow == 0:
                plt.close()
            else:
                plt.show()
            
            if corner_plot==True and no_MCMC==False and k ==0:
                # here the (non-converged) MCMC chain of the non-linear parameters
                if not samples_mcmc == []:
                   n, num_param = np.shape(samples_mcmc)
                   plot = corner.corner(samples_mcmc, labels=param_mcmc, show_titles=True)
                   if tag is not None:
                       plot.savefig('{0}_para_corner.pdf'.format(tag))
                   if pltshow == 0:
                       plt.close()
                   else:
                       plt.show()
            if flux_ratio_plot==True and no_MCMC==False:
                param = Param(kwargs_model, kwargs_fixed_source=source_params[2], kwargs_fixed_ps=fixed_ps, **kwargs_constraints)
                mcmc_new_list = []
                labels_new = [r"Quasar flux", r"host_flux", r"source_x", r"source_y"]
                # transform the parameter position of the MCMC chain in a lenstronomy convention with keyword arguments #
                for i in range(len(samples_mcmc)/10):
                    kwargs_lens_out, kwargs_light_source_out, kwargs_light_lens_out, kwargs_ps_out, kwargs_cosmo = param.getParams(samples_mcmc[i+ len(samples_mcmc)/10*9])
                    image_reconstructed, _, _, _ = imageLinearFit_list[k].image_linear_solve(kwargs_source=kwargs_light_source_out, kwargs_ps=kwargs_ps_out)
                    
                    image_ps = imageModel_list[k].point_source(kwargs_ps_out)
                    flux_quasar = np.sum(image_ps)
                    image_disk = imageModel_list[k].source_surface_brightness(kwargs_light_source_out,de_lensed=True,unconvolved=False, k=0)
                    flux_disk = np.sum(image_disk)
                    source_x = kwargs_ps_out[0]['ra_image']
                    source_y = kwargs_ps_out[0]['dec_image']
                    if flux_disk>0:
                        mcmc_new_list.append([flux_quasar, flux_disk, source_x, source_y])
                plot = corner.corner(mcmc_new_list, labels=labels_new, show_titles=True)
                if tag is not None:
                    plot.savefig('{0}_HOSTvsQSO_corner_band{1}.pdf'.format(tag,new_band_seq[k]))
                if pltshow == 0:
                    plt.close()
                else:
                    plt.show()
    errp_list = []
    for k in range(len(QSO_im_list)):
        if QSO_std_list is None:
            errp_list.append(np.sqrt(data_class_list[k].C_D+np.abs(error_map_list[k])))
        else:
            errp_list.append(np.sqrt(QSO_std_list[k]**2+np.abs(error_map_list[k])))
    return source_result_list, ps_result_list, image_ps_list, image_host_list, errp_list, shift_RADEC_list, fitting_seq     #fitting_seq.multi_band_list

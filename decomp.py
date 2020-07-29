import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import glob
#from gen_fit_id import gen_fit_id
from photutils import make_source_mask
import os
#plt.ion()
import sys
sys.path.insert(0,'./py_tools/')
from astropy.io import fits
from fit_qso import fit_qso
from transfer_to_result import transfer_to_result, transfer_obj_to_result, transfer_to_result_i
from mask_objects import detect_obj
from flux_profile import profiles_compare, flux_profile
from matplotlib.colors import LogNorm
import copy
import time
import pickle

import photutils
import numpy as np
from astropy.nddata.utils import Cutout2D
from astropy import units as u

import aplpy

def decomp(fitsfile, psffile, deepseed, fix_center, runMCMC, name):
    
    #Setting the fitting condition:
    deep_seed = deepseed  #Set as True to put more seed and steps to fit.
#    pltshow = 1 #Note that setting plt.ion() in line27, the plot won't show anymore if running in terminal.
    pltshow=0
    pix_scale = 0.168
    fixcenter = fix_center
    run_MCMC = runMCMC
    zp=27.0
    fitfile = fits.open(fitsfile)
    psf1 = pyfits.getdata(psffile)
    #was 3 before, changed it to 1
    QSO_img1 = fitfile[1].data
    #print(QSO_img1)
    #print(QSO_img1.shape)
    #I CHANGED IT TO 1 IT SHOUDL ABSOLUTELY BE 3
    QSO_std1 =  fitfile[3].data **0.5
    
   
    frame_size = len(QSO_img1)
    cut_to = 38
    #cut_to = 30
    ct = (frame_size - cut_to)/2
    
    '''
    QSO_img = QSO_img1[ct:-ct, ct:-ct]
    QSO_std = QSO_std1[ct:-ct, ct:-ct]
    
    '''
    #print('before for image: ', len(QSO_img1), len(QSO_img1[1]))
    #print('before for std: ', len(QSO_std1), len(QSO_std1[1]))
    
#goes [y,x]    
    if(len(QSO_img1) - len(QSO_img1[1])==0):
        cut_to = len(QSO_img1) - 2
        ct = int((frame_size - cut_to)/2)
        #print(ct)
        #print('square: ', len(QSO_img1),len(QSO_img1[1]))
        QSO_img = QSO_img1[ct:-ct, ct:-ct]
    if(len(QSO_img1) - len(QSO_img1[1]) > 0):
        cut_to = len(QSO_img1[1]) - 1
        ct = int((frame_size - cut_to)/2)
        #print('left is bigger', len(QSO_img1),len(QSO_img1[1]))
        diff = len(QSO_img1) - len(QSO_img1[1])
        QSO_img = QSO_img1[ct+diff:-ct, ct:-ct]
        #QSO_img = QSO_img1[20:-ct, ct:-ct]
    if(len(QSO_img1) - len(QSO_img1[1]) < 0):
        cut_to = len(QSO_img1) - 2
        ct = int((frame_size - cut_to)/2)
        diff = len(QSO_img1) - len(QSO_img1[1])
        QSO_img = QSO_img1[ct+diff:-ct, ct:-ct]
        #print('right is bigger', len(QSO_img1),len(QSO_img1[1]))
        #print(len(QSO_img1), len(QSO_img1[1]))
    
    #print('after for img: ', len(QSO_img), len(QSO_img[1]))
             

        
    #print('ok now QSO_std time')        
    if(len(QSO_std1) == len(QSO_std1[1])):
        ct = int((frame_size - cut_to)/2)
        QSO_std = QSO_std1[ct:-ct, ct:-ct]
    if(len(QSO_std1) - len(QSO_std1[1]) >0):
        ct = int((frame_size - cut_to)/2)
        diff = len(QSO_std1) - len(QSO_std1[1])
        QSO_std = QSO_std1[ct+diff:-ct, ct:-ct]
    if(len(QSO_std1) - len(QSO_std1[1]) <0):
        ct = int((frame_size - cut_to)/2)
        diff = len(QSO_std1) - len(QSO_std1[1])
        QSO_std = QSO_std1[ct+diff:-ct, ct:-ct]
        #print('fix this!!!')

    #print('after for std: ', len(QSO_std), len(QSO_std[1]))
    
    
    #print('psf size before: ', psf1.shape)
    
    #now do ct for psf
    frame_size = len(psf1)
    cut_to = 39
    ct = int((frame_size - cut_to)/2)
   
    if(len(psf1) == len(psf1[1])):
        cut_to = len(psf1) - 2
        ct = int((frame_size - cut_to)/2)
        psf = psf1[ct:-ct, ct:-ct]
    if(len(psf1) - len(psf1[1]) > 0):
        cut_to = len(psf1[1]) - 1
        ct = int((frame_size - cut_to)/2)
        #print('weird psf')
        diff  =len(psf1) - len(psf1[1])
        psf = psf1[ct + diff:-ct, ct:-ct]
    if(len(psf1) - len(psf1[1]) < 0):
        cut_to = len(psf1) - 2
        ct = int((frame_size - cut_to)/2)
        #print('weird psf')
        diff  = len(psf1) - len(psf1[1])
        psf = psf1[ct:-ct, ct:-ct+diff]
        
    #print('psf size after: ', psf.shape)
    
    
    #if(len(psf1) - len(psf1[1]) == 1):
     #   psf = psf1[ct + 1:-ct, ct:-ct]
    #if(len(psf1) - len(psf1[1]) == -1):
     #   psf = psf1[ct:-ct, ct+1:-ct]
    #print('psf size after: ', psf.shape)
        
    #print(QSO_img)
    #print(QSO_img.shape)
    
    #psf = psf1

    #%%
    #==============================================================================
    # input the objects components and parameteres
    #==============================================================================
 
    objs, Q_index = detect_obj(QSO_img,pltshow = pltshow)
    if objs=="bad!": return "No sources!",1,1
    qso_info = objs[Q_index]
    obj = [objs[i] for i in range(len(objs)) if i != Q_index]
    fixed_source = []
    kwargs_source_init = []
    kwargs_source_sigma = []
    kwargs_lower_source = []
    kwargs_upper_source = []
    fixed_source.append({})  
    
    kwargs_source_init.append({'R_sersic': 0.3, 'n_sersic': 2., 'e1': 0., 'e2': 0., 'center_x': 0., 'center_y': 0.})
    kwargs_source_sigma.append({'n_sersic': 0.5, 'R_sersic': 0.5, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1})
    kwargs_lower_source.append({'e1': -0.5, 'e2': -0.5, 'R_sersic': 0.1, 'n_sersic': 0.3, 'center_x': -0.5, 'center_y': -0.5})
    kwargs_upper_source.append({'e1': 0.5, 'e2': 0.5, 'R_sersic': 3., 'n_sersic': 7., 'center_x': 0.5, 'center_y': 0.5})
    if len(obj) >= 1:
        for i in range(len(obj)):
            fixed_source.append({})  
            kwargs_source_init.append({'R_sersic': obj[i][1] * pix_scale, 'n_sersic': 2., 'e1': 0., 'e2': 0., 'center_x': -obj[i][0][0]*pix_scale, 'center_y': obj[i][0][1]*pix_scale})
            kwargs_source_sigma.append({'n_sersic': 0.5, 'R_sersic': 0.5, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1})
            kwargs_lower_source.append({'e1': -0.5, 'e2': -0.5, 'R_sersic': obj[i][1] * pix_scale/5, 'n_sersic': 0.3, 'center_x': -obj[i][0][0]*pix_scale-10, 'center_y': obj[i][0][1]*pix_scale-10})
            kwargs_upper_source.append({'e1': 0.5, 'e2': 0.5, 'R_sersic': 3., 'n_sersic': 7., 'center_x': -obj[i][0][0]*pix_scale+10, 'center_y': obj[i][0][1]*pix_scale+10})
    source_params = [kwargs_source_init, kwargs_source_sigma, fixed_source, kwargs_lower_source, kwargs_upper_source]
    #%%
    #%%
    # =============================================================================
    # Creat the QSO mask
    # =============================================================================
    from mask_objects import mask_obj
    _ , _, deblend_sources = mask_obj(QSO_img, snr=1.2, npixels=50, return_deblend = True)
    

    print("deblend image to find the ID for the Objects for the mask:")
    plt.imshow(deblend_sources, origin='lower',cmap=deblend_sources.cmap(random_state=12345))
    plt.colorbar()
    if pltshow == 0:
        plt.close()
    #else:
        #plt.show()
    QSO_msk = None
    
    
    
    
    import os
    if not os.path.exists('/Users/jasminwashington/Google Drive File Stream/My Drive/usrp/git_repo/hostgal/py_tools/targets/shortlist/' + name + '/'):
        os.makedirs('/Users/jasminwashington/Google Drive File Stream/My Drive/usrp/git_repo/hostgal/py_tools/targets/shortlist/' + name + '/')
    tag = '/Users/jasminwashington/Google Drive File Stream/My Drive/usrp/git_repo/hostgal/py_tools/targets/shortlist/' + name + '/' + name
    

    
    
    #tag = 'home/michelle/Desktop/' + name

    
    # for if you want to save the image to your computer    
    
   
    source_result, ps_result, image_ps, image_host, error_map,rx2=fit_qso(QSO_img, psf_ave=psf, psf_std = None,
                                                                     source_params=source_params, QSO_msk = QSO_msk, fixcenter=fixcenter,
                                                                     pix_sz = pix_scale, no_MCMC = (run_MCMC==False),
                                                                     QSO_std =QSO_std, tag=tag, deep_seed= deep_seed, pltshow=pltshow,
                                                                     corner_plot=False, flux_ratio_plot=True, dump_result=run_MCMC,band='I',return_Chisq=True)
#    print("Reds")
#    print(rx2)
    if pltshow == 0:
        plot_compare=False
        fits_plot =False
    else:
        plot_compare=True
        fits_plot =True
    
#    print("imps:")
#    print(image_ps)
    worm=np.array(image_ps[0])
#    print(np.unique(image_ps))
    image_ps=worm
    print("Source")
    print(source_result)
    result,flux_list,c2mask = transfer_to_result_i(data=QSO_img, pix_sz = pix_scale,
                                source_result=source_result, ps_result=ps_result, image_ps=image_ps, image_host=image_host, error_map=error_map,
                                zp=zp, fixcenter=fixcenter,ID='Example', QSO_msk = QSO_msk, tag=tag, plot_compare = plot_compare, band='I')
    
    if QSO_msk is None:
        QSO_mask = QSO_img*0 + 1
    dmps=(flux_list[0]-flux_list[1])*QSO_mask
    hdu=fits.PrimaryHDU(dmps)
    hdu.writeto('/Users/jasminwashington/Google Drive File Stream/My Drive/usrp/git_repo/hostgal/py_tools/targets/shortlist/' + name + '/'+name+'_I.fits',overwrite=True)
    hdu1=fits.PrimaryHDU(flux_list[0])
    hdu1.writeto('/Users/jasminwashington/Google Drive File Stream/My Drive/usrp/git_repo/hostgal/py_tools/targets/shortlist/' + name + '/'+name+'_I_all.fits',overwrite=True)
    return result,source_result,c2mask
                                                               
    ''' 
    # for if you dont want to save the image to your computer    
    source_result, ps_result, image_ps, image_host, error_map=fit_qso(QSO_img, psf_ave=psf, psf_std = None,
                                                                     source_params=source_params, QSO_msk = QSO_msk, fixcenter=fixcenter,
                                                                     pix_sz = pix_scale, no_MCMC = (run_MCMC==False),
                                                                     QSO_std =QSO_std, tag = None, deep_seed= deep_seed, pltshow=pltshow,
                                                                     corner_plot=False, flux_ratio_plot=True, dump_result=run_MCMC)


    
    
    if pltshow == 0:
        plot_compare=False
        fits_plot =False
    else:
        plot_compare=True
        fits_plot =True
     


    result = transfer_to_result(data=QSO_img, pix_sz = pix_scale,
                                source_result=source_result, ps_result=ps_result, image_ps=image_ps, image_host=image_host, error_map=error_map,
                                zp=zp, fixcenter=fixcenter,ID='Example', QSO_msk = QSO_msk, tag=None, plot_compare = plot_compare)

    return result
    '''

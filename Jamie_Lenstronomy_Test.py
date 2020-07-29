#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from transfer_to_result import transfer_to_result, transfer_obj_to_result
from mask_objects import detect_obj
from flux_profile import profiles_compare, flux_profile
from matplotlib.colors import LogNorm
from matplotlib import rcParams
import copy
import time
import pickle
import aplpy
import pandas as pd

import photutils

#Special extra file
from decomp import decomp
from decomp1 import decomp as d1
from decompsf import decomp as dpsf

from astropy.visualization import make_lupton_rgb
from astropy.utils.data import get_pkg_data_filename

rcParams['figure.figsize'] = (8, 8)
rcParams['axes.titlesize'] = 16
rcParams['axes.labelsize'] = 14
rcParams['legend.fontsize'] = 16
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12

import os,os.path
#numfile=len([name for name in os.listdir('./psf-20200706-230617Z/') if os.path.isfile('./psf-20200706-230617Z/'+name)])
numfile=len([name for name in os.listdir('./shortlistpsf_I/') if os.path.isfile('./shortlistpsf_I/'+name)])
print(numfile)

#hdulsamp=fits.open("sample.fits")
#headerssamp=hdulsamp[0].data
#datasamp=hdulsamp[1].data

#name=datasamp.field('SDSS_NAME')
#ra=datasamp.field('RA')
#zshift=datasamp.field('Z')

#pathpsf = './psf-20200708-002430Z/'
#pathfit = './arch-200708-002458/'

pathpsf = './shortlistpsf_'
pathfit = './shortlistcut_'

name=([])
ra=([])
dec=([])
zshift=([])
X = pd.read_csv('shorterlistsources.txt', sep=" ", header=0)
for d in range(len(X['name'])):
    name.append(X['name'][d])
    ra.append(X['ra'][d])
    dec.append(X['dec'][d])
    zshift.append(X['z'][d])

#print(name)
    
#print(np.arange(2,numfile+2))
'''
def cshort(mag1,mag2,zshift):
    color=([],[])
    maglim=24
    diffmag=mag1-mag2
    if mag1<maglim and mag2<maglim: return diffmag
'''
        

def colors(hostmag,zshift,name,baddies):
    c=open("colors.txt","w")
    r=0
    b=0
    gicolor=([],[])
    ricolor=([],[])
    izcolor=([],[])
    iycolor=([],[])
    grcolor=([],[])
    zycolor=([],[])
    gzcolor=([],[])
    gycolor=([],[])
    rzcolor=([],[])
    rycolor=([],[])
    
    gi=open("gicolor.dat","w")
    ri=open("ricolor.dat","w")
    iz=open("izcolor.dat","w")
    iy=open("iycolor.dat","w")
    gr=open("grcolor.dat","w")
    zy=open("zycolor.dat","w")
    gz=open("gzcolor.dat","w")
    gy=open("gycolor.dat","w")
    rz=open("rzcolor.dat","w")
    ry=open("rycolor.dat","w")

    gi.write("z color\n")
    ri.write("z color\n")
    iz.write("z color\n")
    iy.write("z color\n")
    gr.write("z color\n")
    zy.write("z color\n")
    gz.write("z color\n")
    gy.write("z color\n")
    rz.write("z color\n")
    ry.write("z color\n")
    
    for t in np.arange(0,len(hostmag),step=5):
        if t==baddies[b]:
            b+=1
            continue
        maglim=24
        imag=hostmag[t]
        gmag=hostmag[t+1]
        rmag=hostmag[t+2]
        zmag=hostmag[t+3]
        ymag=hostmag[t+4]
        diffgi=gmag-imag
        diffri=rmag-imag
        diffiz=imag-zmag
        diffiy=imag-ymag
        diffgr=gmag-rmag
        diffzy=zmag-ymag
        diffgz=gmag-zmag
        diffgy=gmag-ymag
        diffrz=rmag-zmag
        diffry=rmag-ymag
        if gmag<maglim and imag<maglim:
            gicolor[0].append(zshift[r])
            gicolor[1].append(diffgi)
            gi.write("%f %f\n" %(zshift[r],diffgi))
        if rmag<maglim and imag<maglim:
            ricolor[0].append(zshift[r])
            ricolor[1].append(diffri)
            ri.write("%f %f\n" %(zshift[r],diffri))
        if zmag<maglim and imag<maglim:
            izcolor[0].append(zshift[r])
            izcolor[1].append(diffiz)
            iz.write("%f %f\n" %(zshift[r],diffiz))
        if ymag<maglim and imag<maglim:
            iycolor[0].append(zshift[r])
            iycolor[1].append(diffiy)
            iy.write("%f %f\n" %(zshift[r],diffiy))
        if gmag<maglim and rmag<maglim:
            grcolor[0].append(zshift[r])
            grcolor[1].append(diffgr)
            gr.write("%f %f\n" %(zshift[r],diffgr))
        if zmag<maglim and ymag<maglim:
            zycolor[0].append(zshift[r])
            zycolor[1].append(diffzy)
            zy.write("%f %f\n" %(zshift[r],diffzy))
        if gmag<maglim and zmag<maglim:
            gzcolor[0].append(zshift[r])
            gzcolor[1].append(diffgz)
            gz.write("%f %f\n" %(zshift[r],diffgz))
        if gmag<maglim and ymag<maglim:
            gycolor[0].append(zshift[r])
            gycolor[1].append(diffgy)
            gy.write("%f %f\n" %(zshift[r],diffgy))
        if rmag<maglim and zmag<maglim:
            rzcolor[0].append(zshift[r])
            rzcolor[1].append(diffrz)
            rz.write("%f %f\n" %(zshift[r],diffrz))
        if rmag<maglim and ymag<maglim:
            rycolor[0].append(zshift[r])
            rycolor[1].append(diffry)
            ry.write("%f %f\n" %(zshift[r],diffry))
        r+=1
    gi.close()
    ri.close()
    iz.close()
    iy.close()
    gr.close()
    zy.close()
    gz.close()
    gy.close()
    rz.close()
    ry.close()

    return gicolor,ricolor,izcolor,iycolor,grcolor,zycolor,gzcolor,gycolor,rzcolor,rycolor


# In[2]:


#k=0

hostamp=([])
hostmag=([])
decomps=([])
c2=([])
c2psf=([])
rc21=([])
rc22=([])
baddies=([])

q=open("hostmag.dat","w")
q.write("ID filter hostmag\n")
#numfile+2
for j in np.arange(2,numfile+2):
    fils=['I','G','R','Z','Y']
#    if j==134: 
#        baddies.append(134)
#        continue
    k=j-2
    if name[k] == '00000-00000': sdssid=str(ra[k])
    else: sdssid=name[k]
    print(sdssid)
    startpsf=str(j)+'-p'
    startfit=str(j)+'-c'
    for i in os.listdir(pathpsf+'I/'):
        if os.path.isfile(os.path.join(pathpsf+'I/',i)) and i.startswith(startpsf):
            hsc_psf_file = pathpsf+'I/'+i
    for c in os.listdir(pathfit+'I/'):
        if os.path.isfile(os.path.join(pathfit+'I/',c)) and c.startswith(startfit):
            hsc_fits_file = pathfit+'I/'+c
    decomp_info,iresults,c2mask = decomp(hsc_fits_file, hsc_psf_file, False, False, False, sdssid)
    if decomp_info=="No sources!":
        hostamp.append(0)
        hostmag.append(0)
        hostmag.append(0)
        hostmag.append(0)
        hostmag.append(0)
        hostmag.append(0)
        decomps.append(0)
        c2.append(0)
        c2psf.append(0)
        rc21.append(0)
        rc22.append(0)
        baddies.append(k)
        for l in range(len(fils)):
            q.write("%s %s 0\n" %(sdssid,fils[l]))
        continue
    decomp_info_psf,c2maskpsf=dpsf(hsc_fits_file, hsc_psf_file, False, False, False, sdssid)
    if abs(c2mask-c2maskpsf)<7:
        hostamp.append(0)
        hostmag.append(0)
        hostmag.append(0)
        hostmag.append(0)
        hostmag.append(0)
        hostmag.append(0)
        decomps.append(0)
        c2.append(0)
        c2psf.append(0)
        rc21.append(0)
        rc22.append(0)
        baddies.append(k)
        for l in range(len(fils)):
            q.write("%s %s 0 \n" %(sdssid,fils[l]))
        continue
    hostamp.append(decomp_info['host_amp'])
    hostmag.append(decomp_info['host_mag'])
    decomps.append(decomp_info)
    c2.append(c2mask)
    c2psf.append(c2maskpsf)
    rc21.append(decomp_info['redu_Chisq'])
    rc22.append(decomp_info_psf['redu_Chisq'])
    q.write("%s I %f\n" %(sdssid,decomp_info['host_mag']))
    filters=['G','R','Z','Y']
#    locs=[j+1,j+2,j+3,j+4]
#    continue

    for f in range(len(filters)):
        print(filters[f])
#        startpsf1=str(locs[f])+'-p'
#        startfit1=str(locs[f])+'-c'
        for i in os.listdir(pathpsf+filters[f]+'/'):
            if os.path.isfile(os.path.join(pathpsf+filters[f]+'/',i)) and i.startswith(startpsf):
                hsc_psf = pathpsf+filters[f]+'/'+i
        for c in os.listdir(pathfit+filters[f]+'/'):
            if os.path.isfile(os.path.join(pathfit+filters[f]+'/',c)) and c.startswith(startfit):
                hsc_fits = pathfit+filters[f]+'/'+c
        decomp_info_1 = d1(hsc_fits, hsc_psf, False, False, False,sdssid,iresults,filters[f])
        hostamp.append(decomp_info_1['host_amp'])
        hostmag.append(decomp_info_1['host_mag'])
        decomps.append(decomp_info_1)
        q.write("%s %s %f\n" %(sdssid,filters[f],decomp_info_1['host_mag']))

        
    path2='/Users/jasminwashington/Google Drive File Stream/My Drive/usrp/git_repo/hostgal/py_tools/targets/shortlist/'+sdssid+'/'
    g_name = get_pkg_data_filename(path2+sdssid+'_G.fits')
    r_name = get_pkg_data_filename(path2+sdssid+'_R.fits')
    i_name = get_pkg_data_filename(path2+sdssid+'_I.fits')
    z_name = get_pkg_data_filename(path2+sdssid+'_Z.fits')
    y_name = get_pkg_data_filename(path2+sdssid+'_Y.fits')
    g = fits.open(g_name)[0].data
    r = fits.open(r_name)[0].data
    i = fits.open(i_name)[0].data
    z = fits.open(z_name)[0].data
    y = fits.open(y_name)[0].data

    g_name_all = get_pkg_data_filename(path2+sdssid+'_G_all.fits')
    r_name_all = get_pkg_data_filename(path2+sdssid+'_R_all.fits')
    i_name_all = get_pkg_data_filename(path2+sdssid+'_I_all.fits')
    z_name_all = get_pkg_data_filename(path2+sdssid+'_Z_all.fits')
    y_name_all = get_pkg_data_filename(path2+sdssid+'_Y_all.fits')
    g_all = fits.open(g_name_all)[0].data
    r_all = fits.open(r_name_all)[0].data
    i_all = fits.open(i_name_all)[0].data
    z_all = fits.open(z_name_all)[0].data
    y_all = fits.open(y_name_all)[0].data


    irg_default = make_lupton_rgb(i, r, g, Q=10, stretch=0.5, filename=path2+sdssid+'_irg.png')
    yzi_default = make_lupton_rgb(y, z, i, Q=5,stretch=1, filename=path2+sdssid+'_yzi.png')

    irg_default_all = make_lupton_rgb(i_all, r_all, g_all, Q=10, stretch=0.5, filename=path2+sdssid+'_irg_all.png')
    yzi_default_all = make_lupton_rgb(y_all, z_all, i_all, Q=5,stretch=1, filename=path2+sdssid+'_yzi_all.png')

    rows = 2
    cols = 2
    axes=[]
    fig=plt.figure()

    ims=[irg_default,irg_default_all,yzi_default,yzi_default_all]
    titles=['irg w/o quasar','irg w/ quasar','yzi w/o quasar','yzi w/quasar']
    for a in range(rows*cols):
        b = ims[a]
        axes.append( fig.add_subplot(rows, cols, a+1) )
        subplot_title=titles[a]
        axes[-1].set_title(subplot_title)  
        plt.suptitle(sdssid+", z="+str(np.round(zshift[k],2)),size=16)
        plt.imshow(b, origin='lower')
#    fig.tight_layout()
    fig.savefig(path2+sdssid+'_square.png')
#    plt.show()

    #k+=1

q.close()
# In[3]:


print(baddies)
#print(name)
#print(rc2)
#print(rc21)
#print(rc2psf)
#print(rc22)
#print(hostamp)
print(hostmag)
#print(decomps)

gicolor,ricolor,izcolor,iycolor,grcolor,zycolor,gzcolor,gycolor,rzcolor,rycolor=colors(hostmag,zshift,name,baddies)

fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(16,12))
#fig, ax1=plt.subplots(figsize=(16,8))
plt.rc('font',family='Times New Roman')
plt.suptitle("Host Color as Function of Redshift (z)",size=16)
ax1.scatter(grcolor[0],grcolor[1],label='g-r')
ax2.scatter(ricolor[0],ricolor[1],label='r-i')
ax3.scatter(izcolor[0],izcolor[1],label='i-z')
ax4.scatter(zycolor[0],zycolor[1],label='z-y')
#ax2.scatter(grcolor[0],grcolor[1],label='g-r')
#ax2.scatter(zycolor[0],zycolor[1],label='z-y')
#ax2.scatter(gzcolor[0],gzcolor[1],label='g-z')
#ax2.scatter(gycolor[0],gycolor[1],label='g-y')
#ax2.scatter(rzcolor[0],rzcolor[1],label='r-z')
#ax2.scatter(rycolor[0],rycolor[1],label='r-y')

#ax1.legend()
#ax2.legend()
ax1.set_title("g-r, %i sources"%(len(grcolor[0])))
ax2.set_title("r-i, %i sources"%(len(ricolor[0])))
ax3.set_title("i-z, %i sources"%(len(izcolor[0])))
ax4.set_title("z-y, %i sources"%(len(zycolor[0])))
ax3.set_xlabel("z")
ax4.set_xlabel("z")
ax1.set_ylabel("color [mag]")
ax3.set_ylabel("color [mag]")
ax1.set_ylim(top=3,bottom=-0.5)
fig.savefig("colorfromz0727.png",bbox_inches='tight')
#ax2.set_ylabel("color [mag]")
#plt.ylim(top=6.0)
#for i in range (len(izcolor[0])):
#    plt.text(izcolor[0]+0.00025,izcolor[1][i],name[i])
#    plt.text(zshift[i]+0.00025,ricolor[i],name[i])


# In[ ]:


#print(iresults)
#print(len(iresults))


# # THE SINGLE GOOD BOY 

# In[ ]:


#The fits file and the psf file

#Image file
hsc_fits_file = '/Users/jasminwashington/Google Drive File Stream/My Drive/usrp/git_repo/hostgal/py_tools/targets/cutout-HSC-I-8765-s19a_wide-200630-202731.fits'
#PSF File
hsc_psf_file = '/Users/jasminwashington/Google Drive File Stream/My Drive/usrp/git_repo/hostgal/py_tools/targets/psf-calexp-s19a_wide-HSC-I-8765-0,2-34.96731--4.15553.fits'



# doing the decomposition using the function 'decomp'

#goes image file, psf file, deepseed(set as True to put more seed and steps to fit. don't worry about it, you'll always leave it false), 

#fix_center (incase lenstronomy is messing up the fit and you want to fix the center), you can usually leave it false though, 

#runMCMC(stands for Markov chain Monte Carlo)
#Will remain false, but this article is interesting if you want to learn more about it:
#https://astrobites.org/2012/02/20/code-you-can-use-the-mcmc-hammer/

#last one is name. you can edit the decomp.py file to make it so you don't need to enter a name to run,
#but I found it helped me w my bookkeeping, since when you download a file off of the HSC databse, they don't
#include the name, ra, or dec with that info- you have to sort it yourself.
decomp_info,iresults,c2mask = decomp(hsc_fits_file, hsc_psf_file, False, False, False, '021952.15-040919.9')
decomp_info_psf,c2maskpsf = dpsf(hsc_fits_file, hsc_psf_file, False, False, False, '021952.15-040919.9')

hostamp1=([])
hostmag1=([])
decomps1=([])
c21=([])
c2psf1=([])
rc211=([])
rc221=([])

hostamp1.append(decomp_info['host_amp'])
hostmag1.append(decomp_info['host_mag'])
decomps1.append(decomp_info)
c21.append(c2mask)
c2psf1.append(c2maskpsf)
rc211.append(decomp_info['redu_Chisq'])
rc221.append(decomp_info_psf['redu_Chisq'])


#print(iresults)


# In[ ]:


print("021952.15-040919.9: %f %f %f\n" %(c21[0],c2psf1[0],abs(c21[0]-c2psf1[0])))


# In[ ]:


#PSF File

path='/Users/jasminwashington/Google Drive File Stream/My Drive/usrp/git_repo/hostgal/py_tools/targets/'

filters=['G','R','Z','Y']

for band in filters:
    print(band)
    hsc_fits = path+'cutout-HSC-'+band+'-8765-s19a_wide-200701.fits'
    hsc_psf = path+'psf-calexp-s19a_wide-HSC-'+band+'-8765-0,2-34.96731--4.15553.fits'
    decomp_info_1 = d1(hsc_fits, hsc_psf, False, False, False, '021952.15-040919.9',iresults,band)
    hostamp1.append(decomp_info_1['host_amp'])
    hostmag1.append(decomp_info_1['host_mag'])
    decomps1.append(decomp_info_1)


# In[ ]:


# Read in the three images downloaded from here:
path1='/Users/jasminwashington/Google Drive File Stream/My Drive/usrp/git_repo/hostgal/py_tools/targets/021952.15-040919.9/'

g_name = get_pkg_data_filename(path1+'021952.15-040919.9_G.fits')
r_name = get_pkg_data_filename(path1+'021952.15-040919.9_R.fits')
i_name = get_pkg_data_filename(path1+'021952.15-040919.9_I.fits')
z_name = get_pkg_data_filename(path1+'021952.15-040919.9_Z.fits')
y_name = get_pkg_data_filename(path1+'021952.15-040919.9_Y.fits')
g = fits.open(g_name)[0].data
r = fits.open(r_name)[0].data
i = fits.open(i_name)[0].data
z = fits.open(z_name)[0].data
y = fits.open(y_name)[0].data

g_name_all = get_pkg_data_filename(path1+'021952.15-040919.9_G_all.fits')
r_name_all = get_pkg_data_filename(path1+'021952.15-040919.9_R_all.fits')
i_name_all = get_pkg_data_filename(path1+'021952.15-040919.9_I_all.fits')
z_name_all = get_pkg_data_filename(path1+'021952.15-040919.9_Z_all.fits')
y_name_all = get_pkg_data_filename(path1+'021952.15-040919.9_Y_all.fits')
g_all = fits.open(g_name_all)[0].data
r_all = fits.open(r_name_all)[0].data
i_all = fits.open(i_name_all)[0].data
z_all = fits.open(z_name_all)[0].data
y_all = fits.open(y_name_all)[0].data

irg_default = make_lupton_rgb(i, r, g, Q=10, stretch=0.5, filename=path1+'021952.15-040919.9_irg.png')
yzi_default = make_lupton_rgb(y, z, i, Q=5, stretch=1, filename=path1+'021952.15-040919.9_yzi.png')

irg_default_all = make_lupton_rgb(i_all, r_all, g_all, Q=10, stretch=0.5, filename=path1+'021952.15-040919.9_irg_all.png')
yzi_default_all = make_lupton_rgb(y_all, z_all, i_all, Q=5, stretch=1, filename=path1+'021952.15-040919.9_yzi_all.png')

#plt.imshow(yzi_default, origin='lower')


# In[ ]:


#aplpy.make_rgb_image([path1+'021952.15-040919.9I.fits', path1+'021952.15-040919.9R.fits',path1+'021952.15-040919.9G.fits'],path1+'021952.15-040919.9_rgb.png')
#f = aplpy.FITSFigure(path1+'021952.15-040919.9_rgb.png')
#f.show_rgb()
#print(hostamp)
#print(hostmag)
#print(decomps)

#width=5
#height=5
rows = 2
cols = 2
axes=[]
fig=plt.figure()

ims=[irg_default,irg_default_all,yzi_default,yzi_default_all]
titles=['irg w/o quasar','irg w/ quasar','yzi w/o quasar','yzi w/quasar']
for a in range(rows*cols):
    b = ims[a]
    axes.append( fig.add_subplot(rows, cols, a+1) )
    subplot_title=titles[a]
    axes[-1].set_title(subplot_title)
    plt.suptitle('021952.15-040919.9, z=0.63',size=16)
    plt.imshow(b, origin='lower')
#fig.suptitle("test")
#fig.tight_layout()
fig.savefig(path1+'021952.15-040919.9_square.png')
#plt.show()


# In[ ]:


#gi=hostmag[t+1]-hostmag[t]
#zshift1=0.63
#plt.scatter(zshift1,gi)


# In[ ]:





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

def colors(hostmag,zshift,name):
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
        if hostmag[t]==0: continue
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


q=open("hostmag.dat","r")
hq=q.readline()
nlines=sum(1 for line in open("hostmag.dat"))-1

hostmag=np.zeros(nlines)
e=0
for line in q:
    line = line.strip()
    columns = line.split()
    hostmag[e]=columns[2]
    e+=1
    
gic,ric,izc,iyc,grc,zyc,gzc,gyc,rzc,ryc=colors(hostmag,zshift,name)


ri=open("ricolor.dat","r")
iz=open("izcolor.dat","r")
gr=open("grcolor.dat","r")
zy=open("zycolor.dat","r")


headerri=ri.readline()
headeriz=iz.readline()
headergr=gr.readline()
headerzy=zy.readline()

nsourcesgr = sum(1 for line in open("grcolor.dat"))-1
nsourcesri = sum(1 for line in open("ricolor.dat"))-1
nsourcesiz = sum(1 for line in open("izcolor.dat"))-1
nsourceszy = sum(1 for line in open("zycolor.dat"))-1

print(nsourcesgr)

ricolor=np.zeros((2,nsourcesri))
izcolor=np.zeros((2,nsourcesiz))
grcolor=np.zeros((2,nsourcesgr))
zycolor=np.zeros((2,nsourceszy))


i=0
for line in ri:
    line = line.strip()
    columns = line.split()
    ricolor[0][i]=columns[0]
    ricolor[1][i]=columns[1]
    i = i+1

j=0
for line in gr:
    line = line.strip()
    columns = line.split()
    grcolor[0][j]=columns[0]
    grcolor[1][j]=columns[1]
    j = j+1

k=0
for line in iz:
    line = line.strip()
    columns = line.split()
    izcolor[0][k]=columns[0]
    izcolor[1][k]=columns[1]
    k = k+1

l=0
for line in zy:
    line = line.strip()
    columns = line.split()
    zycolor[0][l]=columns[0]
    zycolor[1][l]=columns[1]
    l = l+1

gr.close()
ri.close()
iz.close()
zy.close()

print(grcolor[0])

fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(16,12))
#fig, ax1=plt.subplots(figsize=(16,8))
plt.rc('font',family='Times New Roman')
plt.suptitle("Host Color as Function of Redshift (z)",size=18)
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
ax1.set_title("g-r, %i sources"%(len(grcolor[0])),size=18)
ax2.set_title("r-i, %i sources"%(len(ricolor[0])),size=18)
ax3.set_title("i-z, %i sources"%(len(izcolor[0])),size=18)
ax4.set_title("z-y, %i sources"%(len(zycolor[0])),size=18)
ax1.set_xlabel("z")
ax2.set_xlabel("z")
ax3.set_xlabel("z")
ax4.set_xlabel("z")
ax1.set_ylabel("color [mag]")
ax2.set_ylabel("color [mag]")
ax3.set_ylabel("color [mag]")
ax4.set_ylabel("color [mag]")
ax1.set_ylim(top=3,bottom=-0.5)
ax2.set_ylim(top=3,bottom=-0.5)
ax3.set_ylim(top=3,bottom=-0.5)
ax4.set_ylim(top=3,bottom=-0.5)
fig.savefig("colorfromz0728.png",bbox_inches='tight')
#ax2.set_ylabel("color [mag]")
#plt.ylim(top=6.0)
#for i in range (len(izcolor[0])):
#    plt.text(izcolor[0]+0.00025,izcolor[1][i],name[i])
#    plt.text(zshift[i]+0.00025,ricolor[i],name[i])

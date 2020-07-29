#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:39:27 2017

@author: dartoon
"""
import numpy as np
def expend_grid(p,num=10):
    p = np.concatenate([p,np.zeros([num,len(p.T)])])
    p = np.concatenate([p,np.zeros([len(p),num])],axis=1) #expand the array
    return p   
def block(image, shape,factor=None):
    sh = shape[0],image.shape[0]//shape[0],shape[1],image.shape[1]//shape[1]
    return image.reshape(sh).mean(-1).mean(1)*factor**2
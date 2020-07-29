#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 10:50:18 2017

@author: dxh
"""
import numpy as np
def roundme(dicts, prec=3):
    if isinstance(dicts,float):
        dicts= round(dicts, prec)
        return dicts
    if isinstance(dicts, np.ndarray):
        dicts=np.around(dicts,prec)
        return dicts
    if isinstance(dicts, dict):
        for k, v in dicts.items():
            if isinstance(v,float):
                dicts[k] = round(v, prec)
            if isinstance(v,np.ndarray):
                for i in range(len(v)):
                    v[i]=round(v[i], prec)
        return dicts
    if isinstance(dicts, tuple):
        for i in range(len(dicts)):
            for k, v in dicts[i].items():
                dicts[i][k] = round(v, prec)
        return dicts
    else:
        return dicts
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 10:06:43 2020

This script is only for specifying a few variables that might change. Both rossTheoretical and rossMain will import all variables from here.

@author: Kristoffer Svendsen
"""

import numpy as np

ePair=0.274; # number of electrons generated per eV x-ray, tempereature dependent but if the camera is cooled it shouldn't change
ePerCount=21.4; # counts per electron, check table from performance report for this, depends on preamp, sensitive/capacity mode, read out rate
camResp = ePair*ePerCount;  #counts per eV

Al_layers = 8;  #number of layers Al foil, this has changed betweeen 4 and 8 between a few runs
pixelSize = 13.5e-6;    #camera pixel size [m]
r = 2.7+0.085;  # Distance source-detector [m] 
Ec = np.linspace(1e3, 25e3,100);    # critical energies to be evaluated (the more the better but also slower) [eV]
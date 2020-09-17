# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 19:11:48 2020

This is an example script to show how the xRayAnalysis is used. Just call the function with a run_name of the same form as given here. 
It will take the date of that name and import the latest calibration file. 

Be sure to change the path to the MIRAGE folder, all other pathing withing MIRAGE folder is done by xRayAnalysis, pressuming it has the same structure as the BOX. 

Process either a single image, a range of images or an entire run folder as shown below


Note: Currently there is only calibration for 2020/09/03 and onwards (the rest will be added on soon)

@author: poff
"""

import numpy as np
from xRayAnalysis import xRayAnalysis
import matplotlib.pyplot as plt
import pickle, os, sys

rossPath = os.path.dirname(os.path.realpath(__file__))
if rossPath not in sys.path:
    sys.path.append(rossPath)




#change this path to the current MIRAGE folder
BASE_PATH =r'\Users\poff\Box\GeminiXANES2020\MIRAGE'

#Define date and run name
run_name='20200907/run01'    



# %% Below follows a few examples on how to call the code, either with a single shot, a range of shots or an entire run

# #single iamge, second argument is shot number
# Ec, photonperSteradian, Ec_error, shotNumber  = xRayAnalysis(run_name, 1)  

# #alternatively put in a range, and get all the shots in this range
# Ec, photonperSteradian, Ec_error, shotNumber  = xRayAnalysis(run_name, [*range(38, 49, 1)] ) 

#If no 2nd argument is given it will process everything int the folder for that run
Ec, photonperSteradian, Ec_error, shotNumber  = xRayAnalysis(run_name) 


#This part is only to set all the critical energy to 0 for all the shots that have a large error (usually due to blank images or super low x-ray fluence)
for ind, i in enumerate(Ec_error):
    if  i > 1:
        Ec[ind]=0

# %% some plotting to make sure everything is working
plt.plot(shotNumber, np.array(Ec)*1e-3)
plt.title('critical energy')
plt.xlabel('shot number')
plt.ylabel('Energy [KeV]')


plt.figure()
plt.plot(shotNumber, Ec_error)
plt.title('error')
plt.xlabel('shot number')
plt.ylabel('a.u')


plt.figure()
plt.plot(shotNumber, photonperSteradian)
plt.title('photons per steradian')
plt.xlabel('shot number')
plt.ylabel('1/sr')
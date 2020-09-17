# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 19:11:48 2020

This is an example script.
Ross main takes an image as argument and returns critical energy, photons/steradian, residual.
The residual is just to make sure that it has converged to a somewhat acceptable solution. This value is usually lower than 0.045 (in a perfect world it would be 0).

Some common variables such as camera response and distance can be changed in rossSetup.py that all modules will import. 

@author: poff
"""





import pickle, os, sys
from PIL import Image
from rossMain import rossMain
import glob


sys.path.insert(0, "C:\\Users\poff\Documents\GitHub\GeminiXANES2020_analysis")




#change this path to the current MIRAGE folder
BASE_PATH =r'\Users\poff\Box\GeminiXANES2020\MIRAGE'


def xRayAnalysis(run_name, shotNum=0, plotting=False, debug = False):
    
    EcList=[]
    photonperSteradianList=[] 
    Ec_errorList=[]
    shotNumerList=[]
    
    
    splitDateRun = run_name.split('/')
    date = splitDateRun[0]
    
    calibList = glob.glob('rossFiltCalib*')
    
    diff=[]
    for i in calibList:
        calibPrefix, calibDate = i.split('_')
        diff.append(int(date)-int(calibDate))
        
    calibRecentInd = diff.index(min(i for i in diff if i >= 0))
    
    calibFile=calibList[calibRecentInd]
    
    
    # use date here to check for latest calib file
    # calibFile = 'rossFiltCalib_' + date

    
    
    my_dir = BASE_PATH + '\\Lundatron\\' + run_name

    if shotNum == 0:   
        dir_list = os.listdir(my_dir)
        
        for f in dir_list:      
            shotNum=f[4:7].lstrip("0")
            # print('Processing shot: ' + shotNum)
            raw = Image.open(my_dir +'\\'+ f)
            Ec, photonperSteradian, Ec_error = rossMain(raw, calibFile, plotting, debug)   
            EcList.append(Ec)
            photonperSteradianList.append(photonperSteradian)
            Ec_errorList.append(Ec_error)
            shotNumerList.append(shotNum)
            # out['shotNumber'] = shotNumber               

            
    else:        

        if isinstance(shotNum, int):
            raw = Image.open(my_dir + '\\' + 'Shot' + f"{shotNum:03d}"+'.tif')
            Ec, photonperSteradian, Ec_error = rossMain(raw, calibFile, plotting, debug)   
            EcList.append(Ec)
            photonperSteradianList.append(photonperSteradian)
            Ec_errorList.append(Ec_error)
            shotNumerList.append(shotNum)
            # outputDict['shotNumber'] = shotNum                

            
        else:          
            for ind, f in enumerate(shotNum):
                shot=f
                # print('Processing shot: ' + shotNum)
                raw = Image.open(my_dir + '\\' + 'Shot' + f"{shot:03d}"+'.tif')                
                Ec, photonperSteradian, Ec_error=rossMain(raw, calibFile, plotting, debug) 
                EcList.append(Ec)
                photonperSteradianList.append(photonperSteradian)
                Ec_errorList.append(Ec_error)
                shotNumerList.append(f)

                
    return EcList, photonperSteradianList, Ec_errorList, shotNumerList
    




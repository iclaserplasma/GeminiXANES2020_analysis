# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 09:53:32 2020

@author: poff
"""
# %% setup for data on 2020-09-03

import pickle
import numpy as np

#setup the filter vectors
filtMatSetup=['V', 'Fe', 'Co', 'Zn', 'Nb', 'Mo', 'Pd', 'Sn', 'V', 'Fe', 'Zn', 'Nb', 'Mo', 'Sn'];  #vector with all filters (counting left to rigth top to bottom
filtLeg=['V', 'Fe', 'Co', 'Zn', 'Nb', 'Mo', 'Pd', 'Sn'];   #legend vector, first name here is 1 in index etc..
filtMat_ind=np.array([1,2,3,4,5,6,7,8,1, 2, 4, 5, 6, 7]);  #indexing to get the rigth legends

ePair=0.274; # number of electrons generated per eV x-ray, tempereature dependent but if the camera is cooled it shouldn't change
ePerCount=21.4; # counts per electron, check table from performance report for this, depends on preamp, sensitive/capacity mode, read out rate
camResp = ePair*ePerCount;  #counts per eV
Al_layers = 4;  #number of layers Al foil, this has changed betweeen 4 and 8 between a few runs
r = 2.7+0.085;  # Distance source-detector [m] 

pixelSize = 13.5e-6;    #camera pixel size [m]
r = 2.7+0.085;  # Distance source-detector [m] 
Ec = np.linspace(1e3, 50e3,100);    # critical energies to be evaluated (the more the better but also slower) [eV]


w=120
wb=100

# y = np.array([134, 141, 146, 150, 158, 160, 438, 434, 438, 734, 746, 753, 753, 748, 1054, 1063, 1365, 1365])-34-w/2; 
# x = np.array([176, 408, 636, 859, 1091, 1321, 415, 637, 868, 173, 402, 647, 872, 1092, 1318, 1552, 1312, 1561])-26-w/2; 

matlab2pyOffset=20;

x = np.array([
582,
830,
1046,
352,
580,
826,
1056,
1292,
1519,
1745,
121,
1506,
1746,
117])-w/2+matlab2pyOffset; 

y = np.array([
60,
55,
60,
370,
362,
364,
374,
368,
677,
664,
985,
987,
981,
1296])-w/2+matlab2pyOffset;   


 
BGcoord=np.array([1048, 1298])-wb/2+np.array([10, 0])

#input coordinates, these will generate a meshgrid. i.e. each value represents an entire row/column
xb=np.linspace(183, 1810, 8)-w/2+matlab2pyOffset-20;
yb=np.linspace(258, 1490, 5)-w/2+matlab2pyOffset-20;

# xb=np.array([128, 361, 575, 810, 1040, 1260, 1500]) 
# yb=np.array([230, 520, 830, 1140, 1450, 1750])

#index to know which row each y value is located in...This is ugly but i couldn't think of a better way at the moment
x_ind = np.array([2, 3, 4, 1, 2, 3, 4, 5, 6, 7,0, 6,7,0])   #start counting at 0...




dict= {'x' : x,
       'y' : y,
       'w' : w,
       'BGcoord' : BGcoord,
       'rot' : 92,
       'flip' : 1,
       'camResp' : camResp,
       'Al_layers' : Al_layers,
       'xb' : xb,
       'yb' : yb,
       'x_ind' : x_ind,
       'filtMatSetup' : filtMatSetup,
       'filtMat_ind' : filtMat_ind,
       'filtLeg' : filtLeg,     
       'r' : r,
       'pixelSize' : pixelSize,    #camera pixel size [m]
       'Ec' : Ec    # critical energies to be evaluated (the more the better but also slower) [eV]
       } 



filename = 'rossFiltCalib_20200903'
outfile = open(filename,'wb')
pickle.dump(dict, outfile)
outfile.close()




# %% setup for data on 2020-09-07

import pickle
import numpy as np

#setup the filter vectors
filtMatSetup=['V', 'Fe', 'Co', 'Zn', 'Nb', 'Mo', 'Pd', 'Sn', 'V', 'Fe', 'Co', 'Zn', 'Nb', 'Mo', 'Pd', 'Sn'];  #vector with all filters (counting left to rigth top to bottom
filtLeg=['V', 'Fe', 'Co', 'Zn', 'Nb', 'Mo', 'Pd', 'Sn'];   #legend vector, first name here is 1 in index etc..
filtMat_ind=np.array([1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8]);  #indexing to get the rigth legends

ePair=0.274; # number of electrons generated per eV x-ray, tempereature dependent but if the camera is cooled it shouldn't change
ePerCount=21.4; # counts per electron, check table from performance report for this, depends on preamp, sensitive/capacity mode, read out rate
camResp = ePair*ePerCount;  #counts per eV
Al_layers = 4;  #number of layers Al foil, this has changed betweeen 4 and 8 between a few runs

w=120
wb=50

pixelSize = 13.5e-6;    #camera pixel size [m]
r = 2.7+0.085;  # Distance source-detector [m] 
Ec = np.linspace(1e3, 50e3,100);    # critical energies to be evaluated (the more the better but also slower) [eV]

# y = np.array([134, 141, 146, 150, 158, 160, 438, 434, 438, 734, 746, 753, 753, 748, 1054, 1063, 1365, 1365])-34-w/2; 
# x = np.array([176, 408, 636, 859, 1091, 1321, 415, 637, 868, 173, 402, 647, 872, 1092, 1318, 1552, 1312, 1561])-26-w/2; 

matlab2pyOffset=20;

x = np.array([772,
1001,
1231,
529,
759,
1001,
1240,
1465,
1701,
1936,
59,
290,
1699,
1932,
82,
300])-w/2+matlab2pyOffset; 

y = np.array([58,
56,
63,
364,
366,
362,
362,
362,
674,
670,
983,
984,
987,
981,
1288,
1296])-w/2+matlab2pyOffset;   


 
BGcoord=np.array([1225, 1282])-w/2+matlab2pyOffset;

#input coordinates, these will generate a meshgrid. i.e. each value represents an entire row/column
xb=np.linspace(128, 2006, 9)-w/2+matlab2pyOffset-20;
yb=np.linspace(258, 1490, 5)-w/2+matlab2pyOffset-20;

# xb=np.array([128, 361, 575, 810, 1040, 1260, 1500]) 
# yb=np.array([230, 520, 830, 1140, 1450, 1750])

#index to know which row each y value is located in...This is ugly but i couldn't think of a better way at the moment
x_ind = np.array([3, 4, 5, 2, 3, 4, 5, 6, 7, 8, 0, 1, 7, 8, 0, 1])




dict= {'x' : x,
       'y' : y,
       'w' : w,
       'BGcoord' : BGcoord,
       'rot' : 92,
       'flip' : 1,
       'camResp' : camResp,
       'Al_layers' : Al_layers,
       'xb' : xb,
       'yb' : yb,
       'x_ind' : x_ind,
       'filtMatSetup' : filtMatSetup,
       'filtMat_ind' : filtMat_ind,
       'filtLeg' : filtLeg,     
       'r' : r,
       'pixelSize' : pixelSize,    #camera pixel size [m]
       'Ec' : Ec    # critical energies to be evaluated (the more the better but also slower) [eV]
       } 



filename = 'rossFiltCalib_20200907'
outfile = open(filename,'wb')
pickle.dump(dict, outfile)
outfile.close()
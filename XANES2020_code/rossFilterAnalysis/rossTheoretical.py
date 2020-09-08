# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 19:04:13 2020

This script is treated as a a module by rossMain. It calculates the theoretical transmissions of the ross filters which rossMain imports. 
If the filters are exchanged or the position of the filters changes this script needs to be updated.
If the transport line transmission changes, this also needs to be changed in this script.

@author: Kristoffer Svendsen
"""
from XANES2020_code.rossFilterAnalysis.rossSetup import *
import numpy as np
import pandas as pd
from scipy.special import kv
import os
from pathlib import Path
from XANES2020_code import PKG_DATA
rossPath = Path(os.path.join(PKG_DATA,'rossFilterAnalysis'))


# constants
h=6.62607004e-34;
c=299792458;
e=1.602176634e-19;
E=np.linspace(10,25000,501);



# %% Filters and QE

#  Import Andor QE data
QE_data_temp=pd.read_csv(rossPath / 'ikonl_qe.txt', sep = '\t', decimal = '.', header = None); #import quantum efficiency for Ikon-L SO
QE_data_temp=QE_data_temp.values;   #extract values

# plt.plot(QE_data_temp[:,0],QE_data_temp[:,1])
QE_data = np.interp(E, QE_data_temp[:,0], QE_data_temp[:,1]);   #interpolate the quantum efficiency to match the energy vector E
# plt.plot(E,yinterp)

#import filter data in energy ranges 1 kev to 25 kev with 501 values. imports only the transmission, usecol says "Energy" but it is actually the transmission being imported
Al_foil = pd.read_csv(rossPath / 'Al_foil.txt', delim_whitespace = True, decimal = '.', header=1, usecols=['Energy']);  
Air = pd.read_csv(rossPath / 'Air_8p5cm.txt', delim_whitespace = True, decimal = '.', header=1, usecols=['Energy']);  
Be = pd.read_csv(rossPath / 'Be_250mu.txt', delim_whitespace = True, decimal = '.', header=1, usecols=['Energy']);  
Co = pd.read_csv(rossPath / 'Co_5p3mu.txt', delim_whitespace = True, decimal = '.', header=1, usecols=['Energy']); 
Fe = pd.read_csv(rossPath / 'Fe_6p4mu.txt', delim_whitespace = True, decimal = '.', header=1, usecols=['Energy']);  
filterBacking = pd.read_csv(rossPath / 'filterBacking.txt', delim_whitespace = True, decimal = '.', header=1, usecols=['Energy']);  
Kapton = pd.read_csv(rossPath / 'Kapton_150mu.txt', delim_whitespace = True, decimal = '.', header=1, usecols=['Energy']); 
Mo = pd.read_csv(rossPath / 'Mo_100mu.txt', delim_whitespace = True, decimal = '.', header=1, usecols=['Energy']);  
Nb = pd.read_csv(rossPath / 'Nb_125mu.txt', delim_whitespace = True, decimal = '.', header=1, usecols=['Energy']);  
Pd = pd.read_csv(rossPath / 'Pd_69mu.txt', delim_whitespace = True, decimal = '.', header=1, usecols=['Energy']);  
Sn = pd.read_csv(rossPath / 'Sn_90mu.txt', delim_whitespace = True, decimal = '.', header=1, usecols=['Energy']);  
V = pd.read_csv(rossPath / 'V_10p8mu.txt', delim_whitespace = True, decimal = '.', header=1, usecols=['Energy']);  
Zn = pd.read_csv(rossPath / 'Zn_4p7mu.txt', delim_whitespace = True, decimal = '.', header=1, usecols=['Energy']); 

#exract values from filters
Al_foil=Al_foil.values;
Air=Air.values;
Be=Be.values;
Co=Co.values;
Fe=Fe.values;
filterBacking=filterBacking.values;
Kapton=Kapton.values;
Mo=Mo.values;
Nb = Nb.values;
Pd = Pd.values;
Sn = Sn.values;
V = V.values;
Zn = Zn.values;

#define transport line transmission
Ttotal = Air*Be*Kapton*filterBacking*Al_foil**Al_layers;

#setup the filter vectors
# filtMat=[Co, Zn, Nb, Mo, Pd, Sn, V, Fe, Co, Zn, Nb, Mo, Pd, Sn, V, Fe, Nb, Mo];  #vector with all filters (counting left to rigth top to bottom
# filtLeg=['V', 'Fe', 'Co', 'Zn', 'Nb', 'Mo', 'Pd', 'Sn'];   #legend vector, first name here is 1 in index etc..
# filtMat_ind=np.array([3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 5, 6]);  #indexing to get the rigth legends

filtMat=[V, Fe, Co, Zn, Nb, Mo, Pd, Sn, V, Fe, Co, Zn, Nb, Mo, Pd, Sn];  #vector with all filters (counting left to rigth top to bottom
filtLeg=['V', 'Fe', 'Co', 'Zn', 'Nb', 'Mo', 'Pd', 'Sn'];   #legend vector, first name here is 1 in index etc..
filtMat_ind=np.array([1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]);  #indexing to get the rigth legends


# %% Ross filter (Theoretical)

dNdE=[];

#Photon number distribution at critical energy, i, (synchrotron-like)
for i in Ec:
    dNdE.append( (E/i)**2 * kv(2/3, (E/i)/2)**2 /np.trapz(  (E/i)**2 *kv(2/3, (E/i)/2)**2 ,E)  );
    
#convert to np arrays and squeeze the empty dimensions
dNdE=np.array(dNdE);
dNdE=np.squeeze(dNdE)
E=np.array(E);
QE_data=np.array(QE_data);
Ttotal=np.array(Ttotal);
Ttotal=np.squeeze(Ttotal)

# calculate theoretical transmission coefficients

Tth = np.zeros( (len(Ec), len(filtMat)));   #vector for theoretical transmission WITH filters for all critical energies 
Cth0=np.zeros(len(Ec));     #vector for theoretical counts WITHOUT filters for all critical energies 
Cth=np.zeros( (len(Ec), len(filtMat))) ;    #vector for theoretical counts WITH filters for all critical energies 

for n in range(0,len(Ec)):   #looping over all critical energies
    Cth0[n] =  np.trapz (dNdE[n]*E*Ttotal*QE_data/camResp, E)    # Theoretical count without filters for critical energy-i0
    for j in range(0,len(filtMat)):  #looping over each filter
            Cth[n,j] = np.trapz(dNdE[n,:]*E*Ttotal*np.squeeze(filtMat[j])*QE_data/camResp , E);    #Theoretical count for critical energy-i0 and filter-j0
            Tth[n,j] = Cth[n,j] / Cth0[n];      # Theoretical transmission for critical energy-i0 and filter-j0    

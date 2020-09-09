import numpy as np
import scipy as sci
from scipy.optimize import minimize

from scipy.interpolate import interp1d
from scipy.special import kv
import pickle

def load_object(filename):
    with open(filename, 'rb') as fid:
        return pickle.load(fid)

class fRegions(object):
    def __init__(self, label =0, material ='none', d =0, dErr =0, E_keV = [], mu = [], rho = 0,T_E = 0,fTrans=0,fTrans_rms=0):
        self.label = label
        self.material = material
        self.d = d
        self.dErr = dErr
        self.E_keV = E_keV
        self.mu = mu
        self.rho = rho
        self.T_E = T_E
        self.fTrans = fTrans
        self.fTrans_rms = fTrans_rms
        
    def calcCentroid(self,filterMask):
        [yInd,xInd] = np.where(filterMask==self.label)
        yC = np.mean(yInd)
        xC = np.mean(xInd)
        return xC,yC
    
    def calcTrans(self,imgT,filterMask):
        [yInd,xInd] = np.where(filterMask==self.label)
        self.fTrans = np.mean(imgT[yInd,xInd])
        self.fTrans_rms = np.sqrt(np.mean((self.fTrans-imgT[yInd,xInd])**2))
        # self.fTrans_std_err = self.fTrans_rms/np.sqrt(np.sum(filterMask==self.label))
        tLabel = np.mean(filterMask[yInd,xInd])
        return self.fTrans, self.fTrans_rms, tLabel

    def calcT_E(self):
        self.T_E =  np.exp(-self.rho*self.mu*self.d*1e-4)
        return self.T_E

    def calcTransCumDist(self):
        nT = 1000
        tAxis = np.linspace(-10,10,num=nT)*self.fTrans_rms+self.fTrans
        fTransExp = np.exp(-(tAxis-self.fTrans)**2/(2*self.fTrans_rms**2))
        fTransExp = fTransExp/sci.integrate.trapz(fTransExp,x=tAxis)
        tTransExp_cum = sci.integrate.cumtrapz(fTransExp,x=tAxis,initial=0)
        [yCS,ia] = np.unique(tTransExp_cum, return_index=True)
        
        self.tAxisCumDist = yCS
        self.tAxis = tAxis[ia]

        self.randomTransFunc()

        return self.tAxis, self.tAxisCumDist
    
    def randomTransFunc(self):
        f = interp1d(self.tAxisCumDist,self.tAxis)
        self.calcModifiedTrans = f
        return f

class Mask(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value

class Trans(object):
    def __init__(self, name, E_keV, T_E):
        self.name = name
        self.E_keV = E_keV
        self.T_E = T_E

class Beam(object):
    def __init__(self, original_file, beamCounts):
        self.original_file = original_file
        self.beamCounts = beamCounts


x = np.logspace(-5,2,num=1000,endpoint=True)
y = (x*kv(2/3,x))**2
I_Xi = interp1d(x,y,bounds_error=False,fill_value=0)

class Betatron_spec_fitter():
    def __init__(self,transmission_img,filter_obj,mask_obj):
        self.filter_obj = filter_obj
        self.transmission_img = transmission_img
        self.mask_obj = mask_obj
        self._load_info()
        self.measure_transmission()

    def _load_info(self):
        transFuns = self.filter_obj['transmission_functions']
        self.fList = self.filter_obj['filter_fRegions']
        self.E_keV = transFuns[0].E_keV
        self.aQE = transFuns[0].T_E
        self.nE = np.size(self.E_keV)
        self.transMat = transFuns[2].T_E
        self.nullTrans = transFuns[1].T_E

    def measure_transmission(self):
        for fReg in self.fList:
            fReg.calcTrans(self.transmission_img,self.mask_obj['filter_number_regions'])
            fReg.calcTransCumDist()
        self.measured_trans = np.array([x.fTrans for x in self.fList])
        self.measured_trans_rms = np.array([x.fTrans_rms for x in self.fList])
        self.filter_number = [x.label for x in self.fList]

    def theoretical_trans(self,E_c):
        S = 1.0*I_Xi(self.E_keV/(2*E_c))
        S = S/np.sum(S)
        null_integrand = S*self.aQE*self.nullTrans
        
        integrand = null_integrand[:,np.newaxis] * self.transMat
        betatron_signal = np.sum(integrand,axis=0)
        null_signal = np.sum(null_integrand,axis=0)

        total_signal = betatron_signal/null_signal
        return total_signal
        
    def err_func(self,E_c):
        err = self.theoretical_trans(E_c) - self.measured_trans
        return np.sqrt(np.mean(err**2))
    
    def calc_E_crit(self):   
        res = minimize(self.err_func,(10),method='Nelder-Mead', tol=1e-4)
        E_c = res.x[0]
        self.trans_pred = self.theoretical_trans(E_c)
        return E_c, self.trans_pred
        
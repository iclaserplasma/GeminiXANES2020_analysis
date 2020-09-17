import numpy as np
import scipy as sci
from scipy.optimize import minimize

from scipy.interpolate import interp1d
from scipy.special import kv
import pickle

from XANES2020_code.general_tools import load_object

class fRegions(object):
    """ Class used for each filter number to store material, thicnkess and transmission functions
    is initialised when specifying the filter and has methods to calculate the theoretical transmission
    as well as to measure the transmission of a given image and associated labelled mask
    label (int) is a number which is used to identify the filter in a labelled mask
    d (float) is the material thickness in microns
    dErr (float) is the error on that thickness
    E_keV (array) is the energy axis used for calculations
    mu (array) is the photon attentuation coefficient as a function of E_keV in cm^2/g
    rho (float) is the density in g/cm^3
    T_E (array) is the transmission as a function of E_keV and can be calculated from the method calcT_E
    fTrans (float) is the average tranmsission of the filter, can be calculated with calcTrans
    fTrans_rms (float) is the error on fTrans, also calculated with the method calcTrans
    """
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
        """ Find the x,y centroid of the filter using filterMask
        """
        [yInd,xInd] = np.where(filterMask==self.label)
        yC = np.mean(yInd)
        xC = np.mean(xInd)
        return xC,yC
    
    def calcTrans(self,imgT,filterMask):
        """ Get the mean transmission (and RMS) of the pixels of imgT specified by where filterMask matches the label
            Not sure if RMS is the relavent error as we would really like the error on the tranmission measurment
        """
        f_sel = filterMask==self.label
        self.fTrans = np.mean(imgT[f_sel])
        self.fTrans_rms = np.sqrt(np.mean((self.fTrans-imgT[f_sel])**2))
        tLabel = np.mean(filterMask[f_sel])
        return self.fTrans, self.fTrans_rms, tLabel

    def calcT_E(self):
        """ Calculate the theoretical transmission of the filter as a function of E_keV
        """
        self.T_E =  np.exp(-self.rho*self.mu*self.d*1e-4)
        return self.T_E

    def calcTransCumDist(self):
        """ Used for monte-carlo error analysis. Usesd the measurement error set up a statistical model for the measurements
        """
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
        """ Get a randomised transmission value from the distribution from calcTransCumDist
        """
        f = interp1d(self.tAxisCumDist,self.tAxis)
        self.calcModifiedTrans = f
        return f

## These classes were a bad idea, but it's a bit late to change them now.
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

# interpolate bessel function rather than calculate each time for speed
x = np.logspace(-5,2,num=1000,endpoint=True)
y = (x*kv(2/3,x))**2
I_Xi = interp1d(x,y,bounds_error=False,fill_value=0)

class Betatron_spec_fitter():
    """ Class for fitting tranmsission values to get E_crit
    transmission_img (2D array) are values ideally between 0 and 1 of relative transmission of beam
    filter_obj is an dictionary created by a calibration which contains information about each filter 
        and also about the other material in the  beamline
    filter_obj = {
    'transmission_functions': trans_funcs,
    'filter_arrangement': filter_arangement,
    'synthetic_filters': fake_filters,
    'synthetic_transmission': fake_trans,
    'filter_fRegions': fList,
    'null_fRegions': nullList,
    }

    mask_obj is a dictionary containing details of where the beam and filter regions are located on the image
    mask_obj = {
    'beam_mask':beam_mask_final,
    'filter_label_regions': labelled,
    'filter_number_regions': img_filters,
    'filter_label': img_filter_labels,
    'filter_number':  img_filter_numbers,    
    }
    """
    def __init__(self,transmission_img,filter_obj,mask_obj):
        self.filter_obj = filter_obj
        self.transmission_img = transmission_img
        self.mask_obj = mask_obj
        self._load_info()
        self.measured_trans = None

    def _load_info(self):
        transFuns = self.filter_obj['transmission_functions']
        self.fList = self.filter_obj['filter_fRegions']
        self.fList_filter_number = [x.label for x in self.fList]
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
        

    def measure_tranmission_per_block(self):
        filter_label_regions = self.mask_obj['filter_label_regions']
        filter_labels = self.mask_obj['filter_label']
        filter_number = self.mask_obj['filter_number']
        f_trans = []
        f_trans_rms = []
        for f_label in filter_labels:
            f_trans.append(np.mean(self.transmission_img[filter_label_regions==f_label]))
            f_trans_rms.append(np.std(self.transmission_img[filter_label_regions==f_label]))
        self.block_trans = np.array(f_trans)
        self.block_trans_rms = np.array(f_trans_rms)
        self.block_filter_number = filter_number

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
        theory_trans = self.theoretical_trans(E_c) 
        if self.by_block:
            theory_trans = theory_trans[self.block_filter_number-1]
            err = theory_trans - self.block_trans

            if self.f_sel is not None:
                f_test = []
                for n in self.block_filter_number:
                    f_test.append(np.any(self.f_sel==n))
                err = err[f_test]
        else:
            err = self.theoretical_trans(E_c) - self.measured_trans
            
            if self.f_sel is not None:
                err = err[self.f_sel]
        return np.sqrt(np.mean(err**2))
    
    def calc_E_crit(self,by_block=False,f_sel=None):   
        self.by_block = by_block
        self.f_sel = f_sel
        if by_block:
            self.measure_tranmission_per_block()
        else:
            self.measure_transmission()
        res = minimize(self.err_func,(10),method='Nelder-Mead', tol=1e-4)
        E_c = res.x[0]
        self.trans_pred = self.theoretical_trans(E_c)
        return E_c, self.trans_pred
        
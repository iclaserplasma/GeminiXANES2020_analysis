import numpy as np
import cv2, pickle
import scipy.signal as sig
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.interpolate import interp1d

from XANES2020_code import Espec
class ESpec_high_proc():
    """ Object for handling espec_high analysis
        hardcoded fC_per_count value from rough calibration by Matt on 28th August 2020
        args 
            tForm_filepath is the path of the image warp cv2 perspective transform
                contains x_mm and y_mm which are the real spatial axes of the un-warped image
            Espec_cal_filepath is the conversion from x_mm to energy in MeV 
        kwargs:
            img_bkg = can be an espec_high image or a single value or None to disable this subtraction method
            use_median = boolean, if True then the median value will be subtracted from the raw image (crude background noise subtraction)
            kernel_size = (int,int) or None: used for median 2d filter 
    """
    fC_per_count = 0.0012019877034770447
    def __init__(self,tForm_filepath,Espec_cal_filepath,img_bkg=None,use_median=True,kernel_size=None ):
        # warping info
        tForm = pickle.load(open(tForm_filepath,'rb'))
        self.imgArea0 = tForm['imgArea0']
        self.H = tForm['H']
        self.newImgSize = tForm['newImgSize']
        self.imgArea1 = tForm['imgArea1']
        self.screen_x_mm = tForm['x_mm']
        self.screen_y_mm = tForm['y_mm']
        self.screen_dx = np.mean(np.diff(self.screen_x_mm))
        self.screen_dy = np.mean(np.diff(self.screen_y_mm))
        # dispersion calibration
        Espec_cal = loadmat(Espec_cal_filepath)
    
        self.spec_x_mm = Espec_cal['spec_x_mm'].flatten()
        self.spec_MeV = Espec_cal['spec_MeV'].flatten()
        iSel = np.isfinite(self.spec_MeV)
        self.spec_cal_func = interp1d(self.spec_x_mm[iSel], self.spec_MeV[iSel], 
            kind='linear', copy=True, bounds_error=False, fill_value=0)
        
        self.screen_energy = self.spec_cal_func(self.screen_x_mm)
        with np.errstate(divide='ignore'):
            with np.errstate(invalid='ignore'):
                g = -np.gradient(self.screen_x_mm,self.screen_energy)
                g[np.isfinite(g)<1]=0
        self.dispersion = g
        # background subtraction options
        self.img_bkg=img_bkg
        self.use_median=use_median
        self.kernel_size=kernel_size

        # energy axis for final spectra
        self.eAxis_MeV = np.linspace(300,1500,num=600)
        self.dE_MeV = np.mean(np.diff(self.eAxis_MeV))
        

    def espec_warp(self,img_raw):
        """ calc transformed image using tForm file and cv2 perspective transform
        """ 
        img = self.espec_background_sub(img_raw)

        with np.errstate(divide='ignore'):
            with np.errstate(invalid='ignore'):
                imgCountsPerArea = img/self.imgArea0
        imgCountsPerArea[self.imgArea0==0] =0
        imgCountsPerArea[np.isinf(imgCountsPerArea)] = 0
        imgCountsPerArea[np.isnan(imgCountsPerArea)] = 0

        im_out = cv2.warpPerspective(imgCountsPerArea, self.H, self.newImgSize)*self.imgArea1
        return im_out

    def espec_file2sceen(self,file_path):
        """ Takes a data file and returns the screen signal (using perspective transform)
        should be in real units of pC per mm^2
        """
        img_raw = plt.imread(file_path)
        img_pC_permm2 = self.espec_data2screen(img_raw)
        return img_pC_permm2

    def espec_data2screen(self,img_raw):
        """ Takes raw data (previous opened from data file) and returns the screen signal (using perspective transform)
        should be in real units of pC per mm^2
        """

        img_warp= self.espec_warp(img_raw)
        img_pC_permm2 = img_warp*self.fC_per_count/self.imgArea1 *1e-3
        return img_pC_permm2

    def espec_background_sub(self,img_raw):
        """ background subtraction method
        """
        if self.img_bkg is None:
            img_sub = img_raw
        else:
            img_sub = img_raw-self.img_bkg
        
        if self.use_median:
            img_sub = img_sub -np.median(img_sub)

        if self.kernel_size is not None:
            img_sub = sig.medfilt2d(img_sub,kernel_size=self.kernel_size)

        return img_sub

    def espec_screen2spec(self,img_screen):
        """ convert image to spectrum
        Uses 1d interpolation along horrizontal axis (1)
        """
        spec = img_screen*self.dispersion
        spec_func = interp1d(self.screen_energy,spec, bounds_error=False, fill_value=0)
        spec_pC_per_mm_per_MeV = spec_func(self.eAxis_MeV)
        return spec_pC_per_mm_per_MeV

    def total_charge(self,img_raw):
        """ Integrates the screen image to get the total charge
        """
        img_pC_permm2 = self.espec_data2screen(img_raw)
        return np.sum(img_pC_permm2)*self.screen_dx*self.screen_dy
    def energy_spectrum(self,img_raw):
        img_pC_permm2 = self.espec_data2screen(img_raw)
        spec_pC_per_mm_per_MeV = self.espec_screen2spec(img_pC_permm2)
        spec_J_per_mm_per_MeV = self.eAxis_MeV*spec_pC_per_mm_per_MeV
        return spec_J_per_mm_per_MeV

    def total_beam_energy(self,img_raw):
        """ Integrates the spectrum to get total beam energy
        """
        img_pC_permm2 = self.espec_data2screen(img_raw)
        spec_pC_per_mm_per_MeV = self.espec_screen2spec(img_pC_permm2)
        W_b = np.sum(np.sum(spec_pC_per_mm_per_MeV,axis=0)*self.screen_dy*self.dE_MeV *self.eAxis_MeV*1e-12*1e6)
        return W_b

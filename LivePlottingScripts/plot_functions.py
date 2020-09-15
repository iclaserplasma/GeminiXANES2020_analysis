import sys, os, pickle, time
import numpy as np
from scipy.interpolate import interp1d
from skimage.io import imread
from PIL import Image
from XANES2020_code.paths import DATA_FOLDER, CAL_DATA
from XANES2020_code.Espec import espec_processing 
from XANES2020_code.general_tools import choose_cal_file, load_object
from XANES2020_code.Betatron_analysis.beam_fitting import function_beam_fitter
from XANES2020_code.Betatron_analysis import xray_analysis as xray
from skimage.measure import label, regionprops

CAL_DATA = r'C:\Users\CLFUser\Documents\ProcessedCalibrations'

def xray_andor_image(path):
    return np.array(Image.open(path))  

def image_mean(path):
    return np.mean(imread(path))


class Espec_high_proc:
    def __init__(self,run_name,shot_num=1,img_bkg=None):
        self.run_name = run_name
        self.shot_num = shot_num
        self.img_bkg = img_bkg
        self.setup_proc(run_name=run_name,shot_num=shot_num,img_bkg=img_bkg)

    def setup_proc(self,run_name=None,shot_num=None,img_bkg=None):
        if run_name is None:
            run_name = self.run_name
        if shot_num is None:
            shot_num = self.shot_num
        if img_bkg is None:
            img_bkg = self.img_bkg

        # paths for image warp and dispersion calibrations files
        tForm_filepath = choose_cal_file(run_name,shot_num,'Espec_high',
                        'Espec_high_transform',cal_data_path=CAL_DATA)
        Espec_cal_filepath = choose_cal_file(run_name,shot_num,'Espec_high',
                        'Espec_high_disp_cal',cal_data_path=CAL_DATA)

        # setup espec processor
        eSpec_proc = espec_processing.ESpec_high_proc(tForm_filepath,Espec_cal_filepath,
                             img_bkg=img_bkg,use_median=True,kernel_size=None )
        self.eSpec_proc = eSpec_proc

        # image axes
        self.x_mm = eSpec_proc.screen_x_mm
        self.dx = np.mean(np.diff(self.x_mm))
        self.y_mm = eSpec_proc.screen_y_mm
        self.dy = np.mean(np.diff(self.y_mm))

        # energy markers
        spec_x_mm = eSpec_proc.spec_x_mm.flatten()
        spec_MeV = eSpec_proc.spec_MeV.flatten()
        x_MeV_cal = interp1d(spec_MeV, spec_x_mm, kind='linear', copy=True, bounds_error=False, fill_value=1e9)
        E_labels = np.arange(300,1501,100)
        x_labels= x_MeV_cal(E_labels)
        x2p_func = interp1d(self.x_mm, np.arange(0,len(self.x_mm)), kind='linear', copy=True, bounds_error=False, fill_value=0)

        p_labels = x2p_func(x_labels)
        p_lims = x2p_func([0,350])
        self.xaxis = []
        for x,y in zip(p_labels,E_labels):
            self.xaxis.append((x,y))
        self.yaxis = [(0, self.y_mm[0]), (len(self.y_mm), self.y_mm[-1])]
        self.p_labels = p_labels

    def get_image(self,path):
        img_raw = imread(path)
        img_pCpermm2 = self.eSpec_proc.espec_data2screen(img_raw)
        return img_pCpermm2.T

    def get_total_charge(self,path):
        img_pCpermm2 = self.get_image(path)
        return np.sum(img_pCpermm2)*self.dx*self.dy


class Gematron_proc:
    def __init__(self,run_name,shot_num=1,img_bkg=None):
        self.run_name = run_name
        self.shot_num = shot_num
        self.img_bkg = img_bkg
        self.setup_proc(run_name=run_name,shot_num=shot_num,img_bkg=img_bkg)

    def setup_proc(self,run_name=None,shot_num=None,img_bkg=None):
        if run_name is None:
            run_name = self.run_name
        if shot_num is None:
            shot_num = self.shot_num
        if img_bkg is None:
            img_bkg = self.img_bkg

        # paths for mask and filter information files
        diag = 'Gematron'
        mask_filepath = choose_cal_file(run_name,1,diag,'Gematron_filter_masks')
        print(mask_filepath)

        self.mask_obj=load_object(mask_filepath)
        filter_filepath = choose_cal_file(run_name,1,diag,'Gematron_filter_pack')
        self.filter_obj = load_object(filter_filepath)
        print(mask_filepath)
        # setup beam fitter
        self.beam_fitter = function_beam_fitter(self.mask_obj['beam_mask'],method='poly')

        f_labels = self.mask_obj['filter_label_regions']
        self.filter_rects = []
        for region in regionprops(f_labels):
            
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            self.filter_rects.append([(minc, minr), maxc - minc, maxr - minr])
            # arguments for overlaying rectangle as below...
            # rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
            #                         fill=False, edgecolor='red', linewidth=1, ls='--')

    def get_image(self,file_path):
        img = np.array(Image.open(file_path)).astype(float)
        if self.img_bkg is not None:
            img = img - self.img_bkg
        return img
    def get_beam_mean_counts(self,img):
        return np.mean(img[self.mask_obj['beam_mask']>0])



    def get_transmission(self,img_sub):
        
        img_beam = self.beam_fitter.fit_beam(img_sub,down_sel=10000,maxfev=100)
        self.img_beam = img_beam
        img_transmission = img_sub/img_beam
        return img_transmission

    def remove_filter_signal(self,img,f_number_list = [20,21]):
        f_pix = 0
        f_sig = 0
        for f_num in f_number_list:
            p_sel  = self.mask_obj['filter_number_regions']==f_num
            f_pix = f_pix + np.sum(p_sel)
            f_sig = f_sig + np.sum(img[p_sel])
        if f_pix>0:
            img_sub = img - f_sig/f_pix
        else:
            img_sub = img
            print('No filters selected to subtract')
        return img_sub
        

        
    def get_E_crit(self,img_transmission):
        spec_fitter = xray.Betatron_spec_fitter(img_transmission,
                        self.filter_obj,self.mask_obj)
        E_c,filter_trans_fit = spec_fitter.calc_E_crit()
        filter_trans_meas = spec_fitter.measured_trans
        self.E_keV = spec_fitter.E_keV
        self.null_T_E = spec_fitter.nullTrans
        self.QE = spec_fitter.aQE
        return E_c, filter_trans_fit, filter_trans_meas

          
    def file2xray_beam(self,file_path,mean_counts_thesh = 100):

        img_sub = self.get_image(file_path)
        img_sub = self.remove_filter_signal(img_sub)
        print(np.mean(img_sub))
        if np.mean(img_sub)<mean_counts_thesh:
            print('Image signal below threshold')
            return 0,0

        img_transmission = self.get_transmission(img_sub)
        E_c, filter_trans_fit, filter_trans_meas = self.get_E_crit(img_transmission)
        E_keV = self.E_keV
        # spectrum normalised to 1 keV total energy
        I_keV = xray.I_Xi(E_keV/(2*E_c))
        I_keV = I_keV/np.trapz(I_keV,x=E_keV)
        # number spectrum normalised to 1 keV total energy
        N_E_keV = I_keV/E_keV
        # number of photons above 1 keV for total 1 keV in source
        N_E_above_1keV_per_keV = np.trapz(N_E_keV*(E_keV>1),x=E_keV)
        # from Wood thesis. Counts per keV deposited in scintillator
        counts_per_keV = 0.24

        # Convert image counts to keV deposited
        img_beam_keV_det = self.img_beam/counts_per_keV
        # Divide by transmission and QE to get keV at source
        beam_keV = img_beam_keV_det/np.trapz(self.null_T_E*self.QE*I_keV,x=E_keV)

        # convert to number of photons (above 1 keV) 
        # by dividing by number of photons above 1 keV for total 1keV in spectrum
        img_beam_N = beam_keV*N_E_above_1keV_per_keV

        beam_N = np.sum(img_beam_N)
        return E_c, beam_N

        

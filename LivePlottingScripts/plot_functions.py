import sys, os, pickle, time
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from skimage.io import imread
from PIL import Image
from pathlib import Path
from glob import glob
from skimage.measure import label, regionprops

from XANES2020_code.paths import DATA_FOLDER, CAL_DATA
from XANES2020_code.Espec import espec_processing 
from XANES2020_code.general_tools import choose_cal_file, load_object, save_object
from XANES2020_code.Betatron_analysis.beam_fitting import function_beam_fitter
from XANES2020_code.Betatron_analysis import xray_analysis as xray
from XANES2020_code.HAPG.HAPG_analysis import HAPG_processor
from XANES2020_code import PKG_DATA

# CAL_DATA = r'C:\Users\CLFUser\Documents\ProcessedCalibrations'

def pil_img_array(path):
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
        print(tForm_filepath)
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
        x_MeV_cal = interp1d(spec_MeV, spec_x_mm, kind='linear', copy=True, bounds_error=False, fill_value=0)
        E_labels = np.arange(400,1501,100)
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
            # minr, minc, maxr, maxc = region.bbox
            self.filter_rects.append(region.bbox)
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
        if np.mean(img_sub)<mean_counts_thesh:
            print('Gematron image signal below threshold for retrieval')
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

class crystal_spec_proc:
    def __init__(self,run_name,shot_num=1,crystal=None,img_bkg=None):
        self.run_name = run_name
        self.shot_num = shot_num
        self.img_bkg = img_bkg
        self.crystal = crystal

        self.spectral_markers = {'Si': 3350.0, 'TAP': 1560.0, 'HOPG': 3350.0} #eV
        self.d2s = {'Si': 6.271, 'TAP': 25.763, 'HOPG': 6.708} #angstroms

        self.setup_proc(run_name=run_name,shot_num=shot_num,crystal=crystal,img_bkg=img_bkg)
        

    def setup_proc(self,run_name=None,shot_num=None,crystal=None,img_bkg=None):
        if run_name is None:
            run_name = self.run_name
        if shot_num is None:
            shot_num = self.shot_num
        if img_bkg is None:
            img_bkg = self.img_bkg

        if self.crystal is None or str(self.crystal) not in self.spectral_markers.keys():
            #add flag here to ignore all funcs to get energy axis
            pass
        else:
            #get filepaths of processing info
            
            #where is spectral filter shadow on image for energy axis retrieval
            spectral_region_filepath = choose_cal_file(run_name,shot_num, crystal,
                            crystal + '_spectral_region', cal_data_path=CAL_DATA)

            #where is background region on image for bkg subtraction
            bkg_region_filepath = choose_cal_file(run_name,shot_num, crystal,
                            crystal + '_bkg_region', cal_data_path=CAL_DATA)

            #positions in chamber for calculating angles
            setup_filepath = choose_cal_file(run_name,shot_num, crystal,
                            crystal + '_setup', cal_data_path=CAL_DATA)
            
            #process 
        return None    

    def get_image(self,file_path):
        img = np.array(Image.open(file_path)).astype(float)
        if self.img_bkg is not None:
            img = img - self.img_bkg
        return img

    def get_energy_axis(self,file_path):
        #find cut in spectral region
        xtal = np.array(Image.open(file_path)).astype(float)
        bounds = self.spectral_region
        xtal_spectral_region = xtal[bounds]
        offset = bounds[0] #which index here?
        
        spectral_region_lineout = np.mean(xtal_spectral_region, axis=0)
        cut_loc = self.step_finder(spectral_region_lineout) + offset

        #convert position on chip to energy
        

        #offset energy axis at energy of spectral marker to match position of cut in image
        
        
        return None

    def step_finder(self, array):
        """Simplestep-finder of 1d array by convolving with the signal
        you're looking for.
        """
        dary = np.copy(array)
        dary -= np.mean(dary)
        example_step = np.hstack((np.ones(len(dary)), -1.0*np.ones(len(dary) )))
        dary_conv_step = np.convolve(dary, example_step, mode='valid')
        step_idx = np.argmin(dary_conv_step)
        return step_idx
                                 
    
class HAPG_live_plotter():

    cal_file_pref = 'HAPG_cal'
    diag = 'HAPG'
    def __init__(self,beam_run_name):
        self.beam_run_name = beam_run_name
        self.HAPG_cal_file_path = choose_cal_file(self.beam_run_name,999,self.diag,self.cal_file_pref)
        print(self.HAPG_cal_file_path)
        self.HAPG_proc = HAPG_processor(HAPG_cal_file_path=self.HAPG_cal_file_path)
        print(np.min(self.HAPG_proc.spec_eV[self.HAPG_proc.spec_iSel]))
        print(np.max(self.HAPG_proc.spec_eV[self.HAPG_proc.spec_iSel]))

        if self.HAPG_proc.beam_ref is None:
            file_stem = str(Path(DATA_FOLDER) / self.diag / self.beam_run_name / 'Shot*' )
            beam_file_list = glob(file_stem)
            beam_ref,beam_ref_rms = self.HAPG_proc.get_beam_ref(beam_file_list)
            cal_info = load_object(self.HAPG_cal_file_path)
            cal_info['beam_ref'] = beam_ref
            cal_info['beam_ref_rms'] = beam_ref_rms
            
            new_cal_path = (Path(CAL_DATA) / self.diag / (self.cal_file_pref + '_'+
                            self.beam_run_name.split(r'/')[0] +'_' +
                            self.beam_run_name.split(r'/')[1] + f'_shot{len(beam_file_list):03}.pkl'))
            save_object(cal_info,new_cal_path)
            self.HAPG_cal_file_path =new_cal_path
            self.HAPG_proc = HAPG_processor(HAPG_cal_file_path=self.HAPG_cal_file_path)

        self.x = self.HAPG_proc.spec_eV[self.HAPG_proc.spec_iSel]

    def get_theoretical_data(self):
        df = pd.read_csv(Path(PKG_DATA) / 'HAPG' / 'Rehr_Fig4_XAFS.txt',header=None)
        rehr=(df[1]/12-0.05 )*1.2
        xr=df[0]+8988
        y = interp1d(xr,rehr,bounds_error=False,fill_value=0)(self.x)
        return self.x, y

    def get_HAPG_norm_abs(self,file_path):
        norm_abs = self.HAPG_proc.HAPG_file2norm_abs(file_path)

        return (self.HAPG_proc.spec_eV[self.HAPG_proc.spec_iSel], norm_abs[self.HAPG_proc.spec_iSel])

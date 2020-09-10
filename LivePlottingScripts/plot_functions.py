import sys, os, pickle, time
import numpy as np
from scipy.interpolate import interp1d
from skimage.io import imread
from PIL import Image
from XANES2020_code.paths import DATA_FOLDER, CAL_DATA
from XANES2020_code.Espec import espec_processing 
from XANES2020_code.general_tools import choose_cal_file

# CAL_DATA = r'\\clftagw02\data\GeminiXANES2020\ProcessedCalibrations'

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
        x_MeV_cal = interp1d(spec_MeV, spec_x_mm, kind='linear', copy=True, bounds_error=False, fill_value=0)
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




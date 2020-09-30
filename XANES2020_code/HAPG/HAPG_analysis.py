import numpy as np
from scipy.ndimage.filters import median_filter
from skimage.io import imread
from scipy.optimize import minimize

from XANES2020_code import mirage_analysis, PKG_DATA
from XANES2020_code.general_tools import load_object, choose_cal_file, get_cal_files, calc_COW

def load_HAPG_image(file_path):
    return imread(file_path)

def image_process(data,img_bkg=0,sub_median=False):
    
    data = data.astype(float) - img_bkg
    if sub_median:
        data = data - np.median(data)
    return median_filter(data,5)

def smooth_gaus(x,y,sigma_x):
    X1,X2 = np.meshgrid(x,x)
    W = np.exp(-(X1-X2)**2/(2*sigma_x**2))
    y_smooth = np.sum(W*y,axis=1)/np.sum(W,axis=1)
    return y_smooth

def normalise(y):
    return y/np.max(y)


class HAPG_processor:
    def __init__(self,HAPG_cal_file_path,img_bkg=0):
        self.HAPG_cal_file_path = HAPG_cal_file_path
        self.load_cal_info()
        self.img_bkg = img_bkg 
        self.data = None
        self.sig_width = 150
        self.sub_median = True
        self.sigma_x = 1

        self.beam_ref = None

    def load_cal_info(self):
        cal_info = load_object(self.HAPG_cal_file_path)
        # self.pxsize=cal_info['pxsize']
        # self.dispersion=cal_info['dispersion']
        # self.x_offset=cal_info['x_offset']
        self.spec_eV = cal_info['spec_eV']
        self.iSel = cal_info['iSel']
        if 'sig_mask' in cal_info.keys():
            self.sig_mask = cal_info['sig_mask']
        if 'beam_ref' in cal_info.keys():
            self.beam_ref = cal_info['beam_ref']
            self.beam_ref_rms = cal_info['beam_ref_rms']
        return cal_info

    def HAPG_file2data(self,file_path):
        img = load_HAPG_image(file_path)
        data = image_process(img,self.img_bkg,sub_median=self.sub_median)
        return data
        
    def find_sig_mask(self,data):
         
        Ny,Nx = np.shape(data)
        self.x,self.y = np.arange(Nx),np.arange(Ny)
        
        self.X,self.Y = np.meshgrid(self.x,self.y)
        vmax=np.percentile(data,99.5)
        self.c_x,self.c_y= calc_COW(data,img_thresh=vmax)
        self.sig_mask =np.abs(self.X-self.c_x)<(self.sig_width/2)
        return self.sig_mask

    def get_spec_y(self,data):        
        spec_y =  np.sum(data*self.sig_mask,axis=1)
        return spec_y
    
    def get_beam_ref(self,file_path_list):
        spec_y_list = []
        for file_path in file_path_list:
            data = self.HAPG_file2data(file_path)
            self.find_sig_mask(data)
            spec_y_list.append(self.get_spec_y(data))
        
        self.beam_ref,self.beam_ref_rms, y_list = proc_spec(spec_y_list,smooth_sigma=self.sigma_x)

        return self.beam_ref,self.beam_ref_rms

    
    def spec2trans(self,spec_y):
        y_max = np.max(smooth_gaus(self.y,spec_y,10))
        if self.sigma_x is not None:
            spec_y = smooth_gaus(self.y,spec_y,self.sigma_x)


        spec_y_norm = spec_y/y_max

        def err_fcn(A):
            return np.sqrt(np.sum((A*spec_y_norm[self.iSel]-self.beam_ref[self.iSel])**2))

        res = minimize(err_fcn,x0=1)
        spec_y_norm = spec_y_norm*res.x[0]
        trans = spec_y_norm/self.beam_ref
        return trans

    def trans2xafs(self,trans):
        xafs= (0.65-trans)*0.3+0.12
        return xafs
    

    def HAPG_file2xafs(self,file_path):
        data = self.HAPG_file2data(file_path)
        spec_y = self.get_spec_y(data)
        self.trans = self.spec2trans(spec_y)
        xafs = self.trans2xafs(self.trans)
        return xafs

def proc_spec(spec_list,smooth_sigma=None):
    x = np.arange(len(spec_list[0]))
    y_template = normalise(smooth_gaus(x,np.mean(spec_list,axis=0),10))
    def err_fcn(A):
        return np.sqrt(np.sum((A*y-y_template)**2))

    y_list = []
    for y in spec_list:
        y_max =np.max(smooth_gaus(x,y,10))
        y = y/y_max
        res = minimize(err_fcn,x0=1)
        y = y*res.x[0]
        y_list.append(y)
    y_avg = np.mean(y_list,axis=0)
    y_rms = np.std(y_list,axis=0)
    if smooth_sigma is not None:
        y_avg = smooth_gaus(x,y_avg,smooth_sigma)
        y_rms = smooth_gaus(x,y_rms,smooth_sigma)
    return y_avg,y_rms, y_list

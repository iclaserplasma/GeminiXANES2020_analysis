import matplotlib.pyplot as plt
import numpy as np
import pickle, os
from datetime import datetime
from XANES2020_code.paths import CAL_DATA
from glob import glob
import pandas as pd

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as fid:
        return pickle.load(fid)

def imagesc(I,ax = None,  x=None, y=None, **kwargs):
    """ display image with axes using pyplot
    recreates matlab imagesc functionality (roughly)
    argument I = 2D numpy array for plotting
    kwargs:
        ax = axes to plot on, if None will make a new one
        x = horizontal  axis - just uses first an last values to set extent
        y = vetical axis like x
        **kwargs anthing else which is passed to imshow except extent and aspec which are set
    """
    if ax is None:
        plt.figure()
        ax = plt.axes()
    if x is None:
        Nx = np.size(I, axis=1)
        x = np.arange(Nx)
    if y is None:
        Ny = np.size(I, axis=0)
        y = np.arange(Ny)
    ext = (x[0], x[-1], y[-1], y[0])
    return ax.imshow(I, extent=ext, aspect='auto', **kwargs)



### functions for determining which calibration file to use

def choose_cal_file(run_name,shot,diag,file_pref,cal_data_path=None):
    run_dt, run_num = get_run_name_info(run_name)
    c_path_sel = None

    cal_paths = get_cal_files(diag=diag,file_pref=file_pref,cal_data_path=cal_data_path)
    N_t_path = len(cal_paths)
    for c_path in cal_paths:
        c_path_dt, c_path_run_num, c_path_shot_num =get_cal_path_info(c_path,file_pref=file_pref)
        pre_data = is_arg1_geq_arg2((run_dt, run_num, shot ),
                                    (c_path_dt, c_path_run_num, c_path_shot_num ))
        
        if pre_data:
            if c_path_sel is None:
                c_path_sel = c_path
                c_path_sel_info = (c_path_dt,c_path_run_num,c_path_shot_num)

            else:
                post_other =is_arg1_geq_arg2((c_path_dt, c_path_run_num, c_path_shot_num ),
                                    c_path_sel_info)
                if post_other:
                    c_path_sel = c_path
                    c_path_sel_dt = c_path_dt
                    c_path_sel_info = (c_path_dt,c_path_run_num,c_path_shot_num)
    return c_path_sel

def get_cal_files(diag,file_pref,cal_data_path=None):
    if cal_data_path is None:
        cal_data_path = CAL_DATA
    
    cal_paths =  glob(os.path.join(cal_data_path,diag,file_pref+'*'))

    return cal_paths

def get_cal_path_info(filepath,file_pref):
    cal_path_date, cal_path_run_shot = filepath.split(file_pref+'_')[1].split('_run')
    cal_path_run_str, cal_path_shot_str = cal_path_run_shot.split('.')[0].split('_shot')
    cal_path_dt =  datetime.strptime(cal_path_date, '%Y%m%d')
    cal_path_run_num = int(cal_path_run_str)
    cal_path_shot_num = int(cal_path_shot_str)
    return cal_path_dt, cal_path_run_num, cal_path_shot_num
    
def get_run_name_info(run_name):
    run_date, run_num = run_name.split('/')
    run_dt = datetime.strptime(run_date, '%Y%m%d')
    run_num = int(run_num.split('run')[1])
    return run_dt, run_num  

def is_arg1_geq_arg2(dt_shot_run_tup1,dt_shot_run_tup2):
    
    if (dt_shot_run_tup1[0]-dt_shot_run_tup2[0]).total_seconds()==0:
        # same day
        if dt_shot_run_tup1[1]==dt_shot_run_tup2[1]:
            #same run
            if dt_shot_run_tup1[2]>=dt_shot_run_tup2[2]:
                answer = True
            else:
                answer = False
        elif dt_shot_run_tup1[1]>dt_shot_run_tup2[1]:
            answer = True
        else:
            answer = False
    elif (dt_shot_run_tup1[0]-dt_shot_run_tup2[0]).total_seconds()>0:
        answer = True
    else:
        answer = False
    
    return answer

def calc_COW(img,X=None,Y=None,img_thresh=0.5):
    iSel = img>img_thresh
    if (X is None) or (Y is None):
        Ny,Nx = np.shape(img)
        X,Y = np.meshgrid(np.arange(Nx),np.arange(Ny))
    c_x = np.sum(X[iSel]*img[iSel])/np.sum(img[iSel])
    c_y = np.sum(Y[iSel]*img[iSel])/np.sum(img[iSel])
    return c_x,c_y

def glob_path(p):
    return glob(str(p))

def smooth_gauss(x,y,sigma_x):
    X1,X2 = np.meshgrid(x,x)
    W = np.exp(-(X1-X2)**2/(2*sigma_x**2))
    y_smooth = np.sum(W*y,axis=1)/np.sum(W,axis=1)
    return y_smooth
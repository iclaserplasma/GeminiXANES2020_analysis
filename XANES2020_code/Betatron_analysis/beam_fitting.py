## Non-parametric fitting with gaussian process regression
## Matthew Streeter 2020

import numpy as np
from scipy.ndimage.filters import median_filter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel, WhiteKernel)
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline

from scipy.ndimage import gaussian_filter
from scipy import optimize

class GP_beam_fitter():
    
    def __init__(self,beam_mask,N_samples = 1000, N_pred =(100,100)):
        self.beam_mask = beam_mask
        self.mask_ind = np.nonzero(beam_mask.flatten())
        self.N_samples = N_samples
        self.N_pred = N_pred
        
        kernel = 1**2* Matern(
            length_scale=0.1, length_scale_bounds=(1e-2, 10.0),nu=1.5
        ) + WhiteKernel()
        
        self.gp = GaussianProcessRegressor(kernel=kernel)
        self.x,self.y,self.XY = self._make_image_grid(beam_mask)
        self.x_pred,self.y_pred,self.XY_pred = self._make_image_grid(np.ones(N_pred))
        self._determine_pixel_weights()
        
        
    def _make_image_grid(self,img):
        x = np.linspace(-1,1,num=img.shape[1],endpoint=True)
        y = np.linspace(-1,1,num=img.shape[0],endpoint=True)
        [X,Y] = np.meshgrid(x,y)
        XY = np.array([X.flatten(),Y.flatten()]).T
        return x,y,XY
        
    def _determine_pixel_weights(self):
        bmf = gaussian_filter(self.beam_mask,100)        
        pixel_w = 1/bmf.flatten()[self.mask_ind]
        self.pixel_w = pixel_w/np.sum(pixel_w)
        
    def fit_beam(self,image,med_filter=5):
  
        imgMax = np.max(image)
        imgMin = np.min(image)
        if med_filter is not None:
            image = median_filter(image.astype(float),med_filter)
        I  = (image.flatten()-imgMin)/(imgMax-imgMin)
        I_index = np.arange(len(I))

        selected_index = np.random.choice(I_index[self.mask_ind],
                                          size=self.N_samples,replace=False,p=self.pixel_w)

        x_train = self.XY[selected_index,:]
        I_train = I[selected_index]
        self.gp.fit(x_train,I_train)
       

        I_pred,I_pred_err = self.gp.predict(self.XY_pred,return_std=True)
        I_pred = I_pred.reshape(self.N_pred)
        I_pred_err = I_pred_err.reshape(self.N_pred)

        beam_image = RectBivariateSpline(self.x_pred,self.y_pred,I_pred)(self.x,self.y)*(imgMax-imgMin)+imgMin
        beam_unc = RectBivariateSpline(self.x_pred,self.y_pred,I_pred_err)(self.x,self.y)*(imgMax-imgMin)
        
        trans_image = image/beam_image
        null_trans_vals = trans_image[np.nonzero(self.beam_mask)]
        null_trans_mean = np.mean(null_trans_vals)
        null_trans_rms = np.std(null_trans_vals,dtype=np.float64)
        # print(f'Null transmission mean = {null_trans_mean:1.06f}')
        # print(f'Null transmission rms = {null_trans_rms:1.06f}')

        return beam_image, beam_unc

def gauss2Dbeam(U,a0,a1,a2,a3,a4,a5):
    # a0 peak,
    # a2,a4 widths
    # a1,a3 centers
    # a5 angle
    f = a0*np.exp( -(
        ( U[:,0]*np.cos(a5)-U[:,1]*np.sin(a5) - a1*np.cos(a5)+a3*np.sin(a5) )**2/(2*a2**2) + 
        ( U[:,0]*np.sin(a5)+U[:,1]*np.cos(a5) - a1*np.sin(a5)-a3*np.cos(a5) )**2/(2*a4**2) ) )

    return f

def polyBeam(U,a0,a1,a2,a3,a4):
    R = np.sqrt((U[:,0]-a0)**2 + (U[:,1]-a1)**2)
    P = (a2,a3,a4)
        
    N = np.size(P)
    f = np.zeros(np.shape(R))
    for n in range(0,N):
        f = f + P[n]*R**(N-n-1)

    return f

def gauss2DbeamFit(pG,U,I):
    
    f = gauss2Dbeam(U,*pG)
    fErr = np.sqrt(np.mean((f-I)**2))
    return fErr

def setRange2one(x):
    xMin = np.min(x)
    xMax = np.max(x)
    xRange = xMax-xMin
    x = 2*(x-xMin)/xRange-1
    return x

class function_beam_fitter():
    def __init__(self,beam_mask,method = 'polynomial'):
        beam_mask = beam_mask>0
        self.method = method
        self.beam_mask = beam_mask
        self.Ny, self.Nx  = np.shape(beam_mask)
        self.x = setRange2one(np.arange(self.Nx))
        self.y = setRange2one(np.arange(self.Ny))
        X ,Y = np.meshgrid(self.x,self.y)
        self.XY_mask = np.array([X[beam_mask],Y[beam_mask]]).T
        self.XY_full = np.array([X.flatten(),Y.flatten()]).T
        
    def fit_beam(self,img,down_sel = None,**kwargs):
        
        I = img[self.beam_mask]
        I_mean  = np.mean(I)
        I = I/I_mean
        if down_sel is not None:
            Np = len(I)
            p_sel = np.random.choice(np.arange(Np),down_sel)
            XY_mask = self.XY_mask[p_sel,:]
            I = I[p_sel]
        else:
            XY_mask = self.XY_mask
        
    
        if self.method.lower() in 'polynomial':
            if 'maxfev' not in kwargs:
                kwargs['maxfev'] = 400
            if 'xtol' not in kwargs:
                kwargs['xtol'] = 0.5e-3
            if 'ftol' not in kwargs:
                kwargs['ftol'] = 0.1
            if 'p0' not in kwargs:
                kwargs['p0'] = (0, 0, -0.1 ,-0.1  ,-0.5)
            (p_fit,pcov) = optimize.curve_fit(polyBeam, XY_mask, I,
                    **kwargs)

            I_beam = polyBeam(self.XY_full,*p_fit)*I_mean
            
        elif self.method.lower() in 'gaussian':
            pGuess = (1,0,1,0,1,0)

            args = (self.XY_mask,I)
            z = optimize.minimize(gauss2DbeamFit,pGuess,args=args, tol=0.01,method='Nelder-Mead')
            p_fit = z.x
            I_beam = gauss2Dbeam(self.XY_full,*p_fit)*I_mean

        self.img_beam = np.reshape(I_beam,(self.Nx,self.Ny),order='C')
        self.p_fit = p_fit
        return self.img_beam

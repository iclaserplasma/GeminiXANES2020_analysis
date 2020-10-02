# standard libraries
from PyQt5 import QtWidgets, uic, QtCore
import sys, os
import traceback
from datetime import datetime
import numpy as np
from scipy.interpolate import interp1d
from PIL import Image

# Stephens code
import mirage_ui
from XANES2020_code.mirage_analysis import easy_plotting

# UI
from fake_server import Ui_fake_server

# experiment code
from XANES2020_code.paths import DATA_FOLDER, CAL_DATA
from XANES2020_code.general_tools import load_object, save_object, choose_cal_file
from plot_functions import Espec_high_proc, pil_img_array, Gematron_proc, HAPG_live_plotter


from pyqtgraph import PlotDataItem
## setup UI
app = QtCore.QCoreApplication.instance()
if app is None:
    app = QtWidgets.QApplication(sys.argv)
    
HistoryShots = 20

server = Ui_fake_server()

win = easy_plotting.DockView(server, DATA_FOLDER)
win.dock_arrangement_file = 'dock_state_XANES.pkl'

## dx420 plots

def dx420_img(file_path):
    print(file_path)
    try:
        im = pil_img_array(file_path).T
    except(FileNotFoundError):
        print('No file found for ' + file_path)
        im = np.empty((1024, 256))
        im[:] = np.nan
        
    return im

def dx420_mean_x(file_path):
    img = dx420_img(file_path)
    return np.mean(img,axis=1)

def dx420_mean_y(file_path):
    img = dx420_img(file_path)
    return np.mean(img,axis=0)

andor_diag_list = ['Si','TAP','HOPG']
for diag in andor_diag_list:
    win.add_image_plot(diag, diag, dx420_img)
    win.add_line_plot(diag+' lineout', diag, dx420_mean_x)
    server.diag_list.addItem(diag)

win.add_image_plot('Pinhole', 'Pinhole', dx420_img)
win.add_line_plot('Pinhole'+' x lineout', 'Pinhole', dx420_mean_x)
win.add_line_plot('Pinhole'+' y lineout', 'Pinhole', dx420_mean_y)
server.diag_list.addItem('Pinhole')

# --------- HAPG

diag = 'HAPG'
beam_run_name = r'20200929/run28'

HAPG_plot = HAPG_live_plotter(beam_run_name)

win.add_line_plot(diag + ' absorption', diag, HAPG_plot.get_HAPG_norm_abs)
win.add_image_plot(diag, diag, dx420_img)
server.diag_list.addItem(diag)

server.run_name.setText('20200929/run01')
server.shot_num.setValue(1)
win.load_dock_arrangement()
win.show()
win.raise_()
# server.connected.connect(win.show)
app.exec_()


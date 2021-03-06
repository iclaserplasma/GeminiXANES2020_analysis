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
from XANES2020_code.paths import DATA_FOLDER
from plot_functions import Espec_high_proc, pil_img_array, Gematron_proc

from pyqtgraph import PlotDataItem
## setup UI
app = QtCore.QCoreApplication.instance()
if app is None:
    app = QtWidgets.QApplication(sys.argv)
    
HistoryShots = 20

server = Ui_fake_server()

win = easy_plotting.DockView(server, DATA_FOLDER)


## High espec plots
run_name = datetime.today().strftime('%Y%m%d')+'/run01'
Espec_high_obj = Espec_high_proc(run_name,shot_num=1,img_bkg=None)
win.add_image_plot_with_axes('Espec high', 'Espec_high', Espec_high_obj.get_image,
                             Espec_high_obj.xaxis, Espec_high_obj.yaxis)

for p in Espec_high_obj.p_labels:
    win.docks['Espec high'].widgets[0].view.addLine(x=p)

win.add_scalar_history_plot('Espec high counts', 'Espec_high', HistoryShots, Espec_high_obj.get_total_charge)

server.diag_list.addItem('Espec_high')

## Lundatron plot
win.add_image_plot_with_axes('Lundatron', 'Lundatron', pil_img_array,
        [(0, 0), (2048, 2048)],
        [(0, 0), (2048, 2048)])
server.diag_list.addItem('Lundatron')


## Gematron plot
bkg_file = os.path.join(DATA_FOLDER,'Gematron/20200907/run01/Shot001.tif')
img_bkg  = np.array(Image.open(bkg_file)).astype(float)
Gematron_proc_obj = Gematron_proc(run_name,shot_num=1,img_bkg=None)
def get_E_crit(file_path):
    E_c, beam_N = Gematron_proc_obj.file2xray_beam(file_path,mean_counts_thesh=100)
    return E_c
def get_beam_N(file_path):
    E_c, beam_N = Gematron_proc_obj.file2xray_beam(file_path,mean_counts_thesh=100)
    return beam_N
def get_mean_beam_counts(file_path):
    img_sub = Gematron_proc_obj.get_image(file_path)
    return Gematron_proc_obj.get_beam_mean_counts(img_sub)
def gematron_image(file_path):
    return  Gematron_proc_obj.get_image(file_path).T
win.add_image_plot_with_axes('Gematron', 'Gematron', gematron_image,
        [(0, 0), (2048, 2048)],
        [(0, 0), (2048, 2048)])

for f_rect in Gematron_proc_obj.filter_rects:
    # self.filter_rects.append([(minc, minr), maxc - minc, maxr - minr])
            # arguments for overlaying rectangle as below...
    # rect = mpatches.Rectangle(f_rect[0], f_rect[1], f_rect[2],
    #                         fill=False, edgecolor='red', linewidth=1, ls='--')
    minr, minc, maxr, maxc = f_rect
    xValues = [minc, minc, maxc, maxc, minc]
    yValues = [minr, maxr, maxr, minr, minr]
    win.docks['Gematron'].widgets[0].view.addItem(PlotDataItem(xValues, yValues,pen=dict(color='r',dash=[4,4])))

win.add_scalar_history_plot('Gematron E_c', 'Gematron', HistoryShots, get_E_crit)

win.add_scalar_history_plot('Gematron photons', 'Gematron', HistoryShots, get_beam_N)

win.add_scalar_history_plot('Gematron mean counts', 'Gematron', HistoryShots, get_mean_beam_counts)
server.diag_list.addItem('Gematron')


## dx420 plots

def basic_image(file_path):
    print(file_path)
    im = pil_img_array(file_path).T
    return im

def dx420_mean_x(file_path):
    img = basic_image(file_path)
    return np.mean(img,axis=1)

def dx420_mean_y(file_path):
    img = basic_image(file_path)
    return np.mean(img,axis=0)

andor_diag_list = ['Si','TAP','HOPG']
for diag in andor_diag_list:
    win.add_image_plot(diag, diag, basic_image)
    win.add_line_plot(diag+' lineout', diag, dx420_mean_x)
    server.diag_list.addItem(diag)

pinhole_diag_list = ['Pinhole', 'PinholeAl']
for diag in pinhole_diag_list:
    win.add_image_plot(diag, diag, basic_image)
    win.add_line_plot(diag+' x lineout', diag, dx420_mean_x)
    win.add_line_plot(diag+' y lineout', diag, dx420_mean_y)
    server.diag_list.addItem(diag)


gigE_diag_list = ['TopView', 'Crystal', 'F40focus', 'LMS', 'HMS', 'LMI']
for diag in gigE_diag_list:
    win.add_image_plot(diag, diag, basic_image)
    server.diag_list.addItem(diag) 


server.run_name.setText('20200915/run01')
server.shot_num.setValue(5)
win.load_dock_arrangement()
win.show()
win.raise_()
# server.connected.connect(win.show)
app.exec_()

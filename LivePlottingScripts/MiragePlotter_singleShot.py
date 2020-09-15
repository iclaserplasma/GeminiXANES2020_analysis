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
from mirage_analysis import easy_plotting

# UI
from fake_server import Ui_fake_server

# experiment code
from XANES2020_code.paths import DATA_FOLDER
from plot_functions import Espec_high_proc, xray_andor_image, Gematron_proc


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
win.add_image_plot_with_axes('Lundatron', 'Lundatron', xray_andor_image,
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
    return  Gematron_proc_obj.get_image(file_path)
win.add_image_plot_with_axes('Gematron', 'Gematron', gematron_image,
        [(0, 0), (2048, 2048)],
        [(0, 0), (2048, 2048)])

win.add_scalar_history_plot('Gematron E_c', 'Gematron', HistoryShots, get_E_crit)

win.add_scalar_history_plot('Gematron photons', 'Gematron', HistoryShots, get_beam_N)

win.add_scalar_history_plot('Gematron mean counts', 'Gematron', HistoryShots, get_mean_beam_counts)
server.diag_list.addItem('Gematron')



server.run_name.setText('20200907/run04')
server.shot_num.setValue(1)
win.show()
win.raise_()
# server.connected.connect(win.show)
app.exec_()

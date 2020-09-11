from PyQt5 import QtWidgets, uic, QtCore
import mirage_ui
from mirage_analysis import live_plotting
from mirage_analysis.easy_plotting import _make_path
from XANES2020_code.paths import DATA_FOLDER
import pyqtgraph as pg
import numpy as np
import os



class Ui_fake_server(QtWidgets.QMainWindow):
    """ Object for sending file paths to  easy_plotting 
    """
    connected = QtCore.pyqtSignal()
    download_queue_ready = QtCore.pyqtSignal(str, int)
    def __init__(self):
        super(Ui_fake_server, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi('fake_server.ui', self) # Load the .ui file
        self.send_paths_button.clicked.connect(self.send_paths)
        self.next_shot.clicked.connect(self.show_next_shot)
        self.prev_shot.clicked.connect(self.show_prev_shot)
        self.set_manual.clicked.connect(self.set_manual_to_displayed)

        remote_server = mirage_ui.network.ServerConnection('server.pem')
        remote_server.connected.connect(lambda: print('Connected!'))
        remote_server.connection_error.connect(lambda s: print('Error:', s))
        remote_server.connect('clftagw02', 5000, '7IGb5SVx3-I')
        self.relay = Server_relay(remote_server, self)
        self.show() # Show the GUI
        
    def send_paths(self,url=None):
        print('test')
        
        if isinstance(url,str):
            self.last_URL.setText(url)
            if self.auto_update.isChecked():
                self.download_queue_ready.emit(url,0)
        else:

            run_name = self.run_name.text()
            shot_str = f'{int(self.shot_num.text()):03}'
            for n in range(self.diag_list.count()):
                diag_name = self.diag_list.item(n).text()
                diag_file_path = '/'.join((diag_name,run_name,'Shot'+shot_str+'.tif'))
                if os.path.isfile(os.path.join(DATA_FOLDER,diag_name,run_name,'Shot'+shot_str+'.tif')):
                    self.current_run.setText(run_name)
                    self.current_shot.setText(shot_str)
                print('diag_name', diag_name)
                print('diag_file_path', diag_file_path)
                
                self.download_queue_ready.emit(diag_file_path,0)

    def show_next_shot(self):
        v = self.shot_num.value()
        self.shot_num.setValue(np.clip(v+1,1,None))
        self.send_paths()
    def show_prev_shot(self):
        v = self.shot_num.value()
        self.shot_num.setValue(np.clip(v-1,1,None))
        self.send_paths()
    
    def set_manual_to_displayed(self):
        self.shot_num.setValue(int(self.current_shot.text()))
        self.run_name.setText(self.current_run.text())
        
        



class Server_relay(pg.ImageView):
    """Something to take a signal from one server and pass it on through another
    """

    class Processor(live_plotting.DataProcessor):
        def process_data(self, url):
            path = _make_path(self.base_path, url)
            self.data = url
            

        def render_data(self):
            self.local_server.send_paths(self.data)

    def __init__(self, remote_server, local_server):
        super().__init__()
        remote_server.download_queue_ready.connect(local_server.download_queue_ready)


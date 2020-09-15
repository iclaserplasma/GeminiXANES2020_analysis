"""A utility module to make live plotting even easier.

Create an DockView and call one of its add_ methods, passing it the plot name,
the diagnostic name, and your processing function.

Each processing function is given the absolute path name or URL, and it is
expected to return a processed value. The type of value depends on the type of
plot. All processing functions will run in a separate thread.
"""

import os
import re
import urllib.parse

from . import live_plotting

import numpy as np
import pyqtgraph as pg
from pyqtgraph.dockarea import DockArea, Dock
from XANES2020_code.general_tools import save_object, load_object

URL_REGEX = re.compile(r'\w+://')

def _make_path(base_path, relative_path):
    if URL_REGEX.match(base_path):
        return urllib.parse.urljoin(base_path, relative_path)
    else:
        relative_path = urllib.parse.unquote(relative_path)
        return os.path.join(base_path, relative_path)

class ScalarHistoryPlot(live_plotting.SimpleLineGraph):
    """A plot of the history of a scalar value.

    The processing function is expected to return a single scalar value.
    """

    def __init__(self, server, diag_name, base_path, history, processing_func):
        super().__init__(server, diag_name, history)
        self.base_path = base_path
        self.user_process_data = processing_func

    def process_data(self, url):
        path = _make_path(self.base_path, url)
        return self.user_process_data(path)

class LinePlot(pg.GraphicsLayoutWidget):
    """A plot showing a 1D array.
    
    This is suitable for plotting a vector or 1-D array per laser shot. The
    processing function may return a numpy array (or array-like) or it may
    return a 2-element tuple, containing the x and y data.
    """

    class Processor(live_plotting.DataProcessor):
        def process_data(self, url):
            path = _make_path(self.base_path, url)
            data = self.user_process_data(path)
            if isinstance(data, tuple):
                x_data, y_data = data
            else:
                y_data = data
                x_data = np.arange(len(y_data))
            self.x_data = x_data
            self.y_data = y_data

        def render_data(self):
            self.curve.setData(self.x_data, self.y_data)

    def __init__(self, server, diag_name, base_path, processing_func):
        super().__init__()

        p = self.addPlot()
        curve = p.plot([0])

        data_processor = self.Processor(self, diag_name=diag_name)
        data_processor.base_path = base_path
        data_processor.curve = curve
        data_processor.user_process_data = processing_func
        data_processor.start()

        server.download_queue_ready.connect(data_processor.new_data)

class ImagePlot(pg.ImageView):
    """A plot showing a 2D array (a matrix or image).

    The processing function returns a 2D array only.

    The x_axis and y_axis parameters are lists of pairs of values. Each pair is
    a pixel number and an axis label.
    """

    class Processor(live_plotting.DataProcessor):
        def process_data(self, url):
            path = _make_path(self.base_path, url)
            self.data = self.user_process_data(path)

        def render_data(self):
            self.img_view.setImage(self.data, autoRange=False, autoLevels=False, autoHistogramRange=False)

    def __init__(self, server, diag_name, base_path, processing_func, *, x_axis=None, y_axis=None):
        if x_axis or y_axis:
            view = pg.PlotItem()
            if x_axis:
                view.getAxis('bottom').setTicks([x_axis, []])
            if y_axis:
                view.getAxis('left').setTicks([y_axis, []])
            super().__init__(view=view)
        else:
            super().__init__()

        data_processor = self.Processor(self, diag_name=diag_name)
        data_processor.base_path = base_path
        data_processor.img_view = self
        data_processor.user_process_data = processing_func
        data_processor.start()

        server.download_queue_ready.connect(data_processor.new_data)

class DockView(DockArea):
    """A view to hold multiple dockable plots.

    Can be treated as a pyqtgraph DockArea, or you can use the add_ convenience
    functions.
    """
    dock_arrangement_file = 'dock_state.pkl'
    def __init__(self, server, base_path):
        super().__init__()
        server.save_view.clicked.connect(self.save_dock_arrangement)
        self.server = server
        self.base_path = base_path
        

    def add_scalar_history_plot(self, dock_name, diag_name, history, func):
        dock = Dock(dock_name, size=(400, 400))
        self.addDock(dock)
        dock.addWidget(ScalarHistoryPlot(self.server, diag_name, self.base_path, history, func))
    
    def add_line_plot(self, dock_name, diag_name, func):
        dock = Dock(dock_name, size=(400, 400))
        self.addDock(dock)
        dock.addWidget(LinePlot(self.server, diag_name, self.base_path, func))

    def add_image_plot(self, dock_name, diag_name, func):
        dock = Dock(dock_name, size=(400, 400))
        self.addDock(dock)
        dock.addWidget(ImagePlot(self.server, diag_name, self.base_path, func))

    def add_image_plot_with_axes(self, dock_name, diag_name, func, x_axis, y_axis):
        dock = Dock(dock_name, size=(400, 400))
        self.addDock(dock)
        dock.addWidget(ImagePlot(self.server, diag_name, self.base_path, func,
            x_axis=x_axis, y_axis=y_axis))

    def save_dock_arrangement(self):
        dock_state = self.saveState()
        save_object(dock_state,self.dock_arrangement_file)

    def load_dock_arrangement(self):
        if os.path.isfile(self.dock_arrangement_file):
            dock_state = load_object(self.dock_arrangement_file)
            self.restoreState(dock_state)




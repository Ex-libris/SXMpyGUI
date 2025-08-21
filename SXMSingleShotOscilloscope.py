
# single_scope.py
import sys, time, ctypes
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

from SXMOscilloscope import channels, DriverDataSource, MockDataSource, WIN32_AVAILABLE
import win32file, win32con


class CaptureThread(QtCore.QThread):
    finished = QtCore.pyqtSignal(np.ndarray, float)  # data, rate Hz

    def __init__(self, source, chan_idx, npoints=50000):
        super().__init__()
        self.source = source
        self.chan_idx = chan_idx
        self.npoints = npoints
        self._stop = False

    def run(self):
        vals = np.zeros(self.npoints, dtype=np.float64)
        t0 = time.perf_counter()
        for i in range(self.npoints):
            if self._stop:
                vals = vals[:i]
                break
            raw = self.source.read_value(self.chan_idx)
            # scale to physical units
            _, _, unit, scale = [c for c in channels.values() if c[0]==self.chan_idx][0]
            vals[i] = raw * scale
        t1 = time.perf_counter()
        rate = len(vals) / (t1 - t0)
        self.finished.emit(vals, rate)

    def stop(self):
        self._stop = True


class ScopeWindow(QtWidgets.QMainWindow):
    def __init__(self, source):
        super().__init__()
        self.setWindowTitle("SXM One-Shot Scope")
        self.source = source
        self.capture_thread = None

        central = QtWidgets.QWidget(self)
        vbox = QtWidgets.QVBoxLayout(central)

        # Controls
        hbox = QtWidgets.QHBoxLayout()
        self.chan_combo = QtWidgets.QComboBox()
        self.chan_combo.addItems(list(channels.keys()))
        hbox.addWidget(self.chan_combo)

        self.npoints_spin = QtWidgets.QSpinBox()
        self.npoints_spin.setRange(1000, 2_000_000)
        self.npoints_spin.setValue(50000)
        hbox.addWidget(QtWidgets.QLabel("Samples:"))
        hbox.addWidget(self.npoints_spin)

        self.start_btn = QtWidgets.QPushButton("Start Capture")
        self.start_btn.clicked.connect(self.start_capture)
        hbox.addWidget(self.start_btn)

        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_capture)
        hbox.addWidget(self.stop_btn)

        vbox.addLayout(hbox)

        # Plot
        self.plot = pg.PlotWidget()
        self.plot.setLabel('bottom', "Sample")
        self.plot.setLabel('left', "Value")
        vbox.addWidget(self.plot)

        self.setCentralWidget(central)

    def start_capture(self):
        chan_name = self.chan_combo.currentText()
        idx, _, unit, scale = channels[chan_name]
        npts = self.npoints_spin.value()
        self.plot.clear()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.capture_thread = CaptureThread(self.source, idx, npoints=npts)
        self.capture_thread.finished.connect(self.show_data)
        self.capture_thread.start()

    def stop_capture(self):
        if self.capture_thread is not None:
            self.capture_thread.stop()

    def show_data(self, arr, rate):
        self.plot.plot(arr, pen='y')
        QtWidgets.QMessageBox.information(self, "Capture done",
            f"Captured {len(arr)} points\nEstimated rate: {rate:.1f} samples/s")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)


def create_source():
    if WIN32_AVAILABLE:
        try:
            h = win32file.CreateFile(r"\\.\SXM",
                                     win32con.GENERIC_READ|win32con.GENERIC_WRITE,
                                     0, None, win32con.OPEN_EXISTING,
                                     win32con.FILE_ATTRIBUTE_NORMAL, None)
            return DriverDataSource(h)
        except Exception as e:
            print("Driver not available, using mock:", e)
    return MockDataSource()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    src = create_source()
    w = ScopeWindow(src)
    w.resize(900, 600)
    w.show()
    sys.exit(app.exec_())

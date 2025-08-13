"""
SXM Time-Based Multi-Channel Scope
- Spectral SNR (FFT) with toggle
- External PSD window (hide-on-close, frequency span up to 2 kHz)
- Fixed SNR 'blinking': caches last SNR between updates
"""

import sys
import time
import math
import ctypes
from typing import Optional, Tuple, List, Dict
from collections import deque
from itertools import islice

# --- Windows driver imports (fall back to mock if not available) -------------
try:
    import win32file, win32con
    WIN32_AVAILABLE = True
except Exception:
    WIN32_AVAILABLE = False

from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import numpy as np


# --------------------- PyQtGraph global options (performance) -----------------
pg.setConfigOptions(useOpenGL=True, antialias=False)


# ----------------------------- Channel dictionary -----------------------------
# human_name -> (driver_index, short_label, unit, scale_factor)
channels = {
    'Topo':       (  0, 'DAC0',    'nm', -2.60914e-07),
    'Bias':       ( -1, 'DAC1',    'mV',  9.4e-06),
    'x-direction':( -2, 'DAC2',    'nm',  1.34e-06),
    'y-direction':( -3, 'DAC3',    'nm',  1.34e-06),
    'DA1':        ( -4, 'DAC4',     'V', -9.41e-09),
    'Frequency':  ( -9, 'DAC9',    'Hz',  0.00232831),
    'Drive':      (-10, 'DAC10',    'V',  4.97789e-09),
    'QPlusAmpl':  (-12, 'DAC12',    'V',  3.64244e-09),
    'Phase':      (-13, 'DAC13',    '°',  0.001),
    'Lia1X':      (-14, 'DAC14',    'A',  1.56618e-19),
    'Lia1Y':      (-15, 'DAC15',    'A',  1.56618e-19),
    'Lia2X':      (-16, 'DAC16',    'A',  1.56618e-19),
    'Lia2Y':      (-17, 'DAC17',    'A',  1.56618e-19),
    'Lia3X':      (-18, 'DAC18',    'A',  1.56618e-19),
    'Lia3Y':      (-19, 'DAC19',    'A',  1.56618e-19),
    'Lia1R':      (-22, 'DAC22',    'A',  9.51067e-20),
    'Lia2R':      (-23, 'DAC23',    'A',  9.51067e-20),
    'Lia3R':      (-24, 'DAC24',    'A',  9.51067e-20),
    'Lia1Phi':    (-27, 'DAC27',    '*',  0.001),
    'Lia2Phi':    (-28, 'DAC28',    '*',  0.001),
    'Lia3Phi':    (-29, 'DAC29',    '*',  0.001),
    'df':         (-40, 'DAC40',    'Hz', 0.00232838),
    'It_ext':     ( 32, 'ADC0',     'A',  1.011e-19),
    'QPlus_ext':  ( 33, 'ADC1',    'mV',  1.008e-05),
    'AD1':        ( 34, 'ADC2',     'V',  1.012e-08),
    'AD2':        ( 35, 'ADC3',    'mV',  1.011e-05),
    'InA':        (  8, 'ADC4',     'V',  3.07991e-09),
    'It_to_PC':   ( 12, 'ADC9',     'A',  8.04187e-20),
    'Zeit':       ( 23, 'ADC12',    's',  0.001),
    'AD3':        ( 36, 'ADC13',   'mV',  1.01e-05),
    'AD4':        ( 37, 'ADC14',   'mV',  1.013e-05),
    'AD5':        ( 38, 'ADC15',   'mV',  1.01e-05),
    'AD6':        ( 39, 'ADC16',   'mV',  1.009e-05),
    'minmax':     ( 47, 'ADC21',    'A',  3.35e-07)
}


# ================================ Data sources =================================

class IDataSource:
    """Interface. Implementations return a RAW integer (no scaling)."""
    def read_value(self, channel_index: int) -> int:
        raise NotImplementedError


class DriverDataSource(IDataSource):
    """Reads raw values via the SXM device driver (DeviceIoControl)."""
    FILE_DEVICE_UNKNOWN = 0x00000022
    METHOD_BUFFERED     = 0
    FILE_ANY_ACCESS     = 0x0000

    @staticmethod
    def CTL_CODE(DeviceType, Access, Function_code, Method):
        return (DeviceType << 16) | (Access << 14) | (Function_code << 2) | Method

    IOCTL_GET_KANAL = CTL_CODE.__func__(FILE_DEVICE_UNKNOWN, FILE_ANY_ACCESS, 0xF0D, METHOD_BUFFERED)

    def __init__(self, driver_handle):
        self.handle = driver_handle
        self._inbuf = ctypes.create_string_buffer(ctypes.sizeof(ctypes.c_long))  # reused

    def read_value(self, channel_index: int) -> int:
        ctypes.memmove(self._inbuf, ctypes.byref(ctypes.c_long(channel_index)),
                       ctypes.sizeof(ctypes.c_long))
        out = win32file.DeviceIoControl(self.handle, self.IOCTL_GET_KANAL,
                                        self._inbuf, ctypes.sizeof(ctypes.c_long))
        return ctypes.c_long.from_buffer_copy(out).value


class MockDataSource(IDataSource):
    """Synthetic generator: slow drift + sinusoid + Gaussian noise (raw ints)."""
    def __init__(self, seed: int = 12345):
        self.t0 = time.time()
        self.rng = np.random.default_rng(seed)

    def read_value(self, channel_index: int) -> int:
        t = time.time() - self.t0
        f  = 0.5 + (abs(channel_index) % 7) * 0.3
        A  = 2_000_000 + (abs(channel_index) % 5) * 500_000
        sd = 80_000 + (abs(channel_index) % 3) * 40_000
        drift = int(300_000 * math.sin(0.02 * t))
        sine  = int(A * math.sin(2 * math.pi * f * t))
        noise = int(self.rng.normal(0, sd))
        if channel_index == 0:
            drift += int(500_000 * math.sin(0.1 * math.pi * t))
        return drift + sine + noise


# =============================== Tiny DSP (HPF) ===============================

class IIRHighPass:
    """1st-order high-pass for Relative Z on Topo: y[n] = α*(y[n-1] + x[n] − x[n-1])."""
    def __init__(self, tau_s: float, dt_s: float):
        self.alpha = tau_s / (tau_s + dt_s)
        self.prev_x = 0.0
        self.prev_y = 0.0
    def reset(self):
        self.prev_x = 0.0
        self.prev_y = 0.0
    def process(self, x_now: float) -> float:
        y = self.alpha * (self.prev_y + x_now - self.prev_x)
        self.prev_x, self.prev_y = x_now, y
        return y


# ========================= Deque → NumPy tail helper =========================

def deque_tail_to_arrays(t_deq: deque, v_deq: deque, tail_count: int, now: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return only the **last** `tail_count` samples as arrays: xs=(t-now), ys."""
    if not t_deq:
        return np.empty(0), np.empty(0)
    k = min(tail_count, len(t_deq))
    t_rev = list(islice(reversed(t_deq),  0, k))
    v_rev = list(islice(reversed(v_deq), 0, k))
    t_arr = np.asarray(t_rev[::-1], dtype=float)
    v_arr = np.asarray(v_rev[::-1], dtype=float)
    return (t_arr - now), v_arr


def estimate_rate_hz(xs_rel: np.ndarray) -> float:
    """Robust receive rate (pts/s) from relative time axis using median Δt."""
    n = xs_rel.size
    if n < 2:
        return 0.0
    dts = np.diff(xs_rel)
    dts = dts[dts > 0]
    if dts.size == 0:
        denom = xs_rel[-1] - xs_rel[0]
        return (n - 1) / denom if denom > 0 else 0.0
    if dts.size > 50:
        dts = dts[-50:]
    med_dt = float(np.median(dts))
    return (1.0 / med_dt) if med_dt > 0 else 0.0


# =============================== Spectral helpers =============================

def snr_via_fft(ys: np.ndarray, fs_hz: float) -> Optional[float]:
    """
    Spectral SNR of dominant tone: 10*log10(peak_power / median_floor).
    Hann window; power normalized by window energy; DC excluded from peak search.
    """
    if ys is None or ys.size < 64 or fs_hz <= 0:
        return None
    N = min(ys.size, 2048)
    seg = np.asarray(ys[-N:], dtype=float)
    seg = seg - float(np.mean(seg))
    w = np.hanning(N)
    Y = np.fft.rfft(seg * w)
    P = (np.abs(Y) ** 2) / np.sum(w ** 2)
    if P.size < 3:
        return None
    P[0] = 0.0
    k_peak = int(np.argmax(P))
    peak_power = float(P[k_peak])
    noise_bins = np.delete(P, [0, k_peak])
    noise_power = float(np.median(noise_bins)) if noise_bins.size else 0.0
    if noise_power <= 0.0:
        return float('inf')
    return 10.0 * math.log10(peak_power / noise_power)


def compute_psd_v2_per_hz(ys: np.ndarray, fs_hz: float, nmax: int = 8192) -> Tuple[np.ndarray, np.ndarray]:
    """
    One-sided windowed periodogram in **V^2/Hz**.
    - ys: voltage samples (already scaled to physical volts in your pipeline)
    - fs_hz: sampling rate (Hz)
    Returns (freqs_Hz, psd_V2_per_Hz)
    """
    if ys is None or ys.size < 64 or fs_hz <= 0:
        return np.empty(0), np.empty(0)

    N = min(ys.size, nmax)
    y = np.asarray(ys[-N:], float)
    y = y - float(np.mean(y))            # remove DC for nicer PSD

    w = np.hanning(N)
    Y = np.fft.rfft(y * w)

    # Periodogram scaling for windowed data:
    # PSD = |FFT|^2 / (fs * sum(w^2))
    psd = (np.abs(Y) ** 2) / (fs_hz * np.sum(w ** 2))

    # Make it **one-sided** (double all bins except DC and Nyquist)
    if psd.size > 2:
        psd[1:-1] *= 2.0

    freqs = np.fft.rfftfreq(N, d=1.0 / fs_hz)
    return freqs, psd

# =========================== Stats bar (outside plot) =========================

class StatsBar(QtWidgets.QWidget):
    """Compact, high-contrast info: instantaneous Value, SNR (dB), Rate (pts/s)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setStyleSheet("""
            QWidget { background-color: #2b2b2b; border-top: 1px solid #3a3a3a; }
            QLabel  { color: #f0f0f0; font-size: 11px; }
        """)
        self.value_label = QtWidgets.QLabel("Value: —")
        self.snr_label   = QtWidgets.QLabel("SNR: —")
        self.rate_label  = QtWidgets.QLabel("Rate: —")
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 3, 8, 4)
        layout.setSpacing(14)
        layout.addWidget(self.value_label)
        layout.addWidget(self.snr_label)
        layout.addWidget(self.rate_label)
        layout.addStretch(1)

    def update_stats(self,
                     instantaneous_value: Optional[float],
                     unit: str,
                     snr_db: Optional[float],
                     rate_hz: Optional[float]) -> None:
        self.value_label.setText("Value: —" if instantaneous_value is None
                                 else f"Value: {instantaneous_value:.3g} {unit}")
        if snr_db is None:
            # keep existing text – caller controls when to overwrite to avoid blinking
            pass
        else:
            self.snr_label.setText("SNR: ∞" if not math.isfinite(snr_db)
                                   else f"SNR: {snr_db:.1f} dB")
        self.rate_label.setText("Rate: —" if rate_hz is None
                                else f"Rate: {rate_hz:.1f} pts/s")


# =============================== PSD / FFT window =============================

class PSDWindow(QtWidgets.QMainWindow):
    """
    External PSD window. It **hides on close** (no deletion) to avoid dangling
    pointers. Timer pauses when hidden and resumes when shown.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PSD / FFT")
        # self.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # DO NOT delete; we hide instead

        # Store last traces from each plot: index → (xs, ys, fs_hz)
        self._latest: Dict[int, Tuple[np.ndarray, np.ndarray, float]] = {}

        # --- UI ---
        central = QtWidgets.QWidget(self); self.setCentralWidget(central)
        v = QtWidgets.QVBoxLayout(central); v.setContentsMargins(6,6,6,6); v.setSpacing(6)

        # Controls
        top = QtWidgets.QHBoxLayout(); top.setSpacing(10); v.addLayout(top)
        top.addWidget(QtWidgets.QLabel("Source Plot:"))
        self.source_combo = QtWidgets.QComboBox(); self.source_combo.addItems([f"Channel {i+1}" for i in range(4)])
        top.addWidget(self.source_combo)

        top.addWidget(QtWidgets.QLabel("Spectrum Window:"))
        self.time_windows = [0.5, 1, 2, 5, 10, 20, 60]  # seconds used for FFT
        self.window_combo = QtWidgets.QComboBox()
        for s in self.time_windows: self.window_combo.addItem(f"{s} s")
        self.window_combo.setCurrentIndex(3)  # 5 s default
        top.addWidget(self.window_combo)

        top.addWidget(QtWidgets.QLabel("Freq span:"))
        self.freq_spans = ["Nyquist", 50, 100, 200, 500, 1000, 2000]  # Hz
        self.freq_combo = QtWidgets.QComboBox()
        for fs in self.freq_spans: self.freq_combo.addItem(str(fs))
        self.freq_combo.setCurrentIndex(0)  # Nyquist by default
        top.addWidget(self.freq_combo)

        self.log_check = QtWidgets.QCheckBox("Log amplitude (dB)")
        top.addWidget(self.log_check)
        top.addStretch(1)

        # Plot
        self.plot = pg.PlotWidget(title="Power Spectrum")
        self.plot.showGrid(x=True, y=True)
        self.plot.setLabel('bottom', "Frequency", units="Hz")
        self.plot.setLabel('left', "Power")
        self.plot.setBackground('#222222')
        self.curve = self.plot.plot(pen=pg.mkPen('#1abc9c', width=2))
        v.addWidget(self.plot)

        # Timer: compute/refresh PSD at modest rate to save CPU
        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.timer.timeout.connect(self._update_psd)
        self.timer.start(250)  # 4 Hz

    def hideEvent(self, e: QtGui.QHideEvent) -> None:
        if self.timer.isActive():
            self.timer.stop()
        super().hideEvent(e)

    def showEvent(self, e: QtGui.QShowEvent) -> None:
        if not self.timer.isActive():
            self.timer.start(250)
        super().showEvent(e)

    def closeEvent(self, e: QtGui.QCloseEvent) -> None:
        # Hide instead of delete to avoid "wrapped C/C++ object has been deleted"
        e.ignore()
        self.hide()

    # Called by the main window each tick (only when FFT is enabled)
    def set_trace_data(self, plot_index: int, xs: np.ndarray, ys: np.ndarray, fs_hz: float) -> None:
        self._latest[plot_index] = (xs.copy(), ys.copy(), float(fs_hz))

    def _update_psd(self) -> None:
        idx = self.source_combo.currentIndex()
        if idx not in self._latest:
            self.curve.setData([], [])
            return

        xs, ys, fs = self._latest[idx]
        if ys.size < 8 or fs <= 0:        # allow small windows; will still need ≥16 below
            self.curve.setData([], [])
            return

        # Keep only samples inside the requested "spectrum window"
        span = self.time_windows[self.window_combo.currentIndex()]
        mask = xs >= -span                # xs are negative to 0
        ys_win = ys[mask]
        if ys_win.size < 32:              # need some data; at 20 Hz, 32 ≈ 1.6 s
            self.curve.setData([], [])
            return

        # Compute PSD in V^2/Hz
        freqs, psd = compute_psd_v2_per_hz(ys_win, fs, nmax=8192)

        # Clip to requested frequency span, never exceed Nyquist
        choice = self.freq_spans[self.freq_combo.currentIndex()]
        fmax = fs * 0.5 if choice == "Nyquist" else min(float(choice), fs * 0.5)
        sel = freqs <= fmax
        freqs = freqs[sel]
        psd   = psd[sel]                  # ← FIXED (was: power = power[sel])

        # Plot (linear or dB of a power quantity)
        if self.log_check.isChecked():
            self.plot.setLabel('left', "PSD", units="dB(V²/Hz)")
            self.curve.setData(freqs, 10.0 * np.log10(np.maximum(psd, 1e-30)))
        else:
            self.plot.setLabel('left', "PSD", units="V²/Hz")
            self.curve.setData(freqs, psd)



# ================================ Main Window ================================

class ScopeApp(QtWidgets.QMainWindow):
    """
    4-channel time scope with:
      • optional Relative Z (Topo),
      • spectral SNR + togglable PSD window,
      • per-plot stats: Value, SNR (cached), Rate.
    """
    def __init__(self, source: IDataSource, driver_handle=None):
        super().__init__()
        self.source = source
        self.driver_handle = driver_handle
        self.setWindowTitle("SXM Scope — spectral SNR + PSD")

        # ---- Timing ----
        self.sample_period_s       = 0.05   # ~20 Hz GUI/update
        self.max_history_seconds   = 600
        self.time_window_options_s = [1, 5, 10, 30, 60, 120, 600]
        self.max_history_samples   = int(self.max_history_seconds / self.sample_period_s) + 2
        self.tail_frames_cache     = {}
        self.frame_idx             = 0

        # ---- Channels / plots ----
        self.num_plots  = 4
        self.chan_names = list(channels.keys())
        self.raw_t = [deque(maxlen=self.max_history_samples) for _ in range(self.num_plots)]
        self.raw_v = [deque(maxlen=self.max_history_samples) for _ in range(self.num_plots)]
        self.rel_t = [deque(maxlen=self.max_history_samples) for _ in range(self.num_plots)]
        self.rel_v = [deque(maxlen=self.max_history_samples) for _ in range(self.num_plots)]
        self.hpf   = [IIRHighPass(1.0, self.sample_period_s) for _ in range(self.num_plots)]

        # Spectral SNR settings
        self.fft_enabled   = False
        self.snr_every     = 10                  # compute SNR every N frames
        self.last_snr_db   = [None]*self.num_plots  # cache value to avoid blinking

        # Appearance
        self.line_colors = ['#3498db','#e67e22','#16a085','#8e44ad','#c0392b','#27ae60']
        self.bg_color    = '#222222'
        self.line_width  = 2

        # UI
        self._build_controls()
        self._build_plots_grid()
        self._build_menu()
        self._status_source()

        # PSD window (created once; hide/show thereafter)
        self.psd_window: Optional[PSDWindow] = None

        # Timer
        self.timer = QtCore.QTimer(self); self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.timer.timeout.connect(self._tick)
        self.timer.start(int(self.sample_period_s * 1000))

        # Cache axis labels to avoid churn
        self._last_ylabel_text = [None]*self.num_plots
        self._last_ylabel_unit = [None]*self.num_plots

    # -------------------------- UI builders --------------------------

    def _build_controls(self):
        central = QtWidgets.QWidget(self); self.setCentralWidget(central)
        self.vbox = QtWidgets.QVBoxLayout(central); self.vbox.setContentsMargins(6,6,6,6); self.vbox.setSpacing(6)

        top = QtWidgets.QHBoxLayout(); top.setSpacing(10); self.vbox.addLayout(top)

        self.window_combo = QtWidgets.QComboBox()
        for s in self.time_window_options_s: self.window_combo.addItem(f"{s} s")
        self.window_combo.setCurrentIndex(2)
        top.addWidget(QtWidgets.QLabel("Time Window:")); top.addWidget(self.window_combo)

        self.chan_combos: List[QtWidgets.QComboBox] = []
        for i in range(self.num_plots):
            cb = QtWidgets.QComboBox(); cb.addItems(self.chan_names)
            top.addWidget(QtWidgets.QLabel(f"Ch{i+1}:")); top.addWidget(cb)
            self.chan_combos.append(cb)

        self.rel_check = QtWidgets.QCheckBox("Relative Z (Topo)")
        self.rel_check.stateChanged.connect(self._reset_relative_z)
        top.addWidget(self.rel_check)

        # Spectral SNR / PSD toggle + button
        self.fft_check = QtWidgets.QCheckBox("FFT / PSD (SNR)")
        self.fft_check.stateChanged.connect(self._toggle_fft_enabled)
        top.addWidget(self.fft_check)

        self.psd_btn = QtWidgets.QPushButton("Open PSD…")
        self.psd_btn.clicked.connect(self._open_psd_window)
        self.psd_btn.setEnabled(False)
        top.addWidget(self.psd_btn)

        top.addStretch(1)

    def _build_plots_grid(self):
        grid = QtWidgets.QGridLayout(); grid.setHorizontalSpacing(10); grid.setVerticalSpacing(8)
        self.vbox.addLayout(grid)
        self.plots:  List[pg.PlotWidget]    = []
        self.curves: List[pg.PlotDataItem]  = []
        self.stats:  List[StatsBar]         = []

        for i in range(self.num_plots):
            container = QtWidgets.QWidget(); vb = QtWidgets.QVBoxLayout(container)
            vb.setContentsMargins(0,0,0,0); vb.setSpacing(0)

            pw = pg.PlotWidget(title=f"Channel {i+1}")
            pw.showGrid(x=True, y=True); pw.setBackground(self.bg_color)
            pw.setLabel('bottom', "Time", units="s")
            curve = pw.plot(pen=pg.mkPen(self.line_colors[i], width=self.line_width))
            curve.setClipToView(True)
            try:
                curve.setDownsampling(auto=True, method='peak')
                curve.setSkipFiniteCheck(True)
            except Exception:
                pass

            vb.addWidget(pw)
            stats_bar = StatsBar()
            vb.addWidget(stats_bar)

            grid.addWidget(container, i//2, i%2)
            self.plots.append(pw); self.curves.append(curve); self.stats.append(stats_bar)

    def _build_menu(self):
        m = self.menuBar()
        view = m.addMenu("View")
        act = QtWidgets.QAction("Appearance…", self); act.triggered.connect(self._appearance_dialog)
        view.addAction(act)

        src = m.addMenu("Source")
        a1 = QtWidgets.QAction("Use Driver", self); a2 = QtWidgets.QAction("Use Mock Data", self)
        a1.triggered.connect(self._switch_driver); a2.triggered.connect(self._switch_mock)
        src.addAction(a1); src.addAction(a2)

    # ------------------------------ Menu/controls ------------------------------

    def _toggle_fft_enabled(self, state: int) -> None:
        self.fft_enabled = (state == QtCore.Qt.Checked)
        self.psd_btn.setEnabled(self.fft_enabled)
        if not self.fft_enabled and self.psd_window is not None:
            self.psd_window.hide()  # stop its timer via hideEvent

    def _open_psd_window(self) -> None:
        if not self.fft_enabled:
            return
        if self.psd_window is None:
            self.psd_window = PSDWindow(self)
        self.psd_window.show()
        self.psd_window.raise_()
        self.psd_window.activateWindow()

    def _appearance_dialog(self):
        d = QtWidgets.QDialog(self); d.setWindowTitle("Plot Appearance"); f = QtWidgets.QFormLayout(d)
        btns = []
        for i in range(6):
            b = QtWidgets.QPushButton(); b.setStyleSheet(f"background-color:{self.line_colors[i]}"); btns.append(b)
            f.addRow(f"Line {i+1} Color:", b)
        bg = QtWidgets.QPushButton(); bg.setStyleSheet(f"background-color:{self.bg_color}"); f.addRow("Background:", bg)
        w  = QtWidgets.QSpinBox(); w.setRange(1,10); w.setValue(self.line_width); f.addRow("Line Thickness:", w)
        for i,b in enumerate(btns):
            b.clicked.connect(lambda _,i=i: self._pick_color(i, btns))
        bg.clicked.connect(lambda: self._pick_bg(bg))
        ok = QtWidgets.QPushButton("OK"); ok.clicked.connect(lambda: self._apply_appearance(w.value(), d)); f.addRow(ok)
        d.exec_()

    def _pick_color(self, i:int, btns:List[QtWidgets.QPushButton]):
        c = QtWidgets.QColorDialog.getColor()
        if c.isValid():
            css = c.name(); btns[i].setStyleSheet(f"background-color:{css}"); self.line_colors[i] = css

    def _pick_bg(self, bg_btn:QtWidgets.QPushButton):
        c = QtWidgets.QColorDialog.getColor()
        if c.isValid():
            css = c.name(); bg_btn.setStyleSheet(f"background-color:{css}"); self.bg_color = css

    def _apply_appearance(self, lw:int, dialog:QtWidgets.QDialog):
        self.line_width = lw
        for i,(pw,cv) in enumerate(zip(self.plots,self.curves)):
            pw.setBackground(self.bg_color); cv.setPen(pg.mkPen(self.line_colors[i], width=self.line_width))
        if self.psd_window is not None:
            self.psd_window.plot.setBackground(self.bg_color)
        dialog.accept()

    def _switch_mock(self):
        self.source = MockDataSource(); self._status_source()

    def _switch_driver(self):
        if not WIN32_AVAILABLE:
            QtWidgets.QMessageBox.warning(self,"Driver","win32 modules not available."); return
        try:
            h = win32file.CreateFile(r"\\.\SXM",
                                     win32con.GENERIC_READ|win32con.GENERIC_WRITE,
                                     0, None, win32con.OPEN_EXISTING,
                                     win32con.FILE_ATTRIBUTE_NORMAL, None)
            self.source = DriverDataSource(h); self.driver_handle = h; self._status_source()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self,"Driver",f"Could not open driver: {e}")

    def _status_source(self):
        src = "Driver" if isinstance(self.source, DriverDataSource) else "Mock"
        self.statusBar().showMessage(f"Data Source: {src}")

    def _reset_relative_z(self):
        for f in self.hpf: f.reset()
        for dq in self.rel_t: dq.clear()
        for dq in self.rel_v: dq.clear()

    # ------------------------------ Main update -------------------------------

    def _tick(self):
        """
        One acquisition/draw tick:
          1) Deduplicate channel reads
          2) Update histories
          3) Draw tails
          4) Update Value + Rate
          5) Compute SNR (if enabled, throttled) and keep last value to avoid flicker
          6) Feed PSD window (if open)
        """
        self.frame_idx += 1
        now = time.time()
        window_s = self.time_window_options_s[self.window_combo.currentIndex()]
        tail = self.tail_frames_cache.get(window_s)
        if tail is None:
            tail = int(window_s / self.sample_period_s) + 2
            self.tail_frames_cache[window_s] = tail

        # 1) Unique channels needed
        needed: Dict[int, Tuple[str,str,float]] = {}
        for cb in self.chan_combos:
            nm = cb.currentText(); idx, _, unit, scale = channels[nm]
            needed[idx] = (nm, unit, scale)

        # Read once per channel
        scaled_snapshot: Dict[int, float] = {}
        for idx, (_nm, _unit, scale) in needed.items():
            raw = self.source.read_value(idx)
            scaled_snapshot[idx] = raw * scale

        # 2–6) Per-plot handling
        for i, cb in enumerate(self.chan_combos):
            nm = cb.currentText()
            idx, _, unit, _scale = channels[nm]
            y_raw = scaled_snapshot[idx]

            # Append raw
            self.raw_t[i].append(now)
            self.raw_v[i].append(y_raw)

            # Optional Relative Z
            use_rel = (nm == 'Topo' and self.rel_check.isChecked())
            if use_rel:
                y_rel = self.hpf[i].process(y_raw)
                self.rel_t[i].append(now)
                self.rel_v[i].append(y_rel)

            # Choose stream (and series for PSD)
            if use_rel:
                xs, ys = deque_tail_to_arrays(self.rel_t[i], self.rel_v[i], tail, now)
                ytxt, yunit = 'ΔZ', 'nm'
                series_for_psd = ys
            else:
                xs, ys = deque_tail_to_arrays(self.raw_t[i], self.raw_v[i], tail, now)
                ytxt, yunit = nm, unit
                series_for_psd = ys

            # 3) Draw
            self.curves[i].setData(xs, ys)
            self.plots[i].setXRange(-window_s, 0, padding=0)
            if self._last_ylabel_text[i] != ytxt or self._last_ylabel_unit[i] != yunit:
                self.plots[i].setLabel('left', ytxt, yunit)
                self._last_ylabel_text[i], self._last_ylabel_unit[i] = ytxt, yunit
            if (self.frame_idx % 3) == 0:
                self.plots[i].enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

            # 4) Rate + Value
            rate_hz = estimate_rate_hz(xs)
            v_inst = float(ys[-1]) if ys.size else None

            # 5) Spectral SNR (throttled) – keep last value to avoid blinking
            if self.fft_enabled and ys.size >= 64 and (self.frame_idx % self.snr_every) == 0:
                fs = rate_hz if rate_hz > 0 else (1.0 / self.sample_period_s)
                new_snr = snr_via_fft(series_for_psd, fs)
                if new_snr is not None:
                    self.last_snr_db[i] = new_snr  # cache
            # Update stats (use cached SNR)
            self.stats[i].update_stats(v_inst, yunit, self.last_snr_db[i], rate_hz)

            # 6) Feed PSD window if visible
            if self.fft_enabled and self.psd_window is not None and self.psd_window.isVisible():
                fs = rate_hz if rate_hz > 0 else (1.0 / self.sample_period_s)
                self.psd_window.set_trace_data(i, xs, series_for_psd, fs)

    # ------------------------------ Window events ------------------------------

    def changeEvent(self, e: QtCore.QEvent) -> None:
        """Pause when minimized; resume when restored (saves CPU and device I/O)."""
        if e.type() == QtCore.QEvent.WindowStateChange:
            minimized = bool(self.windowState() & QtCore.Qt.WindowMinimized)
            if minimized:
                self.timer.stop()
            else:
                if not self.timer.isActive():
                    self.timer.start(int(self.sample_period_s * 1000))
        super().changeEvent(e)


# ================================ Entry point ================================

def create_data_source():
    """Try driver first; fall back to mock."""
    if WIN32_AVAILABLE:
        try:
            h = win32file.CreateFile(r"\\.\SXM",
                                     win32con.GENERIC_READ|win32con.GENERIC_WRITE,
                                     0, None, win32con.OPEN_EXISTING,
                                     win32con.FILE_ATTRIBUTE_NORMAL, None)
            return DriverDataSource(h), h, "Driver"
        except Exception:
            pass
    return MockDataSource(), None, "Mock"


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    src, handle, srcname = create_data_source()
    w = ScopeApp(src, driver_handle=handle)
    w.resize(1200, 900)
    w.show()
    sys.exit(app.exec_())

"""
SXM Time-Based Multi-Channel Scope (speed-tuned, dedup IO, deque tail slicing)

Two profiles:
-----------
FAST_PROFILE = 'balanced'   -> keeps comparison + stats, but throttles/approximates
FAST_PROFILE = 'max'        -> removes comparison plot and stats bars for maximum throughput

What’s optimized:
-----------------
• Per-tick device I/O is **deduplicated** (each needed channel read once).
• Deques with maxlen store histories; we **extract only the tail** we need to plot.
• Stats use **EWMA** (exponentially weighted) and are **throttled** (fewer updates).
• Labels/axes only update when content actually changes (avoid needless work).
• PyQtGraph clipping/downsampling enabled; antialias off; OpenGL on when available.
"""

# ------------------------- CONFIG: choose your profile -------------------------
FAST_PROFILE = 'balanced'   # 'balanced' or 'max'
# ------------------------------------------------------------------------------

import sys
import time
import math
import ctypes
from typing import Optional, Tuple, List, Dict
from collections import deque
from itertools import islice

# Try Windows driver modules; fall back to mock if unavailable
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
    """
    Reads raw values via the SXM device driver (Windows DeviceIoControl).
    Reuses an input buffer to avoid per-call allocations.
    """
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
    """
    Synthetic generator: slow drift + sinusoid + Gaussian noise (raw ints).
    """
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
    """
    1st-order high-pass: y[n] = α * (y[n-1] + x[n] − x[n-1])
    Used for Relative Z on "Topo".
    """
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


# =========================== Stats bar (optional) ============================

class StatsBar(QtWidgets.QWidget):
    """
    Lightweight bar under each plot showing Value, σ (EWMA), and SNR (dB).
    NOTE: In 'max' profile we do not create/update this to save work.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.value_label = QtWidgets.QLabel("Value: —")
        self.noise_label = QtWidgets.QLabel("σ: —")
        self.snr_label   = QtWidgets.QLabel("SNR: —")
        for lbl in (self.value_label, self.noise_label, self.snr_label):
            lbl.setStyleSheet("color: rgba(220,220,220,0.9); font-size: 11px;")
        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(6,0,6,4); h.setSpacing(12)
        h.addWidget(self.value_label); h.addWidget(self.noise_label); h.addWidget(self.snr_label); h.addStretch(1)

    def update_stats(self, v_inst: Optional[float], unit: str,
                     sigma: Optional[float], snr_db: Optional[float]) -> None:
        self.value_label.setText("Value: —" if v_inst is None else f"Value: {v_inst:.3g} {unit}")
        self.noise_label.setText("σ: —" if sigma is None else f"σ: {sigma:.3g} {unit}")
        if snr_db is None:
            self.snr_label.setText("SNR: —")
        else:
            self.snr_label.setText("SNR: ∞" if not math.isfinite(snr_db) else f"SNR: {snr_db:.1f} dB")


# ========================= Deque → NumPy tail helper =========================

def deque_tail_to_arrays(t_deq: deque, v_deq: deque, tail_count: int, now: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract only the **last** `tail_count` items from time/value deques.
    This is O(tail_count), independent of total history length.
    """
    if not t_deq:
        return np.empty(0), np.empty(0)
    k = min(tail_count, len(t_deq))
    t_rev = list(islice(reversed(t_deq),  0, k))
    v_rev = list(islice(reversed(v_deq),  0, k))
    t_arr = np.asarray(t_rev[::-1], dtype=float)
    v_arr = np.asarray(v_rev[::-1], dtype=float)
    return (t_arr - now), v_arr


# ================================ Main Window ================================

class ScopeApp(QtWidgets.QMainWindow):
    """
    Speed-tuned oscilloscope for 4 SXM channels + (optional) dual-axis comparison.

    Key speed ideas:
    - **Deduplicate I/O** per tick: build set of required channel indices and read each once.
    - **Deque(maxlen)** histories and plot **tail only** (visible window).
    - **EWMA** sigma/SNR computed **throttled** (not every frame), optional in 'max'.
    - Avoid per-frame label churn; only update when the channel/unit actually changes.
    """
    def __init__(self, source: IDataSource, driver_handle=None):
        super().__init__()
        self.source = source
        self.driver_handle = driver_handle
        self.setWindowTitle("SXM Scope — speed tuned")

        # ---------- Timing ----------
        self.sample_period_s       = 0.05    # 20 Hz acquisition/GUI
        self.max_history_seconds   = 600
        self.time_window_options_s = [1, 5, 10, 30, 60, 120, 600]
        self.max_history_samples   = int(self.max_history_seconds / self.sample_period_s) + 2
        self.tail_frames_cache     = {}      # cache: window_s -> tail_count

        # ---------- Channels / plots ----------
        self.num_plots = 4
        self.chan_names = list(channels.keys())

        # per-plot histories (raw and relative-Z)
        self.raw_t = [deque(maxlen=self.max_history_samples) for _ in range(self.num_plots)]
        self.raw_v = [deque(maxlen=self.max_history_samples) for _ in range(self.num_plots)]
        self.rel_t = [deque(maxlen=self.max_history_samples) for _ in range(self.num_plots)]
        self.rel_v = [deque(maxlen=self.max_history_samples) for _ in range(self.num_plots)]
        self.hpf   = [IIRHighPass(1.0, self.sample_period_s) for _ in range(self.num_plots)]

        # EWMA stats (per-plot), throttled updates
        self.stats_alpha = 0.15
        self.ewma_mean   = [0.0]*self.num_plots
        self.ewma_var    = [0.0]*self.num_plots
        self.ewma_init   = [False]*self.num_plots
        self.stats_every = 5    # compute text stats every N frames (balanced)
        self.frame_idx   = 0

        # Appearance
        self.line_colors  = ['#3498db','#e67e22','#16a085','#8e44ad','#c0392b','#27ae60']  # last 2 for comparison
        self.bg_color     = '#222222'
        self.line_width   = 2

        # Build UI
        self._build_controls()
        self._build_plots_grid()
        self._build_comparison_plot() if FAST_PROFILE != 'max' else None
        self._build_menu()

        self._status_source()

        # Timer
        self.timer = QtCore.QTimer(self); self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.timer.timeout.connect(self._tick)
        self.timer.start(int(self.sample_period_s * 1000))

        # cache last labels to avoid churn
        self._last_ylabel_text = [None]*self.num_plots
        self._last_ylabel_unit = [None]*self.num_plots

    # -------------------------- UI building blocks --------------------------

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

    def _build_plots_grid(self):
        grid = QtWidgets.QGridLayout(); grid.setHorizontalSpacing(10); grid.setVerticalSpacing(8)
        self.vbox.addLayout(grid)
        self.plots:  List[pg.PlotWidget]    = []
        self.curves: List[pg.PlotDataItem]  = []
        self.stats:  List[StatsBar]         = []

        for i in range(self.num_plots):
            container = QtWidgets.QWidget(); vb = QtWidgets.QVBoxLayout(container)
            vb.setContentsMargins(0,0,0,0); vb.setSpacing(2)

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

            if FAST_PROFILE == 'max':
                stats_bar = None
            else:
                stats_bar = StatsBar(); vb.addWidget(stats_bar)

            grid.addWidget(container, i//2, i%2)
            self.plots.append(pw); self.curves.append(curve); self.stats.append(stats_bar)

    def _build_comparison_plot(self):
        self.vbox.addWidget(QtWidgets.QLabel("Comparison Plot:"))
        top = QtWidgets.QHBoxLayout(); top.setSpacing(10); self.vbox.addLayout(top)
        self.cmp_left  = QtWidgets.QComboBox();  self.cmp_left.addItems(self.chan_names)
        self.cmp_right = QtWidgets.QComboBox();  self.cmp_right.addItems(self.chan_names)
        top.addWidget(self.cmp_left); top.addWidget(self.cmp_right)

        self.cmp_plot = pg.PlotWidget(title="Compare Channels")
        item = self.cmp_plot.getPlotItem()
        item.showGrid(x=True, y=True); item.setLabel('bottom', "Time", units="s")
        self.cmp_plot.setBackground(self.bg_color)
        item.showAxis('right')
        self.vb_right = pg.ViewBox(); item.scene().addItem(self.vb_right)
        item.getAxis('right').linkToView(self.vb_right)
        self.vb_right.setXLink(item.vb)
        item.vb.sigResized.connect(lambda: self.vb_right.setGeometry(item.vb.sceneBoundingRect()))
        self.cmp_curve_l = pg.PlotDataItem(pen=pg.mkPen(self.line_colors[4], width=self.line_width))
        self.cmp_curve_r = pg.PlotDataItem(pen=pg.mkPen(self.line_colors[5], width=self.line_width))
        item.addItem(self.cmp_curve_l); self.vb_right.addItem(self.cmp_curve_r)
        self.vbox.addWidget(self.cmp_plot)

        # comparison histories (deques)
        self.cmp_t = [deque(maxlen=self.max_history_samples) for _ in range(2)]
        self.cmp_v = [deque(maxlen=self.max_history_samples) for _ in range(2)]

    def _build_menu(self):
        m = self.menuBar()
        view = m.addMenu("View")
        act = QtWidgets.QAction("Appearance…", self); act.triggered.connect(self._appearance_dialog)
        view.addAction(act)
        src = m.addMenu("Source")
        a1 = QtWidgets.QAction("Use Driver", self); a2 = QtWidgets.QAction("Use Mock Data", self)
        a1.triggered.connect(self._switch_driver); a2.triggered.connect(self._switch_mock)
        src.addAction(a1); src.addAction(a2)

    # ------------------------------ Menu handlers ------------------------------

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
        if FAST_PROFILE != 'max':
            self.cmp_plot.setBackground(self.bg_color)
            self.cmp_curve_l.setPen(pg.mkPen(self.line_colors[4], width=self.line_width))
            self.cmp_curve_r.setPen(pg.mkPen(self.line_colors[5], width=self.line_width))
        dialog.accept()

    def _switch_mock(self):
        self.source = MockDataSource(); self._status_source()

    def _switch_driver(self):
        if not WIN32_AVAILABLE:
            QtWidgets.QMessageBox.warning(self,"Driver","win32 modules not available."); return
        try:
            h = win32file.CreateFile(r"\\.\SXM", win32con.GENERIC_READ|win32con.GENERIC_WRITE,
                                     0,None, win32con.OPEN_EXISTING, win32con.FILE_ATTRIBUTE_NORMAL, None)
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
        One GUI/acquisition tick:
        1) Gather ALL required channel indices from 4 plots (+ comparison if present).
        2) **Read each unique channel ONCE** from the device.
        3) Update per-plot histories (raw and ΔZ if enabled).
        4) Plot only the **tail** needed for the visible time window.
        5) Update stats (EWMA) only every N frames (balanced profile).
        """
        self.frame_idx += 1
        now = time.time()
        window_s = self.time_window_options_s[self.window_combo.currentIndex()]
        tail = self.tail_frames_cache.get(window_s)
        if tail is None:
            tail = int(window_s / self.sample_period_s) + 2
            self.tail_frames_cache[window_s] = tail

        # 1) Collect needed channels
        needed: Dict[int, Tuple[str,str,float]] = {}
        for cb in self.chan_combos:
            nm = cb.currentText(); idx, _, unit, scale = channels[nm]
            needed[idx] = (nm, unit, scale)
        if FAST_PROFILE != 'max':
            for cb in (self.cmp_left, self.cmp_right):
                nm = cb.currentText(); idx, _, unit, scale = channels[nm]
                needed[idx] = (nm, unit, scale)

        # 2) Read each unique index ONCE, scale immediately
        snapshot_scaled: Dict[int, float] = {}
        for idx, (_nm, _unit, scale) in needed.items():
            raw = self.source.read_value(idx)
            snapshot_scaled[idx] = raw * scale

        # 3) Update per-plot histories (raw + ΔZ if requested)
        for i, cb in enumerate(self.chan_combos):
            nm = cb.currentText()
            idx, _, unit, scale = channels[nm]
            val = snapshot_scaled[idx]

            # raw stream append
            self.raw_t[i].append(now)
            self.raw_v[i].append(val)

            use_rel = (nm == 'Topo' and self.rel_check.isChecked())
            if use_rel:
                y = self.hpf[i].process(val)
                self.rel_t[i].append(now)
                self.rel_v[i].append(y)

            # select which stream to visualize
            if use_rel:
                xs, ys = deque_tail_to_arrays(self.rel_t[i], self.rel_v[i], tail, now)
                ytxt, yunit = 'ΔZ', 'nm'
            else:
                xs, ys = deque_tail_to_arrays(self.raw_t[i], self.raw_v[i], tail, now)
                ytxt, yunit = nm, unit

            # 4) Draw only tail and avoid repeated label calls
            self.curves[i].setData(xs, ys)
            self.plots[i].setXRange(-window_s, 0, padding=0)
            if self._last_ylabel_text[i] != ytxt or self._last_ylabel_unit[i] != yunit:
                self.plots[i].setLabel('left', ytxt, yunit)
                self._last_ylabel_text[i], self._last_ylabel_unit[i] = ytxt, yunit
            # modest autorange cadence (every 3 frames) to reduce work
            if (self.frame_idx % 3) == 0:
                self.plots[i].enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

            # 5) Stats (balanced only)
            if FAST_PROFILE != 'max' and self.stats[i] is not None:
                if ys.size:
                    v = float(ys[-1])
                    # EWMA update throttled every stats_every frames
                    if (self.frame_idx % self.stats_every) == 0:
                        if not self.ewma_init[i]:
                            self.ewma_init[i] = True
                            self.ewma_mean[i] = v
                            self.ewma_var[i]  = 0.0
                        else:
                            a = self.stats_alpha
                            m = self.ewma_mean[i]
                            m2 = (1-a)*m + a*v
                            self.ewma_var[i] = (1-a)*self.ewma_var[i] + a*(v - m2)*(v - m)
                            self.ewma_mean[i] = m2
                        sigma = (self.ewma_var[i] ** 0.5)
                        meanv = self.ewma_mean[i]
                        snr_db = (20.0 * math.log10(abs(meanv)/sigma)) if sigma > 0 else float('inf')
                        self.stats[i].update_stats(v, yunit, sigma, snr_db)
                else:
                    if (self.frame_idx % self.stats_every) == 0:
                        self.stats[i].update_stats(None, yunit, None, None)

        # Comparison plot (balanced only)
        if FAST_PROFILE != 'max':
            left_nm  = self.cmp_left.currentText()
            right_nm = self.cmp_right.currentText()
            li, _, _, ls = channels[left_nm]
            ri, _, _, rs = channels[right_nm]
            lv = snapshot_scaled.get(li)
            rv = snapshot_scaled.get(ri)
            # Append and draw tails
            self.cmp_t[0].append(now); self.cmp_v[0].append(lv)
            self.cmp_t[1].append(now); self.cmp_v[1].append(rv)
            xsL, ysL = deque_tail_to_arrays(self.cmp_t[0], self.cmp_v[0], tail, now)
            xsR, ysR = deque_tail_to_arrays(self.cmp_t[1], self.cmp_v[1], tail, now)
            self.cmp_curve_l.setData(xsL, ysL)
            self.cmp_curve_r.setData(xsL, ysR)
            item = self.cmp_plot.getPlotItem()
            item.setXRange(-window_s, 0, padding=0)
            # only update axis labels when changed
            item.setLabel('left',  left_nm,  channels[left_nm][2])
            item.getAxis('right').setLabel(right_nm, channels[right_nm][2])
            if (self.frame_idx % 3) == 0:
                item.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
                self.vb_right.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

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

def create_data_source() -> Tuple[IDataSource, Optional[object], str]:
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

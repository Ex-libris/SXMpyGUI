"""
SXM Time-Based Multi-Channel Scope (deque-optimized, mock-capable, well-commented)

What this program does
----------------------
• Plots 4 live, time-based signals from an Anfatec SXM controller (or mock data).
• Optional “Relative Z (Topo)” view using a tiny IIR high-pass filter (ΔZ).
• Dual-axis comparison plot to overlay two channels with independent Y-axes.
• For each plot, a compact stats bar *below* the plot shows:
    – Instantaneous value
    – Noise estimate (σ, std. dev. over visible window)
    – SNR estimate (20*log10(|mean|/σ) over the visible window)
• Clean architecture:
    – IDataSource interface + DriverDataSource / MockDataSource
    – (Future) DeviceCommandInterface placeholder for DDE/COM
• Performance tuned for Python:
    – Uses deque(maxlen=…) for O(1) appends and automatic aging
    – Extracts the tail of each deque efficiently (O(N_window), not O(N_history))
    – Reuses IO buffers for DeviceIoControl
    – PyQtGraph downsampling/clipping enabled

Quick start
-----------
Run this file directly. If the "\\.\SXM" driver can’t be opened, the app
falls back to mock data automatically. You can also switch sources from the menu.
"""

import sys
import time
import math
import ctypes
from typing import Optional, Tuple, List
from collections import deque
from itertools import islice

# Try importing Windows driver modules. If not available, we run in mock mode.
try:
    import win32file, win32con
    WIN32_AVAILABLE = True
except Exception:
    WIN32_AVAILABLE = False

from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import numpy as np


# -----------------------------------------------------------------------------
# PyQtGraph global options
# -----------------------------------------------------------------------------
# • useOpenGL: let GPU draw lines if available (faster on many systems)
# • antialias=False: a small visual trade-off for measurable CPU savings
pg.setConfigOptions(useOpenGL=True, antialias=False)


# -----------------------------------------------------------------------------
# Channel catalog: human_name -> (driver_index, short_label, unit, scale_factor)
#   • driver_index: integer code passed to the Windows driver
#   • unit: human unit for display
#   • scale_factor: raw_int * scale_factor -> physical_units (float)
# -----------------------------------------------------------------------------
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


# =============================================================================
# Data source layer (encapsulated)
# =============================================================================

class IDataSource:
    """
    Interface for all data sources (real driver or mock).
    Implementations return a RAW integer from the hardware/driver.
    Scaling to physical units is applied later using 'channels'.
    """
    def read_value(self, channel_index: int) -> int:
        raise NotImplementedError


class DriverDataSource(IDataSource):
    """
    Read raw values via the SXM device driver (Windows DeviceIoControl).
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
        """
        Parameters
        ----------
        driver_handle : Windows HANDLE opened on "\\\\.\\SXM".
        """
        self.driver_handle = driver_handle
        # Reusable 4-byte input buffer for passing the channel index to the driver.
        self._inbuf = ctypes.create_string_buffer(ctypes.sizeof(ctypes.c_long))

    def read_value(self, channel_index: int) -> int:
        """
        Ask the driver for one channel's raw value and return it as a Python int.
        """
        ctypes.memmove(self._inbuf,
                       ctypes.byref(ctypes.c_long(channel_index)),
                       ctypes.sizeof(ctypes.c_long))
        result_bytes = win32file.DeviceIoControl(
            self.driver_handle,
            self.IOCTL_GET_KANAL,
            self._inbuf,
            ctypes.sizeof(ctypes.c_long)
        )
        return ctypes.c_long.from_buffer_copy(result_bytes).value


class MockDataSource(IDataSource):
    """
    Synthetic signal generator used when the driver is unavailable.
    Produces plausible raw integers: slow drift + sinusoid + Gaussian noise.
    """
    def __init__(self, seed: int = 12345):
        self.start_time = time.time()
        self.rng = np.random.default_rng(seed)

    def read_value(self, channel_index: int) -> int:
        """
        Create a deterministic per-channel raw integer sample based on time.
        """
        t = time.time() - self.start_time

        # Channel-specific parameters (repeatable variety)
        base_freq_hz = 0.5 + (abs(channel_index) % 7) * 0.3
        base_amp     = 2_000_000 + (abs(channel_index) % 5) * 500_000
        noise_sd     = 80_000 + (abs(channel_index) % 3) * 40_000

        # Compose drift + sine + noise in "driver counts"
        drift = int(300_000 * math.sin(0.02 * t))
        sine  = int(base_amp * math.sin(2 * math.pi * base_freq_hz * t))
        noise = int(self.rng.normal(0, noise_sd))

        if channel_index == 0:  # emphasize low-freq drift for "Topo"
            drift += int(500_000 * math.sin(0.1 * math.pi * t))

        return drift + sine + noise


# =============================================================================
# (Future) command layer placeholder (e.g., DDE to SXM.exe)
# =============================================================================

class DeviceCommandInterface:
    """
    Placeholder for a future command/control layer (DDE/COM).
    Keep control concerns separate from plotting and acquisition.
    """
    def connect(self) -> None:
        pass
    def send_command(self, command: str) -> None:
        pass
    def query(self, request: str) -> str:
        return ""


# =============================================================================
# Tiny DSP: 1st-order IIR high-pass (for Relative Z)
# =============================================================================

class IIRHighPass:
    """
    First-order high-pass:
        y[n] = α * (y[n-1] + x[n] − x[n-1])
    Used to display Relative Z (ΔZ) for the “Topo” channel only.
    """
    def __init__(self, time_constant_s: float, sample_period_s: float):
        self.alpha = time_constant_s / (time_constant_s + sample_period_s)
        self.reset()

    def reset(self) -> None:
        """Reset internal state so a new segment does not carry previous baselines."""
        self.prev_x = 0.0
        self.prev_y = 0.0

    def process(self, x_now: float) -> float:
        """Filter one sample and return the high-passed output."""
        y_now = self.alpha * (self.prev_y + x_now - self.prev_x)
        self.prev_x, self.prev_y = x_now, y_now
        return y_now


# =============================================================================
# Stats bar widget (outside the plot)
# =============================================================================

class StatsBar(QtWidgets.QWidget):
    """
    Thin bar with text labels placed *below* a plot.
    Shows:
      • current value (with unit),
      • noise σ over the visible window,
      • SNR estimate (20*log10(|mean|/σ)) over the visible window.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        self.value_label = QtWidgets.QLabel("Value: —")
        self.noise_label = QtWidgets.QLabel("σ: —")
        self.snr_label   = QtWidgets.QLabel("SNR: —")

        # Subtle, readable style
        for lbl in (self.value_label, self.noise_label, self.snr_label):
            lbl.setStyleSheet("color: rgba(220,220,220, 0.9); font-size: 11px;")

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(6, 0, 6, 4)
        layout.setSpacing(12)
        layout.addWidget(self.value_label)
        layout.addWidget(self.noise_label)
        layout.addWidget(self.snr_label)
        layout.addStretch(1)

    def update_stats(self,
                     instantaneous_value: Optional[float],
                     unit: str,
                     sigma: Optional[float],
                     snr_db: Optional[float]) -> None:
        """Update the three textual fields (use None to indicate no data)."""
        self.value_label.setText("Value: —" if instantaneous_value is None
                                 else f"Value: {instantaneous_value:.3g} {unit}")
        self.noise_label.setText("σ: —" if sigma is None
                                 else f"σ: {sigma:.3g} {unit}")
        if snr_db is None:
            self.snr_label.setText("SNR: —")
        else:
            self.snr_label.setText("SNR: ∞" if not math.isfinite(snr_db)
                                   else f"SNR: {snr_db:.1f} dB")


# =============================================================================
# Helpers focused on deque performance
# =============================================================================

def tail_deque_to_arrays(time_deq: deque, value_deq: deque,
                         max_points: int, now: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert the *last* max_points of two deques (time and value) into NumPy arrays.

    Why this is fast:
    -----------------
    • We iterate from the RIGHT using reversed(deque) and take only 'max_points'
      with itertools.islice. This is O(max_points), not O(len(deque)).
    • Then we reverse that short list to restore chronological order.

    Parameters
    ----------
    time_deq : deque of float timestamps
    value_deq: deque of float values
    max_points : number of newest points to extract
    now : current wall-clock time (seconds)

    Returns
    -------
    xs : np.ndarray of time offsets (t - now), shape (K,)
    ys : np.ndarray of values, shape (K,)
    """
    if not time_deq:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    k = min(max_points, len(time_deq))

    # Take only the tail in reverse (rightmost entries). This is O(k).
    times_rev  = list(islice(reversed(time_deq),  0, k))
    values_rev = list(islice(reversed(value_deq), 0, k))

    # Flip to chronological order and convert to arrays
    times = np.asarray(times_rev[::-1], dtype=float)
    ys    = np.asarray(values_rev[::-1], dtype=float)
    xs    = times - now
    return xs, ys


# =============================================================================
# Main GUI application
# =============================================================================

class ScopeApp(QtWidgets.QMainWindow):
    """
    Time-based oscilloscope for SXM channels.

    Responsibilities
    ----------------
    • Build the UI (controls, 4 plots + stats bars, comparison plot).
    • Periodically:
        – read selected channels via the injected IDataSource,
        – scale to physical units,
        – optionally high-pass “Topo” (Relative Z),
        – render plots,
        – compute & display per-plot stats in the StatsBar.

    Performance keys
    ----------------
    • Per-plot histories stored in deque(maxlen=…) for O(1) appends and automatic aging.
    • For plotting and stats we only transform the TAIL of each deque (window-sized),
      not the entire history.
    """
    def __init__(self, data_source: IDataSource, driver_handle=None):
        super().__init__()
        self.data_source = data_source
        self.driver_handle = driver_handle
        self.setWindowTitle("SXM Time-Based Multi-Channel Scope (deque-optimized)")

        # ---------- Timing configuration ----------
        self.sample_period_s       = 0.05   # 20 Hz GUI update & acquisition
        self.max_history_seconds   = 600    # keep at most 10 minutes
        self.time_window_options_s = [1, 5, 10, 30, 60, 120, 600]  # selectable window spans

        # Compute maximum history in samples; deques will auto-drop older entries.
        self.max_history_samples = int(self.max_history_seconds / self.sample_period_s) + 2

        # ---------- Plot/channel setup ----------
        self.num_plots = 4
        self.channel_names = list(channels.keys())

        # Per-plot raw histories: separate deques for time and value (avoids tuple churn)
        self.raw_time_deques   = [deque(maxlen=self.max_history_samples) for _ in range(self.num_plots)]
        self.raw_value_deques  = [deque(maxlen=self.max_history_samples) for _ in range(self.num_plots)]

        # Per-plot filtered (Relative Z) histories (Topo only when enabled)
        self.rel_time_deques   = [deque(maxlen=self.max_history_samples) for _ in range(self.num_plots)]
        self.rel_value_deques  = [deque(maxlen=self.max_history_samples) for _ in range(self.num_plots)]

        # Per-plot high-pass filters for Relative Z
        self.relative_z_tau_s = 1.0
        self.hp_filters = [IIRHighPass(self.relative_z_tau_s, self.sample_period_s) for _ in range(self.num_plots)]

        # Comparison plot histories (raw only): left and right traces
        self.cmp_time_deques  = [deque(maxlen=self.max_history_samples) for _ in range(2)]
        self.cmp_value_deques = [deque(maxlen=self.max_history_samples) for _ in range(2)]

        # Appearance
        self.line_colors       = ['#3498db', '#e67e22', '#16a085', '#8e44ad', '#c0392b', '#27ae60']  # 4 + 2
        self.background_color  = '#222222'
        self.line_width_px     = 2

        # Build UI
        self._build_controls()
        self._build_plots_with_stats()
        self._build_comparison_plot()
        self._build_menu()

        # Status bar displays current data source
        self._update_status_bar()

        # Timer for periodic updates
        self.update_timer = QtCore.QTimer(self)
        self.update_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.update_timer.timeout.connect(self._on_timer_tick)
        self.update_timer.start(int(self.sample_period_s * 1000))

    # ---------- UI builders ----------

    def _build_controls(self) -> None:
        """Create the top control row: time window, 4 channel selectors, Relative Z."""
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        self.main_vbox = QtWidgets.QVBoxLayout(central)
        self.main_vbox.setContentsMargins(6, 6, 6, 6)
        self.main_vbox.setSpacing(6)

        control_row = QtWidgets.QHBoxLayout()
        control_row.setSpacing(10)
        self.main_vbox.addLayout(control_row)

        # Time window selector
        self.time_window_combo = QtWidgets.QComboBox()
        for span in self.time_window_options_s:
            self.time_window_combo.addItem(f"{span} s")
        self.time_window_combo.setCurrentIndex(2)  # default = 10 s
        control_row.addWidget(QtWidgets.QLabel("Time Window:"))
        control_row.addWidget(self.time_window_combo)

        # 4 independent channel selectors (one per plot)
        self.channel_selectors: List[QtWidgets.QComboBox] = []
        for i in range(self.num_plots):
            combo = QtWidgets.QComboBox()
            combo.addItems(self.channel_names)
            control_row.addWidget(QtWidgets.QLabel(f"Ch{i+1}:"))
            control_row.addWidget(combo)
            self.channel_selectors.append(combo)

        # Relative Z checkbox (affects "Topo" only)
        self.relative_z_checkbox = QtWidgets.QCheckBox("Relative Z (Topo)")
        self.relative_z_checkbox.stateChanged.connect(self._reset_relative_z_state)
        control_row.addWidget(self.relative_z_checkbox)

    def _build_plots_with_stats(self) -> None:
        """Create a 2x2 grid of (PlotWidget + StatsBar)."""
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        self.main_vbox.addLayout(grid)

        self.plot_widgets: List[pg.PlotWidget] = []
        self.plot_curves:  List[pg.PlotDataItem] = []
        self.stats_bars:   List[StatsBar] = []

        for i in range(self.num_plots):
            container = QtWidgets.QWidget()
            vbox = QtWidgets.QVBoxLayout(container)
            vbox.setContentsMargins(0, 0, 0, 0)
            vbox.setSpacing(2)

            # Plot
            plot = pg.PlotWidget(title=f"Channel {i+1}")
            plot.showGrid(x=True, y=True)
            plot.setBackground(self.background_color)
            plot.setLabel('bottom', "Time", units="s")
            curve = plot.plot(pen=pg.mkPen(self.line_colors[i], width=self.line_width_px))
            # Reduce overdraw when zoomed out
            curve.setClipToView(True)
            try:
                curve.setDownsampling(auto=True, method='peak')
                curve.setSkipFiniteCheck(True)
            except Exception:
                pass  # older pyqtgraph versions

            vbox.addWidget(plot)

            # Stats bar below the plot
            stats_bar = StatsBar()
            vbox.addWidget(stats_bar)

            grid.addWidget(container, i // 2, i % 2)

            self.plot_widgets.append(plot)
            self.plot_curves.append(curve)
            self.stats_bars.append(stats_bar)

    def _build_comparison_plot(self) -> None:
        """Create the dual-axis comparison plot with two channel selectors."""
        self.main_vbox.addWidget(QtWidgets.QLabel("Comparison Plot:"))
        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(10)
        self.main_vbox.addLayout(top_row)

        self.compare_selector_left  = QtWidgets.QComboBox();  self.compare_selector_left.addItems(self.channel_names)
        self.compare_selector_right = QtWidgets.QComboBox();  self.compare_selector_right.addItems(self.channel_names)
        top_row.addWidget(self.compare_selector_left)
        top_row.addWidget(self.compare_selector_right)

        self.comparison_plot = pg.PlotWidget(title="Compare Channels")
        plot_item = self.comparison_plot.getPlotItem()
        plot_item.showGrid(x=True, y=True)
        plot_item.setLabel('bottom', "Time", units="s")
        self.comparison_plot.setBackground(self.background_color)

        # Dual Y axes via a second ViewBox for the right-hand curve
        plot_item.showAxis('right')
        self.right_viewbox = pg.ViewBox()
        plot_item.scene().addItem(self.right_viewbox)
        plot_item.getAxis('right').linkToView(self.right_viewbox)
        self.right_viewbox.setXLink(plot_item.vb)
        plot_item.vb.sigResized.connect(lambda:
            self.right_viewbox.setGeometry(plot_item.vb.sceneBoundingRect())
        )

        self.compare_curve_left  = pg.PlotDataItem(pen=pg.mkPen(self.line_colors[4], width=self.line_width_px))
        self.compare_curve_right = pg.PlotDataItem(pen=pg.mkPen(self.line_colors[5], width=self.line_width_px))
        plot_item.addItem(self.compare_curve_left)
        self.right_viewbox.addItem(self.compare_curve_right)

        self.main_vbox.addWidget(self.comparison_plot)

    def _build_menu(self) -> None:
        """Build the top menu (Appearance + Data Source selection)."""
        menubar = self.menuBar()

        # Appearance customization
        view_menu = menubar.addMenu("View")
        appearance_action = QtWidgets.QAction("Appearance…", self)
        appearance_action.triggered.connect(self._open_appearance_dialog)
        view_menu.addAction(appearance_action)

        # Data source switching
        source_menu = menubar.addMenu("Source")
        use_driver_action = QtWidgets.QAction("Use Driver", self)
        use_mock_action   = QtWidgets.QAction("Use Mock Data", self)
        use_driver_action.triggered.connect(self._switch_to_driver)
        use_mock_action.triggered.connect(self._switch_to_mock)
        source_menu.addAction(use_driver_action)
        source_menu.addAction(use_mock_action)

    # ---------- menu & utility handlers ----------

    def _update_status_bar(self) -> None:
        """Indicate which data source is active."""
        src = "Driver" if isinstance(self.data_source, DriverDataSource) else "Mock"
        self.statusBar().showMessage(f"Data Source: {src}")

    def _open_appearance_dialog(self) -> None:
        """Small dialog to adjust line colors, background, and line width."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Plot Appearance")
        form = QtWidgets.QFormLayout(dialog)

        color_buttons: List[QtWidgets.QPushButton] = []
        for i in range(6):
            btn = QtWidgets.QPushButton()
            btn.setStyleSheet(f"background-color: {self.line_colors[i]}")
            color_buttons.append(btn)
            form.addRow(f"Line {i+1} Color:", btn)

        bg_button = QtWidgets.QPushButton()
        bg_button.setStyleSheet(f"background-color: {self.background_color}")
        form.addRow("Background Color:", bg_button)

        width_spin = QtWidgets.QSpinBox(); width_spin.setRange(1, 10); width_spin.setValue(self.line_width_px)
        form.addRow("Line Thickness:", width_spin)

        def pick_color(idx: int):
            color = QtWidgets.QColorDialog.getColor()
            if color.isValid():
                css = color.name()
                color_buttons[idx].setStyleSheet(f"background-color: {css}")
                self.line_colors[idx] = css

        for i, btn in enumerate(color_buttons):
            btn.clicked.connect(lambda _, i=i: pick_color(i))

        def pick_background():
            color = QtWidgets.QColorDialog.getColor()
            if color.isValid():
                css = color.name()
                bg_button.setStyleSheet(f"background-color: {css}")
                self.background_color = css
        bg_button.clicked.connect(pick_background)

        def apply_and_close():
            self.line_width_px = width_spin.value()
            # Apply to plots
            for i, (plot, curve) in enumerate(zip(self.plot_widgets, self.plot_curves)):
                plot.setBackground(self.background_color)
                curve.setPen(pg.mkPen(self.line_colors[i], width=self.line_width_px))
            # Comparison
            self.comparison_plot.setBackground(self.background_color)
            self.compare_curve_left.setPen(pg.mkPen(self.line_colors[4], width=self.line_width_px))
            self.compare_curve_right.setPen(pg.mkPen(self.line_colors[5], width=self.line_width_px))
            dialog.accept()

        ok = QtWidgets.QPushButton("OK")
        ok.clicked.connect(apply_and_close)
        form.addRow(ok)
        dialog.exec_()

    def _switch_to_mock(self) -> None:
        """Switch to synthetic data (no hardware required)."""
        self.data_source = MockDataSource()
        self._update_status_bar()

    def _switch_to_driver(self) -> None:
        """Attempt to open and switch to the real driver."""
        if not WIN32_AVAILABLE:
            QtWidgets.QMessageBox.warning(self, "Driver", "win32 modules not available on this system.")
            return
        try:
            handle = win32file.CreateFile(
                r"\\.\SXM",
                win32con.GENERIC_READ | win32con.GENERIC_WRITE,
                0, None,
                win32con.OPEN_EXISTING,
                win32con.FILE_ATTRIBUTE_NORMAL,
                None
            )
            self.data_source = DriverDataSource(handle)
            self.driver_handle = handle
            self._update_status_bar()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Driver", f"Could not open driver: {exc}")

    def _reset_relative_z_state(self) -> None:
        """Clear filter states and ΔZ deques when toggling Relative Z."""
        for filt in self.hp_filters:
            filt.reset()
        for dq in self.rel_time_deques:
            dq.clear()
        for dq in self.rel_value_deques:
            dq.clear()

    # ---------- main periodic update ----------

    def _on_timer_tick(self) -> None:
        """
        Called every sample_period_s.
        Steps:
          1) Acquire new samples for selected channels.
          2) Append to per-plot deques (raw; and ΔZ if enabled).
          3) Extract only the tail needed for the visible window and plot it.
          4) Compute & display stats in the StatsBar (below each plot).
        """
        now = time.time()
        window_seconds = self.time_window_options_s[self.time_window_combo.currentIndex()]
        # Number of points needed to cover the visible window
        n_tail = int(window_seconds / self.sample_period_s) + 2

        # ----- 4 independent plots -----
        for plot_idx, combo in enumerate(self.channel_selectors):
            chan_name = combo.currentText()
            chan_index, _, chan_unit, chan_scale = channels[chan_name]

            # 1) Acquire and scale
            raw_int = self.data_source.read_value(chan_index)
            scaled_val = raw_int * chan_scale

            # 2) Append to raw deques (auto-trim by maxlen)
            self.raw_time_deques[plot_idx].append(now)
            self.raw_value_deques[plot_idx].append(scaled_val)

            # ΔZ (Relative Z) path for Topo only
            use_relative = (chan_name == 'Topo' and self.relative_z_checkbox.isChecked())
            if use_relative:
                hp = self.hp_filters[plot_idx]
                rel_val = hp.process(scaled_val)
                self.rel_time_deques[plot_idx].append(now)
                self.rel_value_deques[plot_idx].append(rel_val)

            # 3) Choose which series to visualize (raw vs ΔZ)
            if use_relative:
                t_deq = self.rel_time_deques[plot_idx]
                v_deq = self.rel_value_deques[plot_idx]
                y_label_text, y_label_unit = 'ΔZ', 'nm'
            else:
                t_deq = self.raw_time_deques[plot_idx]
                v_deq = self.raw_value_deques[plot_idx]
                y_label_text, y_label_unit = chan_name, chan_unit

            # Extract only the newest 'n_tail' points (O(n_tail), not O(n_history))
            xs, ys = tail_deque_to_arrays(t_deq, v_deq, n_tail, now)

            # 4) Plot and label
            self.plot_curves[plot_idx].setData(xs, ys)
            self.plot_widgets[plot_idx].setXRange(-window_seconds, 0, padding=0)
            self.plot_widgets[plot_idx].setLabel('left', y_label_text, y_label_unit)
            self.plot_widgets[plot_idx].enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

            # Stats for the visible window
            if ys.size:
                instantaneous = float(ys[-1])
                sigma = float(np.std(ys)) if ys.size > 1 else 0.0
                mean_val = float(np.mean(ys))
                snr_db = (20.0 * math.log10(abs(mean_val) / sigma)) if sigma > 0 else float('inf')
                self.stats_bars[plot_idx].update_stats(instantaneous, y_label_unit, sigma, snr_db)
            else:
                self.stats_bars[plot_idx].update_stats(None, y_label_unit, None, None)

        # ----- Dual-axis comparison plot (raw values only) -----
        left_name  = self.compare_selector_left.currentText()
        right_name = self.compare_selector_right.currentText()

        # Acquire and append one new point for each comparison series
        for idx_cmp, chan_name in enumerate((left_name, right_name)):
            c_idx, _, _, c_scale = channels[chan_name]
            raw_int = self.data_source.read_value(c_idx)
            val = raw_int * c_scale
            self.cmp_time_deques[idx_cmp].append(now)
            self.cmp_value_deques[idx_cmp].append(val)

        # Extract tails for comparison
        xs_left,  ys_left  = tail_deque_to_arrays(self.cmp_time_deques[0], self.cmp_value_deques[0], n_tail, now)
        xs_right, ys_right = tail_deque_to_arrays(self.cmp_time_deques[1], self.cmp_value_deques[1], n_tail, now)
        # Use left X for both (they are sampled at the same times in this loop)
        self.compare_curve_left.setData(xs_left, ys_left)
        self.compare_curve_right.setData(xs_left, ys_right)

        # Configure axes and labels
        item = self.comparison_plot.getPlotItem()
        item.setXRange(-window_seconds, 0, padding=0)
        item.setLabel('left',  left_name,  channels[left_name][2])
        item.getAxis('right').setLabel(right_name, channels[right_name][2])
        item.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        self.right_viewbox.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

    # ---------- Qt window events ----------

    def changeEvent(self, event: QtCore.QEvent) -> None:
        """
        Pause the timer if the window is minimized; resume when restored.
        Saves CPU and avoids unnecessary driver access when not in use.
        """
        if event.type() == QtCore.QEvent.WindowStateChange:
            minimized = bool(self.windowState() & QtCore.Qt.WindowMinimized)
            if minimized:
                self.update_timer.stop()
            else:
                if not self.update_timer.isActive():
                    self.update_timer.start(int(self.sample_period_s * 1000))
        super().changeEvent(event)


# =============================================================================
# Entry point: open driver if available, otherwise fall back to mock data
# =============================================================================

def create_data_source() -> Tuple[IDataSource, Optional[object], str]:
    """
    Try to open the SXM driver. If it fails (e.g., not on the lab PC), use MockDataSource.
    Returns (source, driver_handle_or_None, "Driver"|"Mock").
    """
    if WIN32_AVAILABLE:
        try:
            handle = win32file.CreateFile(
                r"\\.\SXM",
                win32con.GENERIC_READ | win32con.GENERIC_WRITE,
                0, None,
                win32con.OPEN_EXISTING,
                win32con.FILE_ATTRIBUTE_NORMAL,
                None
            )
            return DriverDataSource(handle), handle, "Driver"
        except Exception:
            pass
    return MockDataSource(), None, "Mock"


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    source, handle, src_name = create_data_source()
    window = ScopeApp(source, driver_handle=handle)
    window.resize(1200, 900)
    window.show()

    sys.exit(app.exec_())

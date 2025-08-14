"""
NC-AFM Control Suite (Production)

- Single-window GUI to read/write NC‑AFM parameters via the official SXM DDE API.
- Production-only: relies on SXMRemote.DDEClient; no mock or shadow functions.
- Parameters tab:
    * Columns: Parameter | Edit Code | Previous | Current | New Value
    * Previous and Current are grouped readouts (light gray).
    * New Value is editable (light yellow), with Auto-Send and Apply Selected.
    * Collapsible change log with timestamps.
- Step Test tab:
    * Symbolic square-wave preview for a chosen parameter (no live measurement yet).
    * Timed Start applies alternating low/high steps every period for N steps.
    * Run log shows each value sent with timestamps.

Author: (your lab)
"""

from __future__ import annotations

import sys
import datetime
from typing import List, Tuple

from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import SXMRemote  # Production module: provides DDEClient and methods


# -----------------------------------------------------------------------------
# Configuration: map friendly keys -> (EditXX, Friendly Label)
# -----------------------------------------------------------------------------
PARAM_DEFINITIONS: List[Tuple[str, str, str]] = [
    ("amp_ki",  "Edit24",  "Amplitude Ki"),
    ("amp_kp",  "Edit32",  "Amplitude Kp"),
    ("pll_kp",  "Edit27",  "PLL Kp"),
    ("pll_ki",  "Edit22",  "PLL Ki"),
    ("amp_ref", "Edit23",  "Amplitude Ref"),
    ("freq_ref","Edit15",  "Frequency Ref"),
]

PARAM_TOOLTIPS = {
    "amp_ki":  "Integral gain of the amplitude feedback loop.\n"
               "Integrates amplitude error to remove long-term offsets.",
    "amp_kp":  "Proportional gain of the amplitude feedback loop.\n"
               "Acts directly on the instantaneous amplitude error.",
    "pll_kp":  "Proportional gain of the PLL loop controlling oscillation frequency.\n"
               "Higher Kp speeds phase correction but can increase noise.",
    "pll_ki":  "Integral gain of the PLL loop.\n"
               "Eliminates steady frequency offsets by integrating phase error.",
    "amp_ref": "Target oscillation amplitude (setpoint).\n"
               "Amplitude loop adjusts drive to maintain this value.",
    "freq_ref":"Target oscillation frequency for the PLL to track.\n"
               "PLL locks measured frequency to this reference.",
}


# =============================================================================
# Tab 1: Parameters
# =============================================================================
class ParametersTab(QtWidgets.QWidget):
    """
    Parameters editor tab.

    - Reads 'Current' values from the instrument via SXMRemote.DDEClient.GetScanPara(EditXX).
    - Writes 'New Value' via SXMRemote.DDEClient.SendWait("ScanPara('EditXX', value);").
    - 'Previous' stores the last 'Current' before a successful write, for quick comparison.
    """

    REFRESH_MS = 1000  # periodic refresh of Current values

    def __init__(self, dde_client: SXMRemote.DDEClient, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.dde_client = dde_client

        outer = QtWidgets.QVBoxLayout(self)

        # --- Toolbar ---
        toolbar = QtWidgets.QHBoxLayout()
        outer.addLayout(toolbar)

        self.apply_selected_button = QtWidgets.QPushButton("Apply Selected")
        self.apply_selected_button.setToolTip("Send all edited 'New Value' cells in the selected rows.")
        self.apply_selected_button.clicked.connect(self.apply_selected_rows)
        toolbar.addWidget(self.apply_selected_button)

        self.auto_send_checkbox = QtWidgets.QCheckBox("Auto-Send on Edit")
        self.auto_send_checkbox.setToolTip("If enabled, pressing Enter in a 'New Value' cell immediately writes it.")
        toolbar.addWidget(self.auto_send_checkbox)

        self.toggle_log_button = QtWidgets.QPushButton("Show Change Log")
        self.toggle_log_button.setCheckable(True)
        self.toggle_log_button.toggled.connect(self._toggle_log_visibility)
        toolbar.addWidget(self.toggle_log_button)

        toolbar.addStretch()

        # --- Table ---
        self.table = QtWidgets.QTableWidget(len(PARAM_DEFINITIONS), 5)
        self.table.setHorizontalHeaderLabels(
            ["Parameter", "Edit Code", "Previous", "Current", "New Value"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        outer.addWidget(self.table)

        # Fill rows
        for row, (key, edit_code, label) in enumerate(PARAM_DEFINITIONS):
            # Parameter name with tooltip (read-only)
            name_item = QtWidgets.QTableWidgetItem(label)
            name_item.setToolTip(PARAM_TOOLTIPS[key])
            name_item.setFlags(name_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(row, 0, name_item)

            # Edit code (read-only)
            self._set_cell(row, 1, edit_code, editable=False, bg=None)

            # Previous (read-only, grouped gray)
            self._set_cell(row, 2, "", editable=False, bg=QtGui.QColor("#e0e0e0"))

            # Current (read-only, grouped gray)
            self._set_cell(row, 3, "", editable=False, bg=QtGui.QColor("#e0e0e0"))

            # New Value (editable, yellow)
            self._set_cell(row, 4, "", editable=True, bg=QtGui.QColor("#fff8dc"))

        # --- Collapsible log ---
        self.change_log = QtWidgets.QTextEdit()
        self.change_log.setReadOnly(True)
        self.change_log.hide()
        outer.addWidget(self.change_log)

        # Signals
        self.table.cellChanged.connect(self._on_cell_changed)

        # Periodic refresh of 'Current'
        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.timeout.connect(self.refresh_current_values)
        self._refresh_timer.start(self.REFRESH_MS)

        # Initial population of Previous/Current
        self.refresh_current_values(first_load=True)

        # Keyboard shortcut for batch apply
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Return"), self, self.apply_selected_rows)

    # ---------- UI helpers ----------

    def _set_cell(self, row: int, col: int, text: str, *, editable: bool, bg: QtGui.QColor | None) -> None:
        """Create or update a table cell."""
        item = QtWidgets.QTableWidgetItem(str(text))
        if not editable:
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
        if bg:
            item.setBackground(bg)
        self.table.setItem(row, col, item)

    # ---------- DDE I/O ----------

    def refresh_current_values(self, *, first_load: bool = False) -> None:
        """
        Refresh 'Current' from the device using GetScanPara(EditXX).
        If first_load=True, also copy Current -> Previous so the table starts consistent.
        """
        for row, (_key, edit_code, _label) in enumerate(PARAM_DEFINITIONS):
            try:
                value = self.dde_client.GetScanPara(edit_code)
            except Exception:
                value = None

            if value is None:
                # Show error but keep table stable
                self.table.item(row, 3).setText("Err")
                if first_load:
                    self.table.item(row, 2).setText("Err")
            else:
                self.table.item(row, 3).setText(str(value))
                if first_load:
                    self.table.item(row, 2).setText(str(value))

    def _write_parameter(self, edit_code: str, new_value: float) -> None:
        """
        Write a parameter using the official syntax used in setParasByComponentName.py:
            SendWait("ScanPara('EditXX', value);")
        """
        self.dde_client.SendWait(f"ScanPara('{edit_code}', {new_value});")

    # ---------- Actions ----------

    def _on_cell_changed(self, row: int, col: int) -> None:
        """
        Triggered when a cell changes. If the edited cell is 'New Value' and
        'Auto-Send on Edit' is enabled, send immediately.
        """
        NEW_VALUE_COL = 4
        if col != NEW_VALUE_COL:
            return
        if not self.auto_send_checkbox.isChecked():
            return

        try:
            target_value = float(self.table.item(row, NEW_VALUE_COL).text())
        except (TypeError, ValueError):
            return  # Ignore invalid numeric input

        self.apply_single_row(row, target_value)

    def apply_single_row(self, row: int, target_value: float) -> None:
        """
        Apply a single row change:
          - Set 'Previous' = last 'Current'
          - Write via DDE
          - Check readback and flash green/red
          - Log the change
        """
        EDIT_CODE_COL, PREV_COL, CURR_COL, NEW_COL = 1, 2, 3, 4

        edit_code = self.table.item(row, EDIT_CODE_COL).text()
        last_current_text = self.table.item(row, CURR_COL).text()
        self.table.item(row, PREV_COL).setText(last_current_text)

        # Write to device
        self._write_parameter(edit_code, target_value)

        # Short delay then verify
        QtCore.QTimer.singleShot(500, lambda: self._verify_and_flash(row, target_value))
        self._append_log(row, old_value=last_current_text, new_value=target_value)

    def apply_selected_rows(self) -> None:
        """Apply all edited 'New Value' cells in the selected rows."""
        NEW_VALUE_COL = 4
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        for row in rows:
            try:
                target_value = float(self.table.item(row, NEW_VALUE_COL).text())
            except (TypeError, ValueError):
                continue
            self.apply_single_row(row, target_value)

    # ---------- Feedback & Logging ----------

    def _verify_and_flash(self, row: int, expected_value: float) -> None:
        """
        Re-read 'Current' and flash the cell:
          - Green if equals the expected value
          - Red otherwise
        Then restore the normal gray background and refresh the 'Current' column.
        """
        CURR_COL = 3
        edit_code = self.table.item(row, 1).text()

        try:
            readback = self.dde_client.GetScanPara(edit_code)
        except Exception:
            readback = None

        if readback is not None and float(readback) == float(expected_value):
            color = QtGui.QColor("#a8e6a3")  # success
        else:
            color = QtGui.QColor("#f4a6a6")  # mismatch

        self.table.item(row, CURR_COL).setBackground(color)
        QtCore.QTimer.singleShot(
            900, lambda: self.table.item(row, CURR_COL).setBackground(QtGui.QColor("#e0e0e0"))
        )
        self.refresh_current_values(first_load=False)

    def _append_log(self, row: int, *, old_value: str, new_value: float) -> None:
        """Append a timestamped entry to the change log."""
        label = self.table.item(row, 0).text()
        edit_code = self.table.item(row, 1).text()
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.change_log.append(f"[{ts}] {label} ({edit_code}): {old_value} → {new_value}")

    def _toggle_log_visibility(self, show: bool) -> None:
        """Show or hide the change log panel."""
        self.change_log.setVisible(show)
        self.toggle_log_button.setText("Hide Change Log" if show else "Show Change Log")


# =============================================================================
# Tab 2: Step Test (symbolic preview + timed sending)
# =============================================================================
class StepTestTab(QtWidgets.QWidget):
    """
    Step Test tab.

    - PREVIEW: Draws a symbolic square wave of the planned steps (no live measurement).
    - START: Sends alternating low/high values to the chosen parameter every 'period'
      seconds, for 'steps' steps, using:
          SendWait("ScanPara('EditXX', value);")
    - A run log records each set action with a timestamp.
    """

    def __init__(self, dde_client: SXMRemote.DDEClient, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.dde_client = dde_client
        self._step_index = 0
        self._timer: QtCore.QTimer | None = None

        # ----- Layout skeleton -----
        outer = QtWidgets.QVBoxLayout(self)

        # Controls (fixed spacing; non-stretch)
        controls_widget = QtWidgets.QWidget()
        controls = QtWidgets.QGridLayout(controls_widget)
        controls.setContentsMargins(8, 6, 8, 6)
        controls.setHorizontalSpacing(12)
        controls.setVerticalSpacing(6)

        # Parameter selector
        self.parameter_combo = QtWidgets.QComboBox()
        for key, edit_code, label in PARAM_DEFINITIONS:
            self.parameter_combo.addItem(label, (key, edit_code))
        self.parameter_combo.setMinimumWidth(160)

        r, c = 0, 0
        controls.addWidget(self.parameter_combo, r, c, 1, 2); c += 2

        # Helpers for labels
        def right_label(text: str) -> QtWidgets.QLabel:
            lab = QtWidgets.QLabel(text)
            lab.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            return lab

        # Low / High
        controls.addWidget(right_label("Low:"), r, c); c += 1
        self.low_value = QtWidgets.QDoubleSpinBox()
        self.low_value.setDecimals(4); self.low_value.setMaximum(1e12); self.low_value.setValue(100.0)
        self.low_value.setFixedWidth(110)
        controls.addWidget(self.low_value, r, c); c += 1

        controls.addWidget(right_label("High:"), r, c); c += 1
        self.high_value = QtWidgets.QDoubleSpinBox()
        self.high_value.setDecimals(4); self.high_value.setMaximum(1e12); self.high_value.setValue(120.0)
        self.high_value.setFixedWidth(110)
        controls.addWidget(self.high_value, r, c); c += 1

        # Period / Steps
        controls.addWidget(right_label("Period (s):"), r, c); c += 1
        self.step_period = QtWidgets.QDoubleSpinBox()
        self.step_period.setDecimals(2); self.step_period.setMaximum(3600); self.step_period.setValue(5.0)
        self.step_period.setFixedWidth(80)
        controls.addWidget(self.step_period, r, c); c += 1

        controls.addWidget(right_label("Steps:"), r, c); c += 1
        self.num_steps = QtWidgets.QSpinBox()
        self.num_steps.setMaximum(10000); self.num_steps.setValue(20)
        self.num_steps.setFixedWidth(70)
        controls.addWidget(self.num_steps, r, c); c += 1

        # Buttons
        self.preview_button = QtWidgets.QPushButton("Preview")
        self.preview_button.setFixedWidth(90)
        self.preview_button.clicked.connect(self.preview_pattern)
        controls.addWidget(self.preview_button, r, c); c += 1

        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.setFixedWidth(90)
        self.start_button.clicked.connect(self.start_test)
        controls.addWidget(self.start_button, r, c)

        outer.addWidget(controls_widget)

        # Plot (symbolic)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        axis_pen = pg.mkPen(color='k', width=1)
        for name in ('bottom', 'left'):
            axis = self.plot_widget.getAxis(name)
            axis.setPen(axis_pen)
            axis.setTextPen('k')
            axis.setStyle(tickTextOffset=5, **{'tickFont': QtGui.QFont('', 10)})
        self.plot_widget.setLabel('left', 'Value')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        outer.addWidget(self.plot_widget)

        # Run log
        self.run_log = QtWidgets.QTextEdit()
        self.run_log.setReadOnly(True)
        outer.addWidget(self.run_log)

    # ----- Plot helpers -----

    def _frame_axes(self, low: float, high: float, period: float, steps: int) -> None:
        """Set reasonable view ranges for the preview plot."""
        self.plot_widget.setXRange(0, max(1.0, steps * period), padding=0.02)
        ymin, ymax = sorted([low, high])
        span = max(1.0, abs(ymax - ymin))
        margin = 0.05 * span
        self.plot_widget.setYRange(ymin - margin, ymax + margin, padding=0.02)

    # ----- Preview / Start logic -----

    def preview_pattern(self) -> None:
        """
        Draw a symbolic square wave of the planned steps.
        pyqtgraph stepMode=True requires len(x) == len(y) + 1 (x are edges, y are heights).
        """
        low = self.low_value.value()
        high = self.high_value.value()
        period = self.step_period.value()
        steps = self.num_steps.value()

        # Build edges (x) and heights (y): start at 0, add one edge per step.
        x = [0.0]
        y = []
        for i in range(steps):
            x.append((i + 1) * period)
            y.append(low if (i % 2 == 0) else high)

        self.plot_widget.clear()
        self.plot_widget.plot(x, y, stepMode=True, pen=pg.mkPen('b', width=2))
        self._frame_axes(low, high, period, steps)

    def start_test(self) -> None:
        """
        Start sending the planned low/high sequence at a fixed interval.
        Does not wait for readback; keeps the preview plot visible.
        """
        self.preview_pattern()
        self._step_index = 0
        self.start_button.setEnabled(False)

        if self._timer is None:
            self._timer = QtCore.QTimer(self)
            self._timer.timeout.connect(self._perform_single_step)

        self._timer.start(int(self.step_period.value() * 1000))

    def _perform_single_step(self) -> None:
        """Send one step value (low on even indices, high on odd) and log it."""
        _, edit_code = self.parameter_combo.currentData()
        low = self.low_value.value()
        high = self.high_value.value()
        value = low if (self._step_index % 2 == 0) else high

        # Production write using SXMRemote's syntax
        self.dde_client.SendWait(f"ScanPara('{edit_code}', {value});")

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        label = self.parameter_combo.currentText()
        self.run_log.append(f"[{timestamp}] Set {label} ({edit_code}) to {value}")

        self._step_index += 1
        if self._step_index >= self.num_steps.value():
            self._timer.stop()
            self.start_button.setEnabled(True)


# =============================================================================
# Main Window
# =============================================================================
class MainWindow(QtWidgets.QTabWidget):
    """
    Main application window containing:
      - ParametersTab (table editor)
      - StepTestTab   (symbolic step preview + timed apply)
    """

    def __init__(self, dde_client: SXMRemote.DDEClient, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("NC-AFM Control Suite")
        self.resize(900, 540)

        self.parameters_tab = ParametersTab(dde_client)
        self.step_test_tab = StepTestTab(dde_client)

        self.addTab(self.parameters_tab, "Parameters")
        self.addTab(self.step_test_tab, "Step Test")


# =============================================================================
# Entrypoint (production)
# =============================================================================
def main() -> int:
    """
    Create the Qt application, connect to the SXM DDE server using SXMRemote,
    and start the GUI.

    Requirements:
      - The SXM software providing the DDE server ("SXM","Remote") must be running.
      - SXMRemote.py must be importable (same folder or on PYTHONPATH).
    """
    app = QtWidgets.QApplication(sys.argv)

    # Production connection: use the official DDEClient as in your examples
    dde_client = SXMRemote.DDEClient("SXM", "Remote")

    # (Optional) Match your examples — set combined mode once. Safe if already set.
    try:
        dde_client.SendWait("FeedPara('Mode', 8);")  # STM + AFM PLL
    except Exception:
        pass

    window = MainWindow(dde_client)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

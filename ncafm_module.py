import sys
import datetime
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

# ---------------- Parameter Map (EditXX only) ----------------
# key, EditXX, label
PARAM_MAP = [
    ("amp_ki",  "Edit24",  "Amplitude Ki"),
    ("amp_kp",  "Edit32",  "Amplitude Kp"),
    ("pll_kp",  "Edit27",  "PLL Kp"),
    ("pll_ki",  "Edit22",  "PLL Ki"),
    ("amp_ref", "Edit23",  "Amplitude Ref"),
    ("freq_ref","Edit15",  "Frequency Ref"),
]

# ---------------- Tooltips for Parameter column ----------------
PARAM_DESCRIPTIONS = {
    "amp_ki":  "The integral gain of the amplitude feedback loop in NC-AFM mode.\n"
               "It corrects long-term deviations from the target oscillation amplitude by integrating the error over time.",
    "amp_kp":  "The proportional gain of the amplitude feedback loop.\n"
               "It adjusts the amplitude based directly on the current error between the measured and target amplitude.",
    "pll_kp":  "The proportional gain of the Phase-Locked Loop controlling the cantilever’s oscillation frequency.\n"
               "Higher values increase the speed of phase correction but can make the loop noisier.",
    "pll_ki":  "The integral gain of the PLL loop.\n"
               "It removes long-term frequency offsets by integrating the phase error, improving tracking stability.",
    "amp_ref": "The target amplitude setpoint for the oscillating sensor (e.g., qPlus sensor).\n"
               "The amplitude feedback loop will adjust drive power to maintain this value.",
    "freq_ref":"The target oscillation frequency for the PLL to track.\n"
               "The PLL will adjust its control to keep the measured frequency locked to this reference."
}

# ---------------- Write-only cache for EditXX ----------------
# We will not read from SXM (no Get for EditXX). Cache last commanded values here.
LAST_WRITTEN = {edit: None for _, edit, _ in PARAM_MAP}

# ---------------- Mock DDE Client (offline testing) ----------------
class MockDDEClient:
    def __init__(self):
        self.params = {edit: None for _, edit, _ in PARAM_MAP}
    def SendWait(self, cmd: str):
        # Expect commands like: ScanPara('Edit24', 12.3);
        try:
            part = cmd.split("ScanPara(")[1].split(")")[0]
            edit, val = part.split(",")
            edit = edit.strip().strip("'\"")
            val = float(val)
            self.params[edit] = val
            print(f"[MOCK] Set {edit} = {val}")
        except Exception as e:
            print(f"[MOCK] Failed to parse SendWait: {e}")

# ---------------- Helpers (write-only, via SXMRemote.DDEClient.SendWait) ----------------
def write_edit(dde, edit_code: str, value: float):
    """
    Write a component edit field using the official 'component name' path:
      ScanPara('EditXX', value);
    and update cache for display.
    """
    dde.SendWait(f"ScanPara('{edit_code}', {value});")
    LAST_WRITTEN[edit_code] = float(value)

def read_edit_cached(edit_code: str):
    """
    There is no readback for EditXX on this build. Return cached last commanded value (or None).
    """
    return LAST_WRITTEN.get(edit_code, None)

# ---------------- Tab 1: Parameter Table ----------------
class ParamTable(QtWidgets.QWidget):
    def __init__(self, dde_client):
        super().__init__()
        self.dde = dde_client
        layout = QtWidgets.QVBoxLayout(self)

        # Toolbar
        toolbar = QtWidgets.QHBoxLayout()
        layout.addLayout(toolbar)

        self.apply_btn = QtWidgets.QPushButton("Apply Selected")
        self.apply_btn.clicked.connect(self.apply_selected)
        toolbar.addWidget(self.apply_btn)

        self.auto_send_chk = QtWidgets.QCheckBox("Auto-Send on Edit")
        toolbar.addWidget(self.auto_send_chk)

        self.toggle_log_btn = QtWidgets.QPushButton("Show Change Log")
        self.toggle_log_btn.setCheckable(True)
        self.toggle_log_btn.toggled.connect(self.toggle_log)
        toolbar.addWidget(self.toggle_log_btn)
        toolbar.addStretch()

        # Table
        self.table = QtWidgets.QTableWidget(len(PARAM_MAP), 5)
        self.table.setHorizontalHeaderLabels(
            ["Parameter", "Edit Code", "Previous", "Current", "New Value"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        layout.addWidget(self.table)

        # Fill rows
        for row, (key, edit, label) in enumerate(PARAM_MAP):
            # Parameter + tooltip
            p_item = QtWidgets.QTableWidgetItem(label)
            p_item.setToolTip(PARAM_DESCRIPTIONS.get(key, ""))
            p_item.setFlags(p_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(row, 0, p_item)

            # Edit code
            self._set_item(row, 1, edit, editable=False, bg=None)

            # Previous (RO, grouped gray)
            self._set_item(row, 2, "—", editable=False, bg=QtGui.QColor("#e0e0e0"))

            # Current (RO, grouped gray) – will show cached value or "—"
            cur = read_edit_cached(edit)
            self._set_item(row, 3, "—" if cur is None else str(cur), editable=False, bg=QtGui.QColor("#e0e0e0"))

            # New value (editable, yellow)
            self._set_item(row, 4, "", editable=True, bg=QtGui.QColor("#fff8dc"))

        # Log panel
        self.log_widget = QtWidgets.QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.hide()
        layout.addWidget(self.log_widget)

        # Signals
        self.table.cellChanged.connect(self.on_cell_changed)

        # Shortcut for Apply Selected
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Return"), self, self.apply_selected)

    def _set_item(self, row, col, text, editable, bg):
        item = QtWidgets.QTableWidgetItem(str(text))
        if not editable:
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
        if bg:
            item.setBackground(bg)
        self.table.setItem(row, col, item)

    def on_cell_changed(self, row, col):
        if col != 4:
            return
        if self.auto_send_chk.isChecked():
            try:
                new_val = float(self.table.item(row, col).text())
            except ValueError:
                return
            self.apply_row(row, new_val)

    def apply_row(self, row, new_val):
        edit_code = self.table.item(row, 1).text()

        # Move current into previous
        prev_val_text = self.table.item(row, 3).text()
        self.table.item(row, 2).setText(prev_val_text)

        # Send & cache
        try:
            write_edit(self.dde, edit_code, new_val)  # ScanPara('EditXX', value);
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "DDE Send Error",
                                          f"Failed to send {edit_code} = {new_val}\n\n{e}")
            return

        # Update Current immediately from cache (write-only system)
        self.table.item(row, 3).setText(str(new_val))

        # Visual feedback: “unverified” (no readback) flash then return to grouped gray
        self.table.item(row, 3).setBackground(QtGui.QColor("#fff2b3"))
        QtCore.QTimer.singleShot(900, lambda: self.table.item(row, 3).setBackground(QtGui.QColor("#e0e0e0")))

        self.log_change(row, prev_val_text, new_val)

    def apply_selected(self):
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()})
        for row in rows:
            try:
                new_val = float(self.table.item(row, 4).text())
            except ValueError:
                continue
            self.apply_row(row, new_val)

    def log_change(self, row, prev_val, new_val):
        param_label = self.table.item(row, 0).text()
        edit_code = self.table.item(row, 1).text()
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_widget.append(f"[{timestamp}] {param_label} ({edit_code}): {prev_val} → {new_val}")

    def toggle_log(self, show):
        self.log_widget.setVisible(show)
        self.toggle_log_btn.setText("Hide Change Log" if show else "Show Change Log")

# ---------------- Tab 2: Step Test (symbolic preview + send, no readback) ----------------
class StepTestTab(QtWidgets.QWidget):
    def __init__(self, dde_client):
        super().__init__()
        self.dde = dde_client
        self.timer = None
        self.step_index = 0

        layout = QtWidgets.QVBoxLayout(self)

        # Controls (fixed spacing)
        ctrl_widget = QtWidgets.QWidget()
        ctrl = QtWidgets.QGridLayout(ctrl_widget)
        ctrl.setContentsMargins(8, 6, 8, 6)
        ctrl.setHorizontalSpacing(12)
        ctrl.setVerticalSpacing(6)

        self.param_combo = QtWidgets.QComboBox()
        for key, edit, label in PARAM_MAP:
            self.param_combo.addItem(label, (key, edit))
        self.param_combo.setMinimumWidth(160)
        row = 0; col = 0
        ctrl.addWidget(self.param_combo, row, col, 1, 2); col += 2

        low_lbl = QtWidgets.QLabel("Low:"); low_lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        ctrl.addWidget(low_lbl, row, col); col += 1
        self.low_val = QtWidgets.QDoubleSpinBox(); self.low_val.setDecimals(4); self.low_val.setMaximum(1e12); self.low_val.setValue(100.0); self.low_val.setFixedWidth(110)
        ctrl.addWidget(self.low_val, row, col); col += 1

        high_lbl = QtWidgets.QLabel("High:"); high_lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        ctrl.addWidget(high_lbl, row, col); col += 1
        self.high_val = QtWidgets.QDoubleSpinBox(); self.high_val.setDecimals(4); self.high_val.setMaximum(1e12); self.high_val.setValue(120.0); self.high_val.setFixedWidth(110)
        ctrl.addWidget(self.high_val, row, col); col += 1

        per_lbl = QtWidgets.QLabel("Period (s):"); per_lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        ctrl.addWidget(per_lbl, row, col); col += 1
        self.period = QtWidgets.QDoubleSpinBox(); self.period.setDecimals(2); self.period.setMaximum(3600); self.period.setValue(5.0); self.period.setFixedWidth(80)
        ctrl.addWidget(self.period, row, col); col += 1

        steps_lbl = QtWidgets.QLabel("Steps:"); steps_lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        ctrl.addWidget(steps_lbl, row, col); col += 1
        self.steps = QtWidgets.QSpinBox(); self.steps.setMaximum(10000); self.steps.setValue(20); self.steps.setFixedWidth(70)
        ctrl.addWidget(self.steps, row, col); col += 1

        self.preview_btn = QtWidgets.QPushButton("Preview"); self.preview_btn.setFixedWidth(90)
        self.preview_btn.clicked.connect(self.preview_pattern); ctrl.addWidget(self.preview_btn, row, col); col += 1
        self.start_btn = QtWidgets.QPushButton("Start"); self.start_btn.setFixedWidth(90)
        self.start_btn.clicked.connect(self.start_test); ctrl.addWidget(self.start_btn, row, col)

        layout.addWidget(ctrl_widget)

        # Plot (symbolic)
        self.plot_widget = pg.PlotWidget()
        # Theme: white bg, black axes and labels
        self.plot_widget.setBackground('w')
        axis_pen = pg.mkPen(color='k', width=1)
        for ax in ['bottom', 'left']:
            self.plot_widget.getAxis(ax).setPen(axis_pen)
            self.plot_widget.getAxis(ax).setTextPen('k')
            self.plot_widget.getAxis(ax).setStyle(tickTextOffset=5, **{'tickFont': QtGui.QFont('', 10)})

        # Stabilize margins/layout (use PlotItem)
        plot_item = self.plot_widget.getPlotItem()
        plot_item.layout.setContentsMargins(50, 10, 10, 40)

        self.plot_widget.setLabel('left', 'Value')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        layout.addWidget(self.plot_widget)

        # Log
        self.log_output = QtWidgets.QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)

    def _set_axis_ranges(self, low, high, period, steps):
        self.plot_widget.setXRange(0, steps * period, padding=0.02)
        ymin, ymax = sorted([low, high])
        margin = 0.05 * max(1.0, abs(ymax - ymin))
        self.plot_widget.setYRange(ymin - margin, ymax + margin, padding=0.02)

    def preview_pattern(self):
        low = self.low_val.value()
        high = self.high_val.value()
        period = self.period.value()
        steps = self.steps.value()

        # Build step edges (x) and heights (y); len(x) = len(y) + 1 for stepMode=True
        x = [0.0]
        y = []
        for i in range(steps):
            x.append((i + 1) * period)
            y.append(low if (i % 2 == 0) else high)

        self.plot_widget.clear()
        self.plot_widget.plot(x, y, stepMode=True, pen=pg.mkPen('b', width=2))
        self._set_axis_ranges(low, high, period, steps)

    def start_test(self):
        self.preview_pattern()
        self.start_btn.setEnabled(False)
        self.step_index = 0
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.perform_step)
        self.timer.start(int(self.period.value() * 1000))

    def perform_step(self):
        _, edit_code = self.param_combo.currentData()
        low = self.low_val.value()
        high = self.high_val.value()
        value = low if self.step_index % 2 == 0 else high

        # Send command and update cache (same path used by table)
        try:
            write_edit(self.dde, edit_code, value)  # ScanPara('EditXX', value);
        except Exception as e:
            self.log_output.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] SEND ERROR: {e}")
            # Stop on error
            self.timer.stop()
            self.start_btn.setEnabled(True)
            return

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        param_label = self.param_combo.currentText()
        self.log_output.append(f"[{timestamp}] Set {param_label} ({edit_code}) to {value}")

        self.step_index += 1
        if self.step_index >= self.steps.value():
            self.timer.stop()
            self.start_btn.setEnabled(True)

# ---------------- Main Window ----------------
class MainWindow(QtWidgets.QTabWidget):
    def __init__(self, dde_client):
        super().__init__()
        self.setWindowTitle("NC-AFM Control Suite (EditXX write-only)")
        self.resize(900, 540)
        self.table_tab = ParamTable(dde_client)
        self.step_tab = StepTestTab(dde_client)
        self.addTab(self.table_tab, "Parameters")
        self.addTab(self.step_tab, "Step Test")

# ---------------- Main ----------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    # Real DDE, else mock
    try:
        import importlib
        SXMRemote = importlib.import_module("SXMRemote")
        dde = SXMRemote.DDEClient("SXM", "Remote")
        print("[INFO] Connected to real DDE server.")
    except Exception as e:
        print(f"[INFO] Using mock DDE client: {e}")
        dde = MockDDEClient()

    win = MainWindow(dde)
    win.show()
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())

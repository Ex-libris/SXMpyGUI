import sys
import datetime
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

# ---------------- Parameter Map ----------------
PARAM_MAP = [
    ("amp_ki",  "Edit24",  "Amplitude Ki"),
    ("amp_kp",  "Edit32",  "Amplitude Kp"),
    ("pll_kp",  "Edit27",  "PLL Kp"),
    ("pll_ki",  "Edit22",  "PLL Ki"),
    ("amp_ref", "Edit23",  "Amplitude Ref"),
    ("freq_ref","Edit15",  "Frequency Ref"),
]

# ---------------- Parameter Descriptions ----------------
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

# ---------------- Mock DDE Client ----------------
class MockDDEClient:
    """Simple in-memory fake DDEClient for offline testing."""
    def __init__(self):
        self.params = {
            "Edit24": 5.0, "Edit32": 10.0, "Edit27": 2.0,
            "Edit22": 1.0, "Edit23": 100.0, "Edit15": 32768.0
        }

    def SendWait(self, cmd: str):
        try:
            part = cmd.split("ScanPara(")[1].split(")")[0]
            edit, val = part.split(",")
            edit = edit.strip().strip("'\"")
            val = float(val)
            self.params[edit] = val
            print(f"[MOCK] Set {edit} = {val}")
        except Exception as e:
            print(f"[MOCK] Failed to parse SendWait: {e}")

    def GetPara(self, topic: str):
        try:
            edit = topic.split("GetScanPara(")[1].split(")")[0].strip().strip("'\"")
            return self.params.get(edit, 0.0)
        except Exception as e:
            print(f"[MOCK] Failed to parse GetPara: {e}")
            return None

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

        # Fill table
        for row, (key, edit, label) in enumerate(PARAM_MAP):
            # Parameter w/ tooltip
            item = QtWidgets.QTableWidgetItem(label)
            item.setToolTip(PARAM_DESCRIPTIONS[key])
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(row, 0, item)
            # Edit code
            self._set_item(row, 1, edit, editable=False, bg=None)
            # Previous
            self._set_item(row, 2, "", editable=False, bg=QtGui.QColor("#e0e0e0"))
            # Current
            self._set_item(row, 3, "", editable=False, bg=QtGui.QColor("#e0e0e0"))
            # New value
            self._set_item(row, 4, "", editable=True, bg=QtGui.QColor("#fff8dc"))

        # Log
        self.log_widget = QtWidgets.QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.hide()
        layout.addWidget(self.log_widget)

        # Signals
        self.table.cellChanged.connect(self.on_cell_changed)

        # Timer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh_currents)
        self.timer.start(1000)
        self.refresh_currents(first_load=True)

        # Shortcut
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Return"), self, self.apply_selected)

    def _set_item(self, row, col, text, editable, bg):
        item = QtWidgets.QTableWidgetItem(str(text))
        if not editable:
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
        if bg:
            item.setBackground(bg)
        self.table.setItem(row, col, item)

    def _get_param_value(self, edit_code):
        topic = f"a:=GetScanPara('{edit_code}');\r\n  writeln(a);"
        return self.dde.GetPara(topic)

    def _send_param_value(self, edit_code, value):
        self.dde.SendWait(f"ScanPara('{edit_code}', {value});")

    def refresh_currents(self, first_load=False):
        for row, (_, edit, _) in enumerate(PARAM_MAP):
            val = self._get_param_value(edit)
            if val is not None:
                self.table.item(row, 3).setText(str(val))
                if first_load:
                    self.table.item(row, 2).setText(str(val))

    def on_cell_changed(self, row, col):
        if col != 4:  # New Value col
            return
        if self.auto_send_chk.isChecked():
            try:
                new_val = float(self.table.item(row, col).text())
            except ValueError:
                return
            self.apply_row(row, new_val)

    def apply_row(self, row, new_val):
        edit_code = self.table.item(row, 1).text()
        prev_val = self.table.item(row, 3).text()
        self.table.item(row, 2).setText(prev_val)
        self._send_param_value(edit_code, new_val)
        QtCore.QTimer.singleShot(500, lambda: self.check_update(row, new_val))
        self.log_change(row, prev_val, new_val)

    def check_update(self, row, expected_val):
        current_val = self.table.item(row, 3).text()
        try:
            current_val_f = float(current_val)
        except ValueError:
            current_val_f = None
        color = QtGui.QColor("#a8e6a3") if current_val_f == expected_val else QtGui.QColor("#f4a6a6")
        self.table.item(row, 3).setBackground(color)
        QtCore.QTimer.singleShot(1000, lambda: self.table.item(row, 3).setBackground(QtGui.QColor("#e0e0e0")))
        self.refresh_currents()

    def apply_selected(self):
        for row in set(idx.row() for idx in self.table.selectedIndexes()):
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


# ---------------- Tab 2: Step Test ----------------
class StepTestTab(QtWidgets.QWidget):
    def __init__(self, dde_client):
        super().__init__()
        self.dde = dde_client
        self.timer = None
        self.step_index = 0

        layout = QtWidgets.QVBoxLayout(self)

        # Controls
        ctrl_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(ctrl_layout)

        self.param_combo = QtWidgets.QComboBox()
        for key, edit, label in PARAM_MAP:
            self.param_combo.addItem(label, (key, edit))
        ctrl_layout.addWidget(self.param_combo)

        self.low_val = QtWidgets.QDoubleSpinBox(); self.low_val.setDecimals(4); self.low_val.setMaximum(1e9)
        self.low_val.setValue(100.0)
        ctrl_layout.addWidget(QtWidgets.QLabel("Low:")); ctrl_layout.addWidget(self.low_val)

        self.high_val = QtWidgets.QDoubleSpinBox(); self.high_val.setDecimals(4); self.high_val.setMaximum(1e9)
        self.high_val.setValue(120.0)
        ctrl_layout.addWidget(QtWidgets.QLabel("High:")); ctrl_layout.addWidget(self.high_val)

        self.period = QtWidgets.QDoubleSpinBox(); self.period.setDecimals(2); self.period.setMaximum(3600)
        self.period.setValue(5.0)
        ctrl_layout.addWidget(QtWidgets.QLabel("Period (s):")); ctrl_layout.addWidget(self.period)

        self.steps = QtWidgets.QSpinBox(); self.steps.setMaximum(1000)
        self.steps.setValue(10)
        ctrl_layout.addWidget(QtWidgets.QLabel("Steps:")); ctrl_layout.addWidget(self.steps)

        self.preview_btn = QtWidgets.QPushButton("Preview")
        self.preview_btn.clicked.connect(self.preview_pattern)
        ctrl_layout.addWidget(self.preview_btn)

        self.start_btn = QtWidgets.QPushButton("Start")
        self.start_btn.clicked.connect(self.start_test)
        ctrl_layout.addWidget(self.start_btn)

        # Plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Value')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.setBackground(QtGui.QColor("white"))
        layout.addWidget(self.plot_widget)
        # Customize plot appearance
        axis_pen = pg.mkPen(color='k', width=1)
        self.plot_widget.getAxis('bottom').setPen(axis_pen)
        self.plot_widget.getAxis('left').setPen(axis_pen)
        self.plot_widget.getAxis('bottom').setTextPen('k')
        self.plot_widget.getAxis('left').setTextPen('k')
        # Log
        self.log_output = QtWidgets.QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)

    def preview_pattern(self):
        low = self.low_val.value()
        high = self.high_val.value()
        period = self.period.value()
        steps = self.steps.value()

        # Build step edges (x) and heights (y) for stepMode=True
        # Requirement: len(x) == len(y) + 1
        x = [0.0]
        y = []
        for i in range(steps):
            x.append((i + 1) * period)
            y.append(low if (i % 2 == 0) else high)

        # Keep the final level for visual closure (optional but pretty)
        # pyqtgraph ignores the extra y, so do NOT append to y here.

        self.plot_widget.clear()
        self.plot_widget.plot(x, y, stepMode=True, pen=pg.mkPen('b', width=2))


    def start_test(self):
        self.preview_pattern()
        self.start_btn.setEnabled(False)
        self.step_index = 0
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.perform_step)
        self.timer.start(int(self.period.value() * 1000))

    def perform_step(self):
        _, edit_code = self.param_combo.currentData()
        low = self.low_val.value()
        high = self.high_val.value()

        value = low if self.step_index % 2 == 0 else high
        self.dde.SendWait(f"ScanPara('{edit_code}', {value});")

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        param_label = self.param_combo.currentText()
        self.log_output.append(f"[{timestamp}] Set {param_label} ({edit_code}) to {value}")

        self.step_index += 1
        if self.step_index >= self.steps.value():
            self.timer.stop()
            self.start_btn.setEnabled(True)

# ---------------- Main ----------------
class MainWindow(QtWidgets.QTabWidget):
    def __init__(self, dde_client):
        super().__init__()
        self.setWindowTitle("NC-AFM Control Suite")
        self.resize(800, 500)

        self.table_tab = ParamTable(dde_client)
        self.step_tab = StepTestTab(dde_client)

        self.addTab(self.table_tab, "Parameters")
        self.addTab(self.step_tab, "Step Test")

def main():
    app = QtWidgets.QApplication(sys.argv)
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

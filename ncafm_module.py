import sys
import datetime
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

# ---------- Slim, context-aware safety warning (used as a footer now) ----------
SHORT_WARNING_TEXT = (
    "⚠ Check SXM units & never exceed ±10 V without attenuation."
)

LONG_WARNING_HTML = (
    "<b>Units matter:</b> SXM interprets numbers in the <b>current</b> GUI units "
    "(Hz/kHz, V/mV/µV, nm/pm, etc.). Verify the unit shown in SXM before sending.<br>"
    "<b>Voltage safety:</b> Do <b>not</b> exceed ±10 V at the output unless a hardware divider/attenuator is in place.<br>"
    "Step Test sends exactly the values you enter — verify LOW/HIGH against the current unit."
)

def make_warning_strip(parent=None):
    """Compact, expandable warning strip (we'll place it as a global footer)."""
    container = QtWidgets.QWidget(parent)
    h = QtWidgets.QHBoxLayout(container)
    h.setContentsMargins(8, 6, 8, 6)
    h.setSpacing(8)

    icon = QtWidgets.QLabel("⚠", container)
    icon.setStyleSheet("font-size: 13pt;")
    h.addWidget(icon, 0, QtCore.Qt.AlignVCenter)

    msg = QtWidgets.QLabel(SHORT_WARNING_TEXT, container)
    msg.setStyleSheet("color:#6b5900; font-size: 10.5pt;")
    h.addWidget(msg, 1, QtCore.Qt.AlignVCenter)

    details_btn = QtWidgets.QToolButton(container)
    details_btn.setText("Details")
    details_btn.setCheckable(True)
    details_btn.setStyleSheet(
        "QToolButton { padding:2px 6px; border:1px solid #e6d9a2; border-radius:4px; background:#fff7da; }"
        "QToolButton:checked { background:#ffeeb7; }"
    )
    h.addWidget(details_btn, 0, QtCore.Qt.AlignVCenter)

    # hidden details popover
    details = QtWidgets.QTextBrowser(container)
    details.setHtml(LONG_WARNING_HTML)
    details.setOpenExternalLinks(True)
    details.setStyleSheet(
        "QTextBrowser { background:#fffaf0; border:1px solid #e6d9a2; border-radius:6px; padding:6px; color:#6b5900; }"
    )
    details.hide()

    # stack the strip and the details vertically
    wrapper = QtWidgets.QWidget(parent)
    v = QtWidgets.QVBoxLayout(wrapper)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(6)
    v.addWidget(container)
    v.addWidget(details)

    def toggle_details(on):
        details.setVisible(on)
    details_btn.toggled.connect(toggle_details)

    # border + background for the strip only (not the expanded details)
    container.setStyleSheet(
        "QWidget { background:#fff7da; border:1px solid #e6d9a2; border-radius:6px; }"
    )
    wrapper.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
    return wrapper

# ---------------- Canonical parameter registry ----------------
# ptype: "EDIT" -> ScanPara('EditXX', v) | "DNC" -> DNCPara(n, v)
PARAMS_CORE = {
    "amp_ki":    ("EDIT", "Edit24",  "Amplitude Ki"),
    "amp_kp":    ("EDIT", "Edit32",  "Amplitude Kp"),
    "pll_kp":    ("EDIT", "Edit27",  "PLL Kp"),
    "pll_ki":    ("EDIT", "Edit22",  "PLL Ki"),
    "amp_ref":   ("EDIT", "Edit23",  "Amplitude Ref"),
    # Removed: "freq_ref": ("EDIT", "Edit15", "Frequency Ref"),
    "used_freq": ("DNC",  3,         "Used Frequency (f0)"),
    "drive":     ("DNC",  4,         "Drive"),
}

EDIT_SIGNALS = {
    "amp_ki":  "Edit24",
    "amp_kp":  "Edit32",
    "pll_kp":  "Edit27",
    "pll_ki":  "Edit22",
    "amp_ref": "Edit23",
    # "freq_ref": "Edit15",  # removed
}

DNC_SIGNALS = {
    "used_freq": 3,   # f0 via DNC
    "drive":     4,
}

CUSTOM_PARAMS = []  # list of (key, (ptype,pcode), label)

UNIT_NOTE = "\n⚠ The value is interpreted in the current unit displayed in the SXM software."
PARAM_DESCRIPTIONS = {
    "amp_ki":   "Integral gain of the amplitude feedback loop (NC-AFM).\nIntegrates amplitude error to correct slow drifts.",
    "amp_kp":   "Proportional gain of the amplitude loop.\nHigher Kp reacts faster but can introduce oscillations.",
    "pll_kp":   "Proportional gain of the PLL (frequency control).\nHigher Kp speeds phase correction but can add noise.",
    "pll_ki":   "Integral gain of the PLL.\nRemoves residual phase/frequency offsets slowly.",
    "amp_ref":  "Target oscillation amplitude setpoint." + UNIT_NOTE,
    "used_freq":"PLL used frequency (controller ‘Use Freq’, DNC=3) — actual f₀." + UNIT_NOTE,
    "drive":    "Drive amplitude used to maintain oscillation (DNC=4)." + UNIT_NOTE,
}

# ---------------- Write-only cache ----------------
def code_id(ptype, pcode):
    return f"{ptype}:{pcode}"

LAST_WRITTEN = {code_id(*PARAMS_CORE[k][:2]): None for k in PARAMS_CORE}

# ---------------- Guardrail helpers ----------------
VOLTAGE_LIMIT_ABS = 10.0  # ±10 V guard

def is_voltage_like(ptype, pcode):
    """Return True for parameters that represent voltages at the output stage."""
    # Amplitude Ref (Edit23) and Drive (DNC 4) are voltage-like in typical setups
    return (ptype == "EDIT" and str(pcode).lower() == "edit23") or (ptype == "DNC" and int(pcode) == 4)

def confirm_voltage_send(parent, ptype, pcode, value):
    """If sending a voltage-like value above ±10 V, ask for confirmation."""
    try:
        v = float(value)
    except Exception:
        return True  # non-numeric handled elsewhere; don't block here
    if is_voltage_like(ptype, pcode) and abs(v) > VOLTAGE_LIMIT_ABS:
        code_txt = pcode if ptype == "EDIT" else f"DNC{pcode}"
        m = QtWidgets.QMessageBox(parent)
        m.setIcon(QtWidgets.QMessageBox.Warning)
        m.setWindowTitle("Confirm High Voltage")
        m.setText(
            f"You are about to send {v} to <b>{code_txt}</b>.\n\n"
            "⚠ SXM interprets values in the <b>current GUI unit</b> (V/mV/µV).\n"
            f"⚠ Do <b>not</b> exceed ±{VOLTAGE_LIMIT_ABS} V at the output unless a divider/attenuator is in place.\n\n"
            "Proceed?"
        )
        m.setStandardButtons(QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok)
        m.setDefaultButton(QtWidgets.QMessageBox.Cancel)
        return m.exec_() == QtWidgets.QMessageBox.Ok
    return True

# ---------------- Mock DDE (offline testing) ----------------
class MockDDEClient:
    def __init__(self):
        self.cache = dict(LAST_WRITTEN)

    def SendWait(self, cmd: str):
        c = cmd.strip().rstrip(";")
        try:
            if c.startswith("ScanPara("):
                inner = c[len("ScanPara("):-1] if c.endswith(")") else c[len("ScanPara("):]
                edit, val = inner.split(",", 1)
                edit = edit.strip().strip("'\"")
                val = float(val)
                self.cache[f"EDIT:{edit}"] = val
                LAST_WRITTEN[f"EDIT:{edit}"] = val
                print(f"[MOCK] ScanPara set {edit} = {val}")
            elif c.startswith("DNCPara("):
                inner = c[len("DNCPara("):-1] if c.endswith(")") else c[len("DNCPara("):]
                n, val = inner.split(",", 1)
                n = int(n.strip())
                val = float(val)
                self.cache[f"DNC:{n}"] = val
                LAST_WRITTEN[f"DNC:{n}"] = val
                print(f"[MOCK] DNCPara set {n} = {val}")
            else:
                print(f"[MOCK] Unknown command: {cmd}")
        except Exception as e:
            print(f"[MOCK] Failed to parse: {cmd} -> {e}")

# ---------------- Write helpers ----------------
def write_param(dde, ptype, pcode, value: float, parent=None):
    """Route write and update cache, with guardrail confirmation for voltage-like params."""
    if not confirm_voltage_send(parent, ptype, pcode, value):
        return  # user cancelled

    if ptype == "EDIT":
        dde.SendWait(f"ScanPara('{pcode}', {value});")
    elif ptype == "DNC":
        dde.SendWait(f"DNCPara({pcode}, {value});")
    else:
        raise ValueError(f"Unknown param type: {ptype}")
    LAST_WRITTEN[code_id(ptype, pcode)] = float(value)

def read_cached(ptype, pcode):
    return LAST_WRITTEN.get(code_id(ptype, pcode), None)

# ---------------- Tab 1: Parameters (table) ----------------
class ParamTable(QtWidgets.QWidget):
    custom_added = QtCore.pyqtSignal()

    def __init__(self, dde_client):
        super().__init__()
        self.dde = dde_client
        layout = QtWidgets.QVBoxLayout(self)

        toolbar = QtWidgets.QHBoxLayout()
        layout.addLayout(toolbar)

        self.apply_btn = QtWidgets.QPushButton("Apply Selected")
        self.apply_btn.clicked.connect(self.apply_selected)
        toolbar.addWidget(self.apply_btn)

        self.auto_send_chk = QtWidgets.QCheckBox("Auto-Send on Edit")
        toolbar.addWidget(self.auto_send_chk)

        self.add_custom_btn = QtWidgets.QPushButton("Add Custom EditXX…")
        self.add_custom_btn.clicked.connect(self.add_custom_param)
        toolbar.addWidget(self.add_custom_btn)

        self.toggle_log_btn = QtWidgets.QPushButton("Show Change Log")
        self.toggle_log_btn.setCheckable(True)
        self.toggle_log_btn.toggled.connect(self.toggle_log)
        toolbar.addWidget(self.toggle_log_btn)

        toolbar.addStretch()

        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Parameter", "Code", "Previous", "Current", "New Value"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        layout.addWidget(self.table)

        self.log_widget = QtWidgets.QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.hide()
        layout.addWidget(self.log_widget)

        self.table.cellChanged.connect(self.on_cell_changed)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Return"), self, self.apply_selected)

        self._rebuild_table()

    def _all_params_list(self):
        core = [(k, PARAMS_CORE[k][:2], PARAMS_CORE[k][2]) for k in PARAMS_CORE]
        custom = CUSTOM_PARAMS[:]
        return core + custom

    def _row_param(self, row):
        return self._all_params_list()[row]

    def _rebuild_table(self):
        rows = self._all_params_list()
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        for row, (key, (ptype, pcode), label) in enumerate(rows):
            p_item = QtWidgets.QTableWidgetItem(label)
            p_item.setToolTip(PARAM_DESCRIPTIONS.get(key, ""))
            p_item.setFlags(p_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(row, 0, p_item)
            code_text = pcode if ptype == "EDIT" else f"DNC{pcode}"
            self._set_item(row, 1, code_text, False, None)
            self._set_item(row, 2, "—", False, QtGui.QColor("#e0e0e0"))
            cur = read_cached(ptype, pcode)
            self._set_item(row, 3, "—" if cur is None else str(cur), False, QtGui.QColor("#e0e0e0"))
            self._set_item(row, 4, "", True, QtGui.QColor("#fff8dc"))
        self.table.blockSignals(False)

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
        key, (ptype, pcode), label = self._row_param(row)
        prev_val_text = self.table.item(row, 3).text()
        self.table.item(row, 2).setText(prev_val_text)
        try:
            write_param(self.dde, ptype, pcode, new_val, parent=self)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "DDE Send Error",
                                          f"Failed to send {ptype}:{pcode} = {new_val}\n\n{e}")
            return
        self.table.item(row, 3).setText(str(new_val))
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
        label = self.table.item(row, 0).text()
        code = self.table.item(row, 1).text()
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_widget.append(f"[{ts}] {label} ({code}): {prev_val} → {new_val}")

    def toggle_log(self, show):
        self.log_widget.setVisible(show)
        self.toggle_log_btn.setText("Hide Change Log" if show else "Show Change Log")

    def add_custom_param(self):
        edit_code, ok = QtWidgets.QInputDialog.getText(self, "Add Custom EditXX", "Enter component code (e.g., Edit37):")
        if not ok or not edit_code.strip():
            return
        edit_code = edit_code.strip()
        if not (edit_code.startswith("Edit") and edit_code[4:].isdigit()):
            QtWidgets.QMessageBox.warning(self, "Invalid", "Please enter a valid code like Edit37.")
            return
        label, ok = QtWidgets.QInputDialog.getText(self, "Label", "Enter a label to show:")
        if not ok or not label.strip():
            return
        label = label.strip()
        existing_keys = set(PARAMS_CORE.keys()).union({k for (k, _, _) in CUSTOM_PARAMS})
        base_key = f"user_{edit_code.lower()}"
        key = base_key
        i = 1
        while key in existing_keys:
            i += 1
            key = f"{base_key}_{i}"
        CUSTOM_PARAMS.append((key, ("EDIT", edit_code), label))
        LAST_WRITTEN.setdefault(code_id("EDIT", edit_code), None)
        self._rebuild_table()
        self.custom_added.emit()

# ---------------- Tab 2: Step Test (symbolic + send + stop) ----------------
class StepTestTab(QtWidgets.QWidget):
    def __init__(self, dde_client):
        super().__init__()
        self.dde = dde_client
        self.timer = None
        self.step_index = 0
        layout = QtWidgets.QVBoxLayout(self)

        # Controls grid
        ctrl_widget = QtWidgets.QWidget()
        ctrl = QtWidgets.QGridLayout(ctrl_widget)
        ctrl.setContentsMargins(8, 6, 8, 6)
        ctrl.setHorizontalSpacing(12)
        ctrl.setVerticalSpacing(6)

        self.param_combo = QtWidgets.QComboBox()
        self.refresh_param_list()
        self.param_combo.setMinimumWidth(200)

        row = 0; col = 0
        ctrl.addWidget(self.param_combo, row, col, 1, 2); col += 2
        low_lbl = QtWidgets.QLabel("Low:"); low_lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        ctrl.addWidget(low_lbl, row, col); col += 1
        self.low_val = QtWidgets.QDoubleSpinBox(); self.low_val.setDecimals(6); self.low_val.setMaximum(1e12); self.low_val.setValue(100.0); self.low_val.setFixedWidth(120)
        ctrl.addWidget(self.low_val, row, col); col += 1
        high_lbl = QtWidgets.QLabel("High:"); high_lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        ctrl.addWidget(high_lbl, row, col); col += 1
        self.high_val = QtWidgets.QDoubleSpinBox(); self.high_val.setDecimals(6); self.high_val.setMaximum(1e12); self.high_val.setValue(101.0); self.high_val.setFixedWidth(120)
        ctrl.addWidget(self.high_val, row, col); col += 1
        per_lbl = QtWidgets.QLabel("Period (s):"); per_lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        ctrl.addWidget(per_lbl, row, col); col += 1
        self.period = QtWidgets.QDoubleSpinBox(); self.period.setDecimals(3); self.period.setMaximum(3600); self.period.setValue(1.000); self.period.setFixedWidth(90)
        ctrl.addWidget(self.period, row, col); col += 1
        steps_lbl = QtWidgets.QLabel("Steps:"); steps_lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        ctrl.addWidget(steps_lbl, row, col); col += 1
        self.steps = QtWidgets.QSpinBox(); self.steps.setMaximum(1000000); self.steps.setValue(20); self.steps.setFixedWidth(90)
        ctrl.addWidget(self.steps, row, col); col += 1
        self.preview_btn = QtWidgets.QPushButton("Preview"); self.preview_btn.setFixedWidth(90)
        self.preview_btn.clicked.connect(self.preview_pattern); ctrl.addWidget(self.preview_btn, row, col); col += 1
        self.start_btn = QtWidgets.QPushButton("Start"); self.start_btn.setFixedWidth(90)
        self.start_btn.clicked.connect(self.start_test); ctrl.addWidget(self.start_btn, row, col); col += 1
        self.stop_btn = QtWidgets.QPushButton("Stop"); self.stop_btn.setFixedWidth(90)
        self.stop_btn.clicked.connect(self.stop_test); self.stop_btn.setEnabled(False)
        ctrl.addWidget(self.stop_btn, row, col)
        layout.addWidget(ctrl_widget)

        # Plot (symbolic)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        axis_pen = pg.mkPen(color='k', width=1)
        for ax in ['bottom', 'left']:
            self.plot_widget.getAxis(ax).setPen(axis_pen)
            self.plot_widget.getAxis(ax).setTextPen('k')
            self.plot_widget.getAxis(ax).setStyle(tickTextOffset=5, **{'tickFont': QtGui.QFont('', 10)})
        plot_item = self.plot_widget.getPlotItem()
        plot_item.layout.setContentsMargins(50, 10, 10, 40)
        self.plot_widget.setLabel('left', 'Value')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        layout.addWidget(self.plot_widget)

        self.log_output = QtWidgets.QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)

    def _all_params_for_combo(self):
        core = [(k, PARAMS_CORE[k][:2], PARAMS_CORE[k][2]) for k in PARAMS_CORE]
        return core + CUSTOM_PARAMS[:]

    def refresh_param_list(self):
        cur_text = self.param_combo.currentText() if hasattr(self, "param_combo") and self.param_combo.count() else None
        items = self._all_params_for_combo()
        if hasattr(self, "param_combo"):
            self.param_combo.blockSignals(True)
            self.param_combo.clear()
        for key, (ptype, pcode), label in items:
            self.param_combo.addItem(label, (key, ptype, pcode, label))
        if cur_text:
            idx = self.param_combo.findText(cur_text, QtCore.Qt.MatchExactly)
            if idx >= 0:
                self.param_combo.setCurrentIndex(idx)
        if hasattr(self, "param_combo"):
            self.param_combo.blockSignals(False)

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
        self.stop_btn.setEnabled(True)
        self.step_index = 0
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.perform_step)
        self.timer.start(int(self.period.value() * 1000))

    def stop_test(self):
        if self.timer and self.timer.isActive():
            self.timer.stop()
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            self.log_output.append(f"[{ts}] Test stopped by user.")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def perform_step(self):
        key, ptype, pcode, label = self.param_combo.currentData()
        low = self.low_val.value()
        high = self.high_val.value()
        value = low if self.step_index % 2 == 0 else high
        try:
            write_param(self.dde, ptype, pcode, value, parent=self)
        except Exception as e:
            self.log_output.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] SEND ERROR: {e}")
            self.stop_test()
            return
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        code_text = pcode if ptype == "EDIT" else f"DNC{pcode}"
        self.log_output.append(f"[{ts}] Set {label} ({code_text}) to {value}")
        self.step_index += 1
        if self.step_index >= self.steps.value():
            self.stop_test()

# ---------------- Main Window (now a QWidget with tabs + footer) ----------------
class MainWindow(QtWidgets.QWidget):
    def __init__(self, dde_client):
        super().__init__()
        self.setWindowTitle("NC-AFM Control")
        self.resize(980, 620)

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)

        # Tabs
        self.tabs = QtWidgets.QTabWidget()
        self.table_tab = ParamTable(dde_client)
        self.step_tab = StepTestTab(dde_client)
        self.table_tab.custom_added.connect(self.step_tab.refresh_param_list)
        self.tabs.addTab(self.table_tab, "Parameters")
        self.tabs.addTab(self.step_tab, "Step Test")

        # Footer warning (always visible)
        self.footer_warning = make_warning_strip(self)

        v.addWidget(self.tabs)
        v.addWidget(self.footer_warning)

# ---------------- Main ----------------
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

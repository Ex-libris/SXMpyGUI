"""
ncafm_module.py
---------------
NC-AFM real-time visualizer (IOCTL driver) + parameter control (DDE).

Features:
    • Reads QPlusAmplitude, Frequency, Drive, Phase via IOCTL driver.
    • Controls parameters via SXMRemote.DDEClient:
        - Amplitude loop Ki/Kp
        - PLL loop Ki/Kp
        - Amplitude Reference
        - Frequency Reference
    • Values are read from SXM at startup and shown in spin boxes.
    • Change a value and press Enter (or click Apply) to send to SXM immediately.
    • Auto-refresh every 2 seconds updates the values unless you’re actively editing.
"""

import sys
import time
import threading
from typing import Dict, Tuple
from PyQt5 import QtWidgets, QtCore

import SXMOscilloscope as scope
import SXMRemote


# ------------------------ Control Panel ------------------------
class ControlPanel(QtWidgets.QWidget):
    """
    Dockable panel for viewing/changing feedback parameters and running step tests.
    Communicates directly with SXMRemote.DDEClient.
    """

    PARAM_MAP: Dict[str, Tuple[str, str]] = {
        # key       EditXX   Label
        "amp_ki":  ("Edit24",  "Amplitude Ki"),
        "amp_kp":  ("Edit32"  "Amplitude Kp"),
        "pll_kp":  ("Edit27", "PLL Kp"),
        "pll_ki":  ("Edit22", "PLL Ki"),
        "amp_ref": ("Edit23", "Amplitude Ref"),
        "freq_ref":("Edit15", "Frequency Ref"),
    }

    def __init__(self, dde_client: SXMRemote.DDEClient, parent=None):
        super().__init__(parent)
        self.client = dde_client
        self._stop_test_flag = False
        self.param_boxes: Dict[str, QtWidgets.QDoubleSpinBox] = {}
        self.ref_boxes: Dict[str, QtWidgets.QDoubleSpinBox] = {}
        self._build_ui()
        self.refresh_from_device()  # load initial values from SXM
        self._start_auto_refresh()

    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(10)

        # Feedback parameters
        fb_group = QtWidgets.QGroupBox("Feedback Loop Parameters")
        fb_form = QtWidgets.QFormLayout(fb_group)
        for key in ["amp_ki", "amp_kp", "pll_kp", "pll_ki"]:
            edit, label = self.PARAM_MAP[key]
            row = QtWidgets.QHBoxLayout()
            sb = QtWidgets.QDoubleSpinBox()
            sb.setDecimals(6)
            sb.setRange(-1e12, 1e12)  # effectively unbounded
            sb.setKeyboardTracking(False)
            sb.editingFinished.connect(lambda k=key, s=sb: self._apply_param(k, s.value()))
            apply_btn = QtWidgets.QPushButton("Apply")
            apply_btn.clicked.connect(lambda _=False, k=key, s=sb: self._apply_param(k, s.value()))
            row.addWidget(sb, 1)
            row.addWidget(apply_btn, 0)
            fb_form.addRow(label + ":", row)
            self.param_boxes[key] = sb
        outer.addWidget(fb_group)

        # References
        ref_group = QtWidgets.QGroupBox("References")
        ref_form = QtWidgets.QFormLayout(ref_group)
        for key in ["amp_ref", "freq_ref"]:
            edit, label = self.PARAM_MAP[key]
            row = QtWidgets.QHBoxLayout()
            sb = QtWidgets.QDoubleSpinBox()
            sb.setDecimals(6)
            sb.setRange(-1e12, 1e12)
            sb.setKeyboardTracking(False)
            sb.editingFinished.connect(lambda k=key, s=sb: self._apply_param(k, s.value()))
            set_btn = QtWidgets.QPushButton("Set")
            set_btn.clicked.connect(lambda _=False, k=key, s=sb: self._apply_param(k, s.value()))
            row.addWidget(sb, 1)
            row.addWidget(set_btn, 0)
            ref_form.addRow(label + ":", row)
            self.ref_boxes[key] = sb
        outer.addWidget(ref_group)

        # Step test
        step_group = QtWidgets.QGroupBox("Step Test (immediate apply)")
        step_form = QtWidgets.QFormLayout(step_group)
        self.step_param = QtWidgets.QComboBox()
        for key, (_edit, label) in self.PARAM_MAP.items():
            self.step_param.addItem(label, key)
        self.step_size = QtWidgets.QDoubleSpinBox(); self.step_size.setDecimals(4); self.step_size.setRange(-1e12, 1e12); self.step_size.setValue(1.0)
        self.step_count = QtWidgets.QSpinBox(); self.step_count.setRange(1, 100); self.step_count.setValue(5)
        self.step_delay = QtWidgets.QDoubleSpinBox(); self.step_delay.setDecimals(2); self.step_delay.setRange(0.01, 60.0); self.step_delay.setValue(1.0)
        self.start_btn = QtWidgets.QPushButton("Start Test"); self.start_btn.setCheckable(True); self.start_btn.toggled.connect(self._toggle_step_test)
        step_form.addRow("Parameter:", self.step_param)
        step_form.addRow("Step size:", self.step_size)
        step_form.addRow("Steps:", self.step_count)
        step_form.addRow("Delay (s):", self.step_delay)
        step_form.addRow(self.start_btn)
        outer.addWidget(step_group)

        outer.addStretch(1)

    def _start_auto_refresh(self):
        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_from_device)
        self.refresh_timer.start(2000)

    def refresh_from_device(self):
        """Update spin boxes from SXM without triggering apply events."""
        for key in self.PARAM_MAP:
            edit, _label = self.PARAM_MAP[key]
            try:
                val = self._dde_get_scan_para(edit)
                if val is not None:
                    box = self.param_boxes.get(key) or self.ref_boxes.get(key)
                    if box is not None and not box.hasFocus():
                        box.blockSignals(True)
                        box.setValue(val)
                        box.blockSignals(False)
            except Exception as e:
                print(f"[ControlPanel] Refresh '{edit}' failed: {e}")

    def _apply_param(self, key: str, value: float):
        edit, label = self.PARAM_MAP[key]
        cmd = f"ScanPara('{edit}', {float(value)});"
        try:
            self.client.SendWait(cmd)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Parameter write failed",
                                           f"Failed to set {label} ({edit}).\n\n{e}")

    def _toggle_step_test(self, checked: bool):
        self._stop_test_flag = not checked
        self.start_btn.setText("Stop Test" if checked else "Start Test")
        if checked:
            threading.Thread(target=self._run_step_test, daemon=True).start()

    def _run_step_test(self):
        key = self.step_param.currentData()
        step = self.step_size.value()
        n = self.step_count.value()
        delay = self.step_delay.value()
        edit, label = self.PARAM_MAP[key]
        try:
            base = self._dde_get_scan_para(edit)
            if base is None:
                raise RuntimeError(f"Could not read initial value for {label} ({edit}).")
            for i in range(1, n + 1):
                if self._stop_test_flag:
                    break
                new_val = base + i * step
                print(f"[StepTest] {label} ({edit}) -> {new_val}")
                self.client.SendWait(f"ScanPara('{edit}', {float(new_val)});")
                time.sleep(max(0.0, delay))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Step test failed", str(e))
        QtCore.QMetaObject.invokeMethod(self.start_btn, "setChecked", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(bool, False))

    def _dde_get_scan_para(self, item: str):
        topic = f"a:=GetScanPara('{item}');\r\n  writeln(a);"
        return self.client.GetPara(topic)


# ------------------------ Main Window ------------------------
class NCAFMApp(scope.ScopeApp):
    """Main app: IOCTL driver for signals + DDE control panel."""
    def __init__(self, driver_source: scope.IDataSource, dde_client: SXMRemote.DDEClient):
        super().__init__(driver_source)
        self.setWindowTitle("NC-AFM Visualizer + Controller (IOCTL + DDE)")
        for cb, name in zip(self.chan_combos, ["QPlusAmpl", "Frequency", "Drive", "Phase"]):
            cb.setCurrentText(name)
        dock = QtWidgets.QDockWidget("Controls", self)
        dock.setWidget(ControlPanel(dde_client))
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)


# ------------------------ Startup ------------------------
def open_driver_or_fail():
    if not getattr(scope, "WIN32_AVAILABLE", False):
        raise RuntimeError("pywin32 is not available; cannot access IOCTL driver.")
    import win32file, win32con  # type: ignore
    try:
        h = win32file.CreateFile(r"\\.\SXM",
                                 win32con.GENERIC_READ | win32con.GENERIC_WRITE,
                                 0, None, win32con.OPEN_EXISTING,
                                 win32con.FILE_ATTRIBUTE_NORMAL, None)
        return scope.DriverDataSource(h)
    except Exception as e:
        raise RuntimeError(f"Could not open IOCTL driver: {e}")


def main():
    app = QtWidgets.QApplication(sys.argv)

    try:
        dde = SXMRemote.DDEClient("SXM", "Remote")
        try:
            dde.SendWait("FeedPara('Mode', 8);")  # optional default mode
        except Exception:
            pass
    except Exception as e:
        QtWidgets.QMessageBox.critical(None, "DDE connection failed",
                                       f"Could not connect to SXM via DDE.\n\n{e}")
        return 1

    try:
        driver_src = open_driver_or_fail()
    except Exception as e:
        QtWidgets.QMessageBox.critical(None, "IOCTL driver not available", str(e))
        return 2

    win = NCAFMApp(driver_src, dde)
    win.resize(1400, 900)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

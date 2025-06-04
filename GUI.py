import sys, os
import numpy as np
from SweepManager import SweepManager
from PlotEngine import PlotEngine
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt


class SpectrogramGeneratorGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.manager = SweepManager()

        self.setWindowTitle("Spectrogram Generator")
        self.resize(1000, 700)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        # Left panel: File tree   UI
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        main_layout.addWidget(left_panel, stretch=1)

        left_layout.addWidget(QtWidgets.QLabel("Loaded Sweeps:"))
        self.file_tree = QtWidgets.QTreeWidget()
        self.file_tree.setHeaderHidden(True)
        self.file_tree.setIndentation(0)

        self.file_tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.file_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_tree.customContextMenuRequested.connect(self.open_context_menu)

        left_layout.addWidget(self.file_tree)

        # Add / Remove buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("Add")
        self.btn_remove = QtWidgets.QPushButton("Remove")
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_remove)
        left_layout.addLayout(btn_layout)

        # Display checkboxes and mode selectors
        display_layout = QtWidgets.QHBoxLayout()
        self.chk_original = QtWidgets.QCheckBox("RAW")
        self.combo_display_org = QtWidgets.QComboBox()
        self.combo_display_org.addItems(["Signal", "Spectrogram", "Both"])
        self.chk_processed = QtWidgets.QCheckBox("PROC")
        self.combo_display_proc = QtWidgets.QComboBox()
        self.combo_display_proc.addItems(["Signal", "Spectrogram", "Both"])
        display_layout.addWidget(self.chk_original)
        display_layout.addWidget(self.combo_display_org)
        display_layout.addWidget(self.chk_processed)
        display_layout.addWidget(self.combo_display_proc)
        left_layout.addLayout(display_layout)

        # FFT & log options
        fft_layout = QtWidgets.QHBoxLayout()
        self.chk_combine = QtWidgets.QCheckBox("Combine all sweeps")
        self.chk_combine.setChecked(False)
        self.chk_log = QtWidgets.QCheckBox("Log Scale")
        fft_layout.addWidget(self.chk_combine)
        fft_layout.addStretch(1)
        fft_layout.addWidget(self.chk_log)
        left_layout.addLayout(fft_layout)

        param_layout = QtWidgets.QHBoxLayout()
        param_layout.addWidget(QtWidgets.QLabel("FFT window:"))
        self.spin_nperseg = QtWidgets.QSpinBox()
        self.spin_nperseg.setRange(32, 8192)
        self.spin_nperseg.setSingleStep(32)
        self.spin_nperseg.setValue(1024)
        param_layout.addWidget(self.spin_nperseg)
        param_layout.addStretch(1)
        param_layout.addWidget(QtWidgets.QLabel("Max Freq:"))
        self.spin_fmax = QtWidgets.QDoubleSpinBox()
        self.spin_fmax.setRange(1, 5000)
        self.spin_fmax.setValue(30.0)
        param_layout.addWidget(self.spin_fmax)
        left_layout.addLayout(param_layout)

        # Plot / Export buttons
        pd_layout = QtWidgets.QHBoxLayout()
        self.btn_plot   = QtWidgets.QPushButton("Plot")
        self.btn_detect = QtWidgets.QPushButton("Detect")
        lbl_threshold = QtWidgets.QLabel("Threshold (SD):", self)
        self.spin_threshold = QtWidgets.QDoubleSpinBox(self)
        self.spin_threshold.setRange(0.1, 10.0)
        self.spin_threshold.setSingleStep(0.1)
        pd_layout.addWidget(lbl_threshold)
        pd_layout.addWidget(self.spin_threshold)
        pd_layout.addWidget(self.btn_plot)
        pd_layout.addWidget(self.btn_detect)
        left_layout.addLayout(pd_layout)

        export_layout = QtWidgets.QHBoxLayout()
        self.btn_export_pdf = QtWidgets.QPushButton("PDF")
        self.btn_export_csv = QtWidgets.QPushButton("CSV")
        export_layout.addWidget(self.btn_export_pdf)
        export_layout.addWidget(self.btn_export_csv)
        left_layout.addLayout(export_layout)

        left_layout.addStretch()

        # Right panel: Embedded PlotEngine
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        main_layout.addWidget(right_panel, stretch=3)

        # Instantiate PlotEngine (matplotlib FigureCanvas) and add to layout
        self.canvas = PlotEngine(parent=self)
        right_layout.addWidget(self.canvas)

        # Status bar (optional)
        self.status_label = QtWidgets.QLabel("Status: Ready")
        right_layout.addWidget(self.status_label)

        # Connect signals
        self.btn_add.clicked.connect(self.add_files)
        self.btn_remove.clicked.connect(self.remove_selected)
        self.btn_plot.clicked.connect(self.plot_selected)
        self.btn_detect.clicked.connect(self.on_detect_clicked)
        self.btn_export_pdf.clicked.connect(self.export_pdf)
        self.btn_export_csv.clicked.connect(self.export_csv)

        # Tree item
        self.file_tree.itemClicked.connect(self.on_tree_item_clicked)

        #save parameter and path
        self.settings = QtCore.QSettings("MyCompany", "SpectrogramGenerator")
        
        last_dir = self.settings.value("lastDir", "", type=str)
        self.lastDir = last_dir 

        draw_raw = self.settings.value("drawRaw", True, type=bool)
        self.chk_original.setChecked(draw_raw)

        self.chk_original.toggled.connect(
            lambda v: self.settings.setValue("drawRaw", v)
        )

        draw_proc = self.settings.value("drawProc", True, type=bool)
        self.chk_processed.setChecked(draw_proc)
        self.chk_processed.toggled.connect(
            lambda v: self.settings.setValue("drawProc", v)
        )

        combine = self.settings.value("combineAll", False, type=bool)
        self.chk_combine.setChecked(combine)
        self.chk_combine.toggled.connect(
            lambda v: self.settings.setValue("combineAll", v)
        )

        saved_thr = self.settings.value("thresholdMult", 3.0, type=float)
        self.spin_threshold.setValue(saved_thr)
        self.spin_threshold.valueChanged.connect(
            lambda v: self.settings.setValue("thresholdMult", v)
        )

        mode_raw = self.settings.value("modeRaw", "Signal", type=str)
        idx_raw = self.combo_display_org.findText(mode_raw)
        if idx_raw >= 0:
            self.combo_display_org.setCurrentIndex(idx_raw)
        self.combo_display_org.currentTextChanged.connect(
            lambda txt: self.settings.setValue("modeRaw", txt)
        )

        mode_proc = self.settings.value("modeProc", "Signal", type=str)
        idx_proc = self.combo_display_proc.findText(mode_proc)
        if idx_proc >= 0:
            self.combo_display_proc.setCurrentIndex(idx_proc)
        self.combo_display_proc.currentTextChanged.connect(
            lambda txt: self.settings.setValue("modeProc", txt)
        )

        nperseg_val = self.settings.value("nperseg", 2048, type=int)
        self.spin_nperseg.setValue(nperseg_val)
        self.spin_nperseg.valueChanged.connect(
            lambda v: self.settings.setValue("nperseg", v)
        )

        fmax_val = self.settings.value("fmax", 30.0, type=float)
        self.spin_fmax.setValue(fmax_val)
        self.spin_fmax.valueChanged.connect(
            lambda v: self.settings.setValue("fmax", v)
        )

        log_scale = self.settings.value("logScale", False, type=bool)
        self.chk_log.setChecked(log_scale)
        self.chk_log.toggled.connect(
        lambda v: self.settings.setValue("logScale", v)
        ) 

    def add_files(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select Files",
            self.settings.value("lastDir", ""),  
            "All Files (*);;ABF Files (*.abf);;HDF5 Files (*.h5)"
        )
        if not files:
            return

        last_dir = os.path.dirname(files[0])
        self.settings.setValue("lastDir", last_dir)

        for fpath in files:
            try:
                display_names = self.manager.load_file(fpath)
                if not display_names:
                    continue
                for name in display_names:
                    self._add_tree_item(name)
                self.status_label.setText(f"Status: Loaded {os.path.basename(fpath)}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Load Error", f"Error loading {fpath}:\n{str(e)}"
                )

    def _add_tree_item(self, display_name):
        item = QtWidgets.QTreeWidgetItem([display_name])
        item.setData(0, QtCore.Qt.UserRole, display_name)
        self.file_tree.addTopLevelItem(item)

    def remove_selected(self):
        """
        Remove selected items from tree. (Does not remove from SweepManager.data)
        """
        paths_to_check = set()
        for item in self.file_tree.selectedItems():
            name = item.data(0, QtCore.Qt.UserRole)
            if name in self.manager.data:
                paths_to_check.add(self.manager.data[name]["filepath"])

        for item in self.file_tree.selectedItems():
            idx = self.file_tree.indexOfTopLevelItem(item)
            self.file_tree.takeTopLevelItem(idx)

        for p in paths_to_check:
            still_exists = False
            for i in range(self.file_tree.topLevelItemCount()):
                nm = self.file_tree.topLevelItem(i).data(0, QtCore.Qt.UserRole)
                if nm in self.manager.data and self.manager.data[nm]["filepath"] == p:
                    still_exists = True
                    break
            if not still_exists and p in self.loaded_paths:
                self.loaded_paths.remove(p)

        self.status_label.setText("Status: Removed selected items")

    def on_tree_item_clicked(self, item, column):
        display_name = item.data(0, QtCore.Qt.UserRole)

        try:
            signal, fs = self.manager.get_signal(display_name, processed=True)
            label = "Processed"
        except KeyError:
            try:
                signal, fs = self.manager.get_signal(display_name, processed=False)
                label = "Raw (fallback)"
            except KeyError:
                QtWidgets.QMessageBox.warning(
                    self, "Plot Error",
                    f"No 'processed' or 'raw' for {display_name}"
                )
                return

        self.canvas.clear()
        self.canvas.plot_processed_signal(signal, fs, label=label)
        self.canvas.draw()
        self.status_label.setText(f"Status: Plotted {display_name} ({label})")

    def plot_selected(self):
        from scipy.signal import spectrogram  # global_max

        selected_items = self.file_tree.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.warning(self, "Warning", "No sweep selected.")
            return
        
        combine = self.chk_combine.isChecked()

        raw_list = []
        proc_list = []
        fs_list = []
        display_order = []
        for item in selected_items:
            name = item.data(0, QtCore.Qt.UserRole)
            display_order.append(name)

            try:
                sig_raw, _ = self.manager.get_signal(name, processed=False)
            except KeyError:
                sig_raw = None
            try:
                sig_proc, fs = self.manager.get_signal(name, processed=True)
            except KeyError:
                sig_proc, fs = sig_raw, None

            if fs is None:
                QtWidgets.QMessageBox.critical(self, "Error", f"No sampling rate for {name}.")
                return

            if self.chk_original.isChecked() and sig_raw is None and sig_proc is not None:
                sig_raw = np.zeros_like(sig_proc)
            if self.chk_processed.isChecked() and sig_proc is None and sig_raw is not None:
                sig_proc = sig_raw.copy()

            raw_list.append(sig_raw)
            proc_list.append(sig_proc)
            fs_list.append(fs)

        if len(set(fs_list)) > 1:
            QtWidgets.QMessageBox.critical(self, "Error", "Selected sweeps have different sampling rates.")
            return
        fs0 = fs_list[0]

        settings = {
            "draw_raw":  self.chk_original.isChecked(),
            "draw_proc": self.chk_processed.isChecked(),
            "mode_raw":  self.combo_display_org.currentText(),   # "Signal"/"Spectrogram"/"Both"
            "mode_proc": self.combo_display_proc.currentText(),
            "nperseg":   self.spin_nperseg.value(),
            "fmax":      self.spin_fmax.value(),
            "log_scale": self.chk_log.isChecked()
        }

        if combine:

            global_max = None
            need_spec = (
                settings["draw_proc"] and settings["mode_proc"] in ["Spectrogram", "Both"]
            ) or (
                settings["draw_raw"] and settings["mode_raw"] in ["Spectrogram", "Both"]
            )
            if need_spec:
                Sxx_max_list = []
                for sig in (proc_list if settings["draw_proc"] else raw_list):
                    if sig is None:
                        continue
                    f_i, t_i, Sxx_i = spectrogram(
                        sig, fs=fs0,
                        nperseg=settings["nperseg"],
                        scaling="density",
                        mode="psd"
                    )
                    mask_i = f_i <= settings["fmax"]
                    Sxx_i = Sxx_i[mask_i, :]
                    if Sxx_i.size > 0:
                        Sxx_max_list.append(np.max(Sxx_i))
                if Sxx_max_list:
                    global_max = max(Sxx_max_list)

            sig_raw_concat = None
            sig_proc_concat = None
            if settings["draw_raw"]:
                arrays = [arr for arr in raw_list if arr is not None]
                if arrays:
                    sig_raw_concat = np.concatenate(arrays)
            if settings["draw_proc"]:
                arrays = [arr for arr in proc_list if arr is not None]
                if arrays:
                    sig_proc_concat = np.concatenate(arrays)

            self.canvas.clear()
            self.canvas.plot_extra(
                signal_raw  = sig_raw_concat,
                signal_proc = sig_proc_concat,
                fs          = fs0,
                settings    = settings,
                global_max  = global_max
            )
            self.canvas.draw()
            self.status_label.setText(f"Plotted concatenated {len(display_order)} sweeps.")

        else:

            first_name = display_order[0]
            try:
                sig_raw, _ = self.manager.get_signal(first_name, processed=False)
            except KeyError:
                sig_raw = None
            try:
                sig_proc, fs = self.manager.get_signal(first_name, processed=True)
            except KeyError:
                sig_proc, fs = sig_raw, None

            if fs is None:
                QtWidgets.QMessageBox.critical(self, "Error", f"No sampling rate for {first_name}.")
                return

            if settings["draw_raw"] and sig_raw is None and sig_proc is not None:
                sig_raw = np.zeros_like(sig_proc)
            if settings["draw_proc"] and sig_proc is None and sig_raw is not None:
                sig_proc = sig_raw.copy()

            global_max = None
            need_spec = (
                settings["draw_proc"] and settings["mode_proc"] in ["Spectrogram", "Both"]
            ) or (
                settings["draw_raw"] and settings["mode_raw"] in ["Spectrogram", "Both"]
            )
            if need_spec:
                data_sig = sig_proc if sig_proc is not None else sig_raw
                f_i, t_i, Sxx_i = spectrogram(
                    data_sig, fs=fs,
                    nperseg=settings["nperseg"],
                    scaling="density",
                    mode="psd"
                )
                mask_i = f_i <= settings["fmax"]
                Sxx_i = Sxx_i[mask_i, :]
                if Sxx_i.size > 0:
                    global_max = np.max(Sxx_i)

            self.canvas.clear()
            self.canvas.plot_extra(
                signal_raw  = sig_raw,
                signal_proc = sig_proc,
                fs          = fs,
                settings    = settings,
                global_max  = global_max
            )
            self.canvas.draw()
            self.status_label.setText(f"Plotted single sweep: {first_name}")

    def clear_all(self):
        self.file_tree.clear()
        self.manager.data.clear()

    def open_context_menu(self, position):
        menu = QtWidgets.QMenu()
        remove_action = menu.addAction("Remove Selected")
        select_all_action = menu.addAction("Select All")
        clear_action = menu.addAction("clear All")

        action = menu.exec_(self.file_tree.viewport().mapToGlobal(position))
        if action == remove_action:
            self.remove_selected()
        elif action == clear_action:
            self.clear_all()
        elif action == select_all_action:
            self.file_tree.selectAll()

    def on_detect_clicked(self):
        if not hasattr(self.canvas, "last_Sxx_norm"):
            QtWidgets.QMessageBox.warning(self, "Warning", "Please plot a spectrogram first.")
            return
 
        threshold_mult = self.spin_threshold.value()
        event_pairs = self.canvas.detect_power_events(threshold_mult)
        
        if not event_pairs:
            QtWidgets.QMessageBox.information(self, "Detect Result", "No events detected.")
            return

        self.canvas.plot_detection_lines(event_pairs)
        self.status_label.setText(f"Detected {len(event_pairs)} event(s).")

    def export_pdf(self):
        """
        Placeholder: implement PDF export here, using current canvas.figure.savefig(...)
        """
        self.status_label.setText("[TODO] Exporting to PDF...")

    def export_csv(self):
        """
        Placeholder: implement CSV export logic here.
        """
        self.status_label.setText("[TODO] Exporting to CSV...")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = SpectrogramGeneratorGUI()
    gui.show()
    sys.exit(app.exec_())

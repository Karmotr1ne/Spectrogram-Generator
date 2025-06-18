import sys, os, re
import numpy as np
from SweepManager import SweepManager
from PlotEngine import PlotEngine
from ExportManager import ExportManager
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGroupBox, QGridLayout, QVBoxLayout, QHBoxLayout, QFrame


class SpectrogramGeneratorGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.manager = SweepManager()
        self.exporter = ExportManager()

        self.currently_plotted_items = []
        self.is_current_plot_combined = False
        self.plot_segment_map = []

        self.setWindowTitle("Spectrogram Generator")
        self.resize(1200, 750)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
 
        # 1. Create a top-level horizontal layout for the main window
        main_layout = QHBoxLayout(central)
        
        # 2. Create a splitter that will manage the left and right panels
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(splitter)
        left_panel = QtWidgets.QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Group 1: File List
        file_group = QGroupBox("Loaded Sweeps")
        file_layout = QVBoxLayout(file_group)
        
        self.file_tree = QtWidgets.QTreeWidget()
        self.file_tree.setHeaderHidden(True)
        self.file_tree.setIndentation(0)
        self.file_tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.file_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        file_layout.addWidget(self.file_tree)

        file_btn_layout = QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("Add Files")
        self.btn_remove = QtWidgets.QPushButton("Remove Selected")
        file_btn_layout.addWidget(self.btn_add)
        file_btn_layout.addWidget(self.btn_remove)
        file_layout.addLayout(file_btn_layout)
        left_layout.addWidget(file_group)

        # Group 2: Display Options
        display_group = QGroupBox("Display Options")
        display_v_layout = QVBoxLayout(display_group) 

        raw_proc_layout = QHBoxLayout()
        self.chk_original = QtWidgets.QCheckBox("RAW")
        self.combo_display_org = QtWidgets.QComboBox()
        self.combo_display_org.addItems(["Signal", "Spectrogram", "Both"])
        self.chk_processed = QtWidgets.QCheckBox("PROC")
        self.combo_display_proc = QtWidgets.QComboBox()
        self.combo_display_proc.addItems(["Signal", "Spectrogram", "Both"])
        raw_proc_layout.addWidget(self.chk_original)
        raw_proc_layout.addWidget(self.combo_display_org)
        raw_proc_layout.addStretch()
        raw_proc_layout.addWidget(self.chk_processed)
        raw_proc_layout.addWidget(self.combo_display_proc)
        display_v_layout.addLayout(raw_proc_layout)

        other_display_layout = QHBoxLayout()
        self.chk_combine = QtWidgets.QCheckBox("Combine all sweeps")
        self.chk_log = QtWidgets.QCheckBox("Log Scale")
        other_display_layout.addWidget(self.chk_combine)
        other_display_layout.addStretch()
        other_display_layout.addWidget(self.chk_log)
        display_v_layout.addLayout(other_display_layout)
        left_layout.addWidget(display_group)

        # Group 3: Analysis Parameters
        params_group = QGroupBox("Analysis Parameters")
        params_layout = QGridLayout(params_group)
        params_layout.addWidget(QtWidgets.QLabel("FFT window:"), 0, 0)
        self.spin_nperseg = QtWidgets.QSpinBox()
        self.spin_nperseg.setRange(32, 8192)
        self.spin_nperseg.setSingleStep(32)
        params_layout.addWidget(self.spin_nperseg, 0, 1)
        
        params_layout.addWidget(QtWidgets.QLabel("Min Freq:"), 1, 0)
        self.spin_fmin = QtWidgets.QDoubleSpinBox()
        self.spin_fmin.setRange(0, 5000)
        params_layout.addWidget(self.spin_fmin, 1, 1)
        
        params_layout.addWidget(QtWidgets.QLabel("Max Freq:"), 2, 0)
        self.spin_fmax = QtWidgets.QDoubleSpinBox()
        self.spin_fmax.setRange(1, 5000)
        params_layout.addWidget(self.spin_fmax, 2, 1)
        left_layout.addWidget(params_group)

        # Group 4: Automatic Detection (Unsupervised)
        auto_detect_group = QGroupBox("Unsupervised Detection")
        auto_detect_layout = QVBoxLayout(auto_detect_group)
        self.btn_plot = QtWidgets.QPushButton("Plot Signal")
        self.btn_plot.setToolTip("Plot the selected signal(s) based on the display options.")
        auto_detect_layout.addWidget(self.btn_plot)

        self.btn_detect = QtWidgets.QPushButton("Auto-Detect Bursts")
        self.btn_detect.setToolTip("Run unsupervised HMM detection on the signal.")
        auto_detect_layout.addWidget(self.btn_detect)
        left_layout.addWidget(auto_detect_group)

        # Group 5: Manual Correction & Training (Semi-Supervised)
        semi_supervised_group = QGroupBox("Semi-Supervised Detection")
        semi_supervised_layout = QVBoxLayout(semi_supervised_group)

        self.chk_enable_editing = QtWidgets.QCheckBox("Enable Manual Editing")
        self.chk_enable_editing.setToolTip("Enable manual adding, moving, and deleting of burst regions.\nThis is the first step for refining the model.")
        semi_supervised_layout.addWidget(self.chk_enable_editing)

        self.btn_refine_model = QtWidgets.QPushButton("Refine Model from Edits")
        self.btn_refine_model.setToolTip("After manually correcting detections, use them to train and refine the HMM model.")
        self.btn_refine_model.setEnabled(False) 
        semi_supervised_layout.addWidget(self.btn_refine_model)

        self.btn_learn_and_detect = QtWidgets.QPushButton("Learn from Examples")
        self.btn_learn_and_detect.setToolTip("After manually drawing a few 'perfect' examples, train a model and find all similar bursts.")
        self.btn_learn_and_detect.setEnabled(False) 
        semi_supervised_layout.addWidget(self.btn_learn_and_detect)
        
        self.btn_reset_model = QtWidgets.QPushButton("Reset Model")
        self.btn_reset_model.setToolTip("Reset the HMM model to its initial, untrained state.")
        self.btn_reset_model.clicked.connect(self.on_reset_model_clicked)
        semi_supervised_layout.addWidget(self.btn_reset_model)
        left_layout.addWidget(semi_supervised_group)

        # Group 6: Export
        export_group = QGroupBox("Export")
        export_layout = QHBoxLayout(export_group)
        self.btn_export_pdf = QtWidgets.QPushButton("Export PDF")
        self.btn_export_csv = QtWidgets.QPushButton("Export CSV")
        self.btn_band_power = QtWidgets.QPushButton("Calculate Band Power")
        export_layout.addWidget(self.btn_export_pdf)
        export_layout.addWidget(self.btn_export_csv)
        export_layout.addWidget(self.btn_band_power)
        left_layout.addWidget(export_group)

        left_layout.addStretch()

        # --- Right panel: Embedded PlotEngine ---
        right_panel = QtWidgets.QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.canvas = PlotEngine(parent=self)
        right_layout.addWidget(self.canvas, stretch=1)
        self.status_label = QtWidgets.QLabel("Status: Ready")
        # The status label has a default stretch of 0, so it will only take its minimum height.
        right_layout.addWidget(self.status_label)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 850]) 
        splitter.setStretchFactor(1, 1)
        self.connect_signals()
        self.load_settings()

    def connect_signals(self):
        self.btn_add.clicked.connect(self.add_files)
        self.btn_remove.clicked.connect(self.remove_selected)
        self.file_tree.customContextMenuRequested.connect(self.open_context_menu)
        self.file_tree.itemClicked.connect(self.on_tree_item_clicked)
        
        self.btn_plot.clicked.connect(self.plot_selected)
        self.btn_detect.clicked.connect(self.on_detect_clicked)
        self.chk_enable_editing.toggled.connect(self.on_editing_mode_changed)

        self.btn_refine_model.clicked.connect(self.on_refine_model_clicked)
        self.btn_learn_and_detect.clicked.connect(self.on_learn_and_detect_clicked)

        self.btn_export_pdf.clicked.connect(self.export_pdf)
        self.btn_export_csv.clicked.connect(self.export_csv)
        self.btn_band_power.clicked.connect(self.on_band_power_clicked)

    def load_settings(self):
        self.settings = QtCore.QSettings("MyCompany", "SpectrogramGenerator")
        
        self.lastDir = self.settings.value("lastDir", "", type=str)

        self.chk_original.setChecked(self.settings.value("drawRaw", True, type=bool))
        self.chk_original.toggled.connect(lambda v: self.settings.setValue("drawRaw", v))

        self.chk_processed.setChecked(self.settings.value("drawProc", True, type=bool))
        self.chk_processed.toggled.connect(lambda v: self.settings.setValue("drawProc", v))

        self.chk_combine.setChecked(self.settings.value("combineAll", False, type=bool))
        self.chk_combine.toggled.connect(lambda v: self.settings.setValue("combineAll", v))

        mode_raw = self.settings.value("modeRaw", "Signal", type=str)
        idx_raw = self.combo_display_org.findText(mode_raw)
        if idx_raw >= 0: self.combo_display_org.setCurrentIndex(idx_raw)
        self.combo_display_org.currentTextChanged.connect(lambda txt: self.settings.setValue("modeRaw", txt))

        mode_proc = self.settings.value("modeProc", "Signal", type=str)
        idx_proc = self.combo_display_proc.findText(mode_proc)
        if idx_proc >= 0: self.combo_display_proc.setCurrentIndex(idx_proc)
        self.combo_display_proc.currentTextChanged.connect(lambda txt: self.settings.setValue("modeProc", txt))

        self.spin_nperseg.setValue(self.settings.value("nperseg", 1024, type=int))
        self.spin_nperseg.valueChanged.connect(lambda v: self.settings.setValue("nperseg", v))

        self.spin_fmin.setValue(self.settings.value("fmin", 0.0, type=float))
        self.spin_fmin.valueChanged.connect(lambda v: self.settings.setValue("fmin", v))

        self.spin_fmax.setValue(self.settings.value("fmax", 30.0, type=float))
        self.spin_fmax.valueChanged.connect(lambda v: self.settings.setValue("fmax", v))

        self.chk_log.setChecked(self.settings.value("logScale", False, type=bool))
        self.chk_log.toggled.connect(lambda v: self.settings.setValue("logScale", v))

    
    def on_refine_model_clicked(self):
        if not self.canvas.burst_patches:
            QtWidgets.QMessageBox.warning(self, "Action Required",
                                            "There are no burst regions on the plot to learn from. "
                                            "Please perform an auto-detection and/or manually add regions first.")
            return
        self.on_learn_and_detect_clicked()

    def on_learn_and_detect_clicked(self):
        """Handles the semi-supervised 'Learn from Examples' workflow."""
        if self.canvas.spec_data_source is None:
            QtWidgets.QMessageBox.warning(self, "Action Required", "Please plot a signal before learning from it.")
            return
        
        if not self.canvas.burst_patches:
            QtWidgets.QMessageBox.warning(self, "Action Required", "Please enable manual editing and draw at least one example region to learn from.")
            return

        self.status_label.setText("Status: Learning from examples and detecting... Please wait.")
        QtWidgets.QApplication.processEvents()

        try:
            event_pairs = self.canvas.learn_and_detect()
            
            if not event_pairs:
                QtWidgets.QMessageBox.information(self, "Detection Result", "Could not detect any bursts after learning.")
                self.status_label.setText("Status: Learning complete. No bursts found.")
                return

            self.canvas.plot_detection_lines(event_pairs)
            self.status_label.setText(f"Status: Learned from examples and detected {len(event_pairs)} event(s).")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Learning Error", f"An error occurred during learning:\n{e}")
            self.status_label.setText("Status: Learning or detection failed.")

    def on_editing_mode_changed(self, is_checked):
        self.canvas.set_editing_enabled(is_checked)
        if is_checked:
            self.status_label.setText("Status: Manual editing enabled. Left-click drag to add/move, right-click to remove.")
            self.btn_refine_model.setEnabled(True)
            self.btn_learn_and_detect.setEnabled(True)
        else:
            self.status_label.setText("Status: Manual editing disabled.")
            self.btn_refine_model.setEnabled(False)
            self.btn_learn_and_detect.setEnabled(False)

    def add_files(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select Files", self.settings.value("lastDir", ""),  
            "All Files (*);;ABF Files (*.abf);;HDF5 Files (*.h5)"
        )
        if not files: return
        last_dir = os.path.dirname(files[0])
        self.settings.setValue("lastDir", last_dir)
        for fpath in files:
            try:
                display_names = self.manager.load_file(fpath)
                if not display_names: continue
                for name in display_names:
                    item = QtWidgets.QTreeWidgetItem([os.path.basename(name)])
                    item.setData(0, QtCore.Qt.UserRole, name)
                    self.file_tree.addTopLevelItem(item)
                self.status_label.setText(f"Status: Loaded {os.path.basename(fpath)}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Load Error", f"Error loading {fpath}:\n{str(e)}")

    def remove_selected(self):
        for item in self.file_tree.selectedItems():
            self.file_tree.takeTopLevelItem(self.file_tree.indexOfTopLevelItem(item))
        self.status_label.setText("Status: Removed selected items")

    def on_reset_model_clicked(self):
        """Resets the HMM model and clears the canvas."""
        self.canvas.reset_model()
        self.canvas.clear() # Clears both plot and burst_patches list
        self.canvas.draw()
        QtWidgets.QMessageBox.information(self, "Model Status",
                                            "The HMM model has been reset and the canvas has been cleared.")
        self.status_label.setText("Status: HMM model reset. Canvas cleared.")

    def on_tree_item_clicked(self, item, column):
        """
        Handles a click on a file list item by delegating to the main plot function.
        """
        # When an item is clicked, it's not always selected, so we manually
        # clear the previous selection and select the clicked item.
        self.file_tree.clearSelection()
        item.setSelected(True)

        # Uncheck 'Combine all sweeps' for clarity when plotting a single sweep
        self.chk_combine.setChecked(False)

        # Call the main plotting function
        self.plot_selected()

    def plot_selected(self):
        selected_items = self.file_tree.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.warning(self, "Warning", "No sweep selected.")
            return

        self.currently_plotted_items = selected_items
        self.is_current_plot_combined = self.chk_combine.isChecked()

        sweeps_info = []
        fs_set = set()

        # This new loop is more robust. It gets the "true" fs first.
        for item in selected_items:
            name = item.data(0, QtCore.Qt.UserRole)
            
            # Step 1: Get the single, authoritative sampling rate from the manager.
            # This is the "source of truth" loaded directly from the file.
            try:
                definitive_fs = self.manager.data[name]['fs']
                if definitive_fs is None or definitive_fs <= 0:
                    # This catches cases where fs might be None or 0 in the file data
                    raise ValueError(f"Invalid sampling rate ({definitive_fs})")
            except (KeyError, ValueError) as e:
                QtWidgets.QMessageBox.critical(self, "Data Error", f"Could not retrieve a valid sampling rate for '{os.path.basename(name)}'.\nPlease check the source file's metadata.\n\nError: {e}")
                return
            
            fs_set.add(definitive_fs)

            # Step 2: Get the signal data. We will use the definitive_fs we just confirmed.
            try:
                sig_raw, _ = self.manager.get_signal(name, processed=False)
            except KeyError:
                sig_raw = None
            try:
                sig_proc, _ = self.manager.get_signal(name, processed=True)
            except KeyError:
                sig_proc = None

            sweeps_info.append({'item': item, 'signal_raw': sig_raw, 'signal_proc': sig_proc, 'fs': definitive_fs})

        # Step 3: Check for fs consistency across multiple files
        if len(fs_set) > 1:
            QtWidgets.QMessageBox.critical(self, "Error", "Selected sweeps have different sampling rates and cannot be plotted together.")
            return

        # Step 4: Gather UI settings
        settings = {
            "combine": self.is_current_plot_combined,
            "draw_raw": self.chk_original.isChecked(),
            "draw_proc": self.chk_processed.isChecked(),
            "mode_raw": self.combo_display_org.currentText(),
            "mode_proc": self.combo_display_proc.currentText(),
            "nperseg": self.spin_nperseg.value(),
            "fmin": self.spin_fmin.value(),
            "fmax": self.spin_fmax.value(),
            "log_scale": self.chk_log.isChecked()
        }
        
        was_editing = self.chk_enable_editing.isChecked()
        if was_editing: self.canvas.set_editing_enabled(False)

        # Step 5: Delegate all plotting work to the PlotEngine
        self.canvas.plot_sweeps(sweeps_info, settings)
        
        # Step 6: Update status label
        if settings["combine"]:
            status_text = f"Plotted concatenated {len(sweeps_info)} sweeps."
        else:
            name = sweeps_info[0]['item'].data(0, QtCore.Qt.UserRole)
            status_text = f"Plotted single sweep: {os.path.basename(name)}"
        self.status_label.setText(status_text)
        
        if was_editing: self.canvas.set_editing_enabled(True)
        self.canvas.draw()

        # Step 7: Absolute power
        absolute_power = self.canvas.calculate_absolute_power()
        if absolute_power is not None:
            self.status_label.setText(self.status_label.text() + f" | Total Power: {absolute_power:.2e}")

    def on_detect_clicked(self):
        if self.canvas.spec_data_source is None:
            QtWidgets.QMessageBox.warning(self, "Action Required", "Please plot a signal before running detection.")
            return

        self.status_label.setText("Status: Running HMM detection... Please wait.")
        QtWidgets.QApplication.processEvents()

        try:
            event_pairs = self.canvas.unsupervised_detect()

            if not event_pairs:
                QtWidgets.QMessageBox.information(self, "Detection Result", "No events detected.")
                self.status_label.setText("Status: HMM detected 0 events.")
                return

            self.canvas.plot_detection_lines(event_pairs)
            self.status_label.setText(f"Status: HMM detected {len(event_pairs)} event(s).")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "HMM Error", f"An error occurred during HMM detection:\n{e}")
            self.status_label.setText("Status: HMM detection failed.")
    
    def clear_all(self):
        self.file_tree.clear()
        self.canvas.clear()
        self.canvas.draw()
        self.chk_enable_editing.setChecked(False)
        self.currently_plotted_items = []

    def open_context_menu(self, position):
        menu = QtWidgets.QMenu()
        remove_action = menu.addAction("Remove Selected")
        select_all_action = menu.addAction("Select All")
        clear_action = menu.addAction("Clear All")

        action = menu.exec_(self.file_tree.viewport().mapToGlobal(position))
        if action == remove_action: self.remove_selected()
        elif action == clear_action: self.clear_all()
        elif action == select_all_action: self.file_tree.selectAll()
            
    def export_pdf(self):
        """Exports the current plot to a PDF file."""
        status = self.exporter.export_to_pdf(self.canvas.fig, self)
        self.status_label.setText(status)

    def export_csv(self):
        """Handles exporting burst data using the plot context stored in the canvas."""
        if not self.canvas.currently_plotted_items:
            QtWidgets.QMessageBox.warning(self, "No Plot Context", 
                                            "Please plot a signal first before exporting.")
            return

        if not self.canvas.burst_patches:
            QtWidgets.QMessageBox.warning(self, "No Data",
                                            "There are no detected bursts on the plot to export.")
            return

        first_item_name = self.canvas.currently_plotted_items[0].data(0, QtCore.Qt.UserRole)
        base_name = re.sub(r'_sweep\d+$', '', os.path.basename(first_item_name))
        default_filename = f"{base_name}_bursts.csv"

        last_export_dir = self.settings.value("lastExportDir", self.lastDir, type=str)

        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Burst Data as CSV", os.path.join(last_export_dir, default_filename), "CSV Files (*.csv)"
        )

        if not filepath:
            self.status_label.setText("Status: Export cancelled.")
            return

        new_export_dir = os.path.dirname(filepath)
        self.settings.setValue("lastExportDir", new_export_dir)

        # All required context is now inside the self.canvas object
        status = self.exporter.export_to_csv(filepath=filepath, plot_engine=self.canvas)
        self.status_label.setText(status)

    def on_band_power_clicked(self):
        if self.canvas.last_Sxx is None:
            QtWidgets.QMessageBox.warning(self, "No Spectrogram",
                                        "Please plot a signal with a spectrogram first.")
            return

        if self.chk_log.isChecked():
            QtWidgets.QMessageBox.warning(
                self, "Band Power Disabled in Log Scale",
                "Please disable log scale before calculating band power.\n\n"
                "Band power must be calculated from the original (linear) spectrum."
            )
            return

        band_powers = self.canvas.calculate_band_powers()
        if band_powers is None:
            QtWidgets.QMessageBox.warning(self, "Error", "Unable to compute band power.")
            return

        msg = ""
        for band, power in band_powers.items():
            msg += f"{band}: {power:.4f}\n"

        # Show a dialog with selectable and copyable text
        text_dialog = QtWidgets.QDialog(self)
        text_dialog.setWindowTitle("Band Power Results")
        layout = QtWidgets.QVBoxLayout(text_dialog)
        label = QtWidgets.QLabel("Power per frequency band (arbitrary units):")
        layout.addWidget(label)
        
        text_box = QtWidgets.QTextEdit()
        text_box.setReadOnly(True)
        text_box.setText(msg)
        layout.addWidget(text_box)

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(text_dialog.accept)
        layout.addWidget(btn_close)

        text_dialog.resize(400, 300)
        text_dialog.exec_()

if __name__ == "__main__":
    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    gui = SpectrogramGeneratorGUI()
    gui.show()
    sys.exit(app.exec_())

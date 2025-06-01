import sys, os
import numpy as np
from SweepManager import SweepManager
from PlotEngine import PlotEngine
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt


class SpectrogramGeneratorGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.settings = QtCore.QSettings("MyCompany", "SpectrogramGenerator")
        self.loaded_paths = set()

        self.manager = SweepManager()

        self.setWindowTitle("Spectrogram Generator")
        self.resize(1000, 700)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        # Left panel: File tree + UI
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
        action_layout = QtWidgets.QHBoxLayout()
        self.btn_plot = QtWidgets.QPushButton("Plot")
        self.btn_export_pdf = QtWidgets.QPushButton("PDF")
        self.btn_export_csv = QtWidgets.QPushButton("CSV")
        action_layout.addWidget(self.btn_plot)
        action_layout.addWidget(self.btn_export_pdf)
        action_layout.addWidget(self.btn_export_csv)
        left_layout.addLayout(action_layout)

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
        self.btn_export_pdf.clicked.connect(self.export_pdf)
        self.btn_export_csv.clicked.connect(self.export_csv)

        # Tree item
        self.file_tree.itemClicked.connect(self.on_tree_item_clicked)

    def add_files(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select .abf or .h5 Files", "", "Signal Files (*.abf *.h5)"
        )
        for fpath in files:
            if fpath in self.loaded_paths:
                continue
            try:
                display_names = self.manager.load_file(fpath)
                if not display_names:
                    continue
                for name in display_names:
                    item = QtWidgets.QTreeWidgetItem([name])
                    item.setData(0, QtCore.Qt.UserRole, name)
                    self.file_tree.addTopLevelItem(item)
                self.loaded_paths.add(fpath)
                self.status_label.setText(f"Status: Loaded {os.path.basename(fpath)}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Load Error", f"Error loading {fpath}:\n{str(e)}"
                )

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
        selected_items = self.file_tree.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.warning(self, "Warning", "No sweep selected.")
            return

        # For simplicity: only plot the first selected item here.
        display_name = selected_items[0].data(0, QtCore.Qt.UserRole)

        # Build settings dict from UI
        settings = {
            "draw_raw": self.chk_original.isChecked(),
            "draw_proc": self.chk_processed.isChecked(),
            "mode_raw": self.combo_display_org.currentText(),
            "mode_proc": self.combo_display_proc.currentText(),
            "fmax": self.spin_fmax.value(),
            "nperseg": self.spin_nperseg.value(),
            "log_scale": self.chk_log.isChecked()
        }

        try:
            # Attempt to fetch raw and processed
            signal_raw, fs_raw = self.manager.get_signal(display_name, processed=False)
        except KeyError:
            signal_raw, fs_raw = None, None

        try:
            signal_proc, fs_proc = self.manager.get_signal(display_name, processed=True)
        except KeyError:
            if signal_raw is not None:
                signal_proc, fs_proc = signal_raw, fs_raw
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Error", f"No 'processed' or 'raw' for {display_name}."
                )
                return

        fs = fs_proc if fs_proc is not None else fs_raw
        if fs is None:
            QtWidgets.QMessageBox.warning(
                self, "Error", f"No valid sampling rate for {display_name}."
            )
            return

        if settings["draw_raw"] and signal_raw is None:
            signal_raw = np.zeros_like(signal_proc)
            fs_raw = fs

        self.canvas.plot_extra(signal_raw, signal_proc, fs, settings)
        self.status_label.setText(f"Status: Drew signals for {display_name}")

    def reload_all(self):
        self.file_tree.clear()
        self.manager.data.clear()
        self.loaded_paths.clear()

    def open_context_menu(self, position):
        menu = QtWidgets.QMenu()
        remove_action = menu.addAction("Remove Selected")
        reload_action = menu.addAction("Reload All")

        action = menu.exec_(self.file_tree.viewport().mapToGlobal(position))
        if action == remove_action:
            self.remove_selected()
        elif action == reload_action:
            self.reload_all()

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

import csv
import re, os
import numpy as np
import matplotlib 
matplotlib.use("Agg")  # Use non-GUI backend
import matplotlib.pyplot as plt
from pathlib import Path
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog

class ExportManager:

    def export_to_csv(self, filepath, plot_engine):
        """
        Exports burst data. All context is derived from the plot_engine object.
        """
        if not plot_engine.burst_patches:
            return "Error: No burst data to export."

        try:
            # Directly get the context from the plot_engine object
            segment_map = plot_engine.segment_map
            plotted_items = plot_engine.currently_plotted_items
            is_combined = True if segment_map else False

            t_grid = getattr(plot_engine, 'last_t', None)
            
            if hasattr(plot_engine, 'last_detected_events'):
                time_pairs = list(plot_engine.last_detected_events)
            else:
                # fallback
                time_pairs = [
                    (p.get_extents().x0, p.get_extents().x1)
                    for p in plot_engine.burst_patches
                ]
                for sig_patch, _ in plot_engine.burst_patches:
                    ext = sig_patch.get_extents()
                    time_pairs.append((min(ext.x0, ext.x1), max(ext.x0, ext.x1)))

            sorted_bursts = sorted(time_pairs)

            output_rows = []
            for i, (start_time, end_time) in enumerate(sorted_bursts):
                burst_id = i + 1

                if i == 0:
                    inter_burst_interval = np.nan
                else:
                    previous_end_time = sorted_bursts[i-1][1]
                    inter_burst_interval = start_time - previous_end_time

                source_file = "Unknown"
                sweep_index_str = "Unknown"

                if is_combined:
                    for segment in segment_map:
                        if segment['start_time_combined'] <= start_time < segment['end_time_combined']:
                            source_item = segment['source_item']
                            full_name = source_item.data(0, QtCore.Qt.UserRole)
                            source_file = re.sub(r'_sweep\d+$', '', os.path.basename(full_name))
                            match = re.search(r'_sweep(\d+)$', full_name)
                            if match:
                                sweep_index_str = match.group(1)
                            break
                else: # Not combined
                    if plotted_items:
                        full_name = plotted_items[0].data(0, QtCore.Qt.UserRole)
                        source_file = re.sub(r'_sweep\d+$', '', os.path.basename(full_name))
                        match = re.search(r'_sweep(\d+)$', full_name)
                        if match:
                            sweep_index_str = match.group(1)

                output_rows.append([
                    burst_id, source_file, sweep_index_str,
                    start_time, end_time, inter_burst_interval
                ])

            header = [
                'Burst ID', 'Source File', 'Sweep', 
                'Start Time (s)', 'End Time (s)', 'Inter Burst Interval (s)'
            ]

            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
                writer.writerows(output_rows)

            return f"Successfully exported {len(output_rows)} events to {os.path.basename(filepath)}"
        except Exception as e:
            return f"Error exporting to CSV: {e}"

    def export_to_png_transparent(self, plot_engine, parent_widget=None):
        plot_engine.draw()
        fig = plot_engine.fig

        filepath, _ = QFileDialog.getSaveFileName(
            parent_widget, "Save Figure as Transparent PNG", "", "PNG Files (*.png)"
        )

        if not filepath:
            return "Export cancelled."

        try:
            for ax in fig.axes:
                legend = ax.get_legend()
                if legend: legend.remove()

            fig.savefig(filepath, format='png', dpi=3000, transparent=True, bbox_inches='tight')
            return f"Successfully exported transparent PNG to {filepath}"
        except Exception as e:
            return f"Error exporting PNG: {e}"
        
    def export_batch_signals_to_png(self, plot_engine, sweep_manager, selected_items, parent_widget=None):

        if not selected_items:
            return "No items selected for batch export."

        # Ask user to select output folder
        out_dir = QFileDialog.getExistingDirectory(parent_widget, "Select Folder to Save PNGs")
        if not out_dir:
            return "Export cancelled."

        success_list = []
        
        # inside export_batch_signals_to_png
        max_amplitude = 0

        for item in selected_items:
            name = item.data(0, QtCore.Qt.UserRole)
            entry = sweep_manager.data.get(name, {})

            use_proc = getattr(parent_widget.chk_processed, 'isChecked', lambda: False)()
            use_raw  = getattr(parent_widget.chk_original, 'isChecked', lambda: False)()

            sigs = []
            if use_proc and entry.get("processed") is not None:
                sigs.append(entry["processed"])
            if use_raw and entry.get("raw") is not None:
                sigs.append(entry["raw"])

            for sig in sigs:
                max_amp = np.max(np.abs(sig))
                if max_amp > max_amplitude:
                    max_amplitude = max_amp

        for item in selected_items:
            name = item.data(0, QtCore.Qt.UserRole)
            if name not in sweep_manager.data:
                continue

            entry = sweep_manager.data[name]
            is_combined = name.startswith("combine")

            use_proc = getattr(parent_widget.chk_processed, 'isChecked', lambda: False)()
            use_raw  = getattr(parent_widget.chk_original, 'isChecked', lambda: False)()

            sig = None
            fs = None
            if use_proc and entry.get("processed") is not None:
                sig = entry["processed"]
                fs = entry["fs"]
            elif use_raw and entry.get("raw") is not None:
                sig = entry["raw"]
                fs = entry["fs_raw"]
            else:
                continue

            if sig is None or fs is None:
                continue

            # Plot & export
            fig = plt.figure(figsize=(8, 4))
            ax = fig.add_subplot(111)
            t = np.arange(len(sig)) / fs
            ax.plot(t, sig, color='black', linewidth=0.5)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.set_xlim(0, 300)
            ax.set_ylim(-max_amplitude, max_amplitude)

            safe_name = re.sub(r'[\\/:"*?<>|]+', '_', name)
            save_path = str(Path(out_dir) / f"{safe_name}.png")

            fig.savefig(save_path, format='png', dpi=3000, transparent=True, bbox_inches='tight')
            plt.close(fig)

            success_list.append(name)

        return f"Exported {len(success_list)} signal PNGs to {out_dir}"



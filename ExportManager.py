import csv
import re, os
import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog

class ExportManager:

    def export_to_pdf(self, figure, parent_widget=None):
        if not figure.axes or not figure.axes[0].has_data():
            return "Cannot export: The plot is empty."
        
        filepath, _ = QFileDialog.getSaveFileName(
            parent_widget, "Save Figure as PDF", "", "PDF Files (*.pdf)"
        )

        if not filepath:
            return "Export cancelled."

        try:
            figure.savefig(filepath, format='pdf', bbox_inches='tight')
            return f"Successfully exported to {filepath}"
        except Exception as e:
            return f"Error exporting to PDF: {e}"

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

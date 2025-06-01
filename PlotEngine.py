# PlotEngine.py

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import spectrogram


class PlotEngine(FigureCanvas):

    def __init__(self, parent=None):
        # Create a Figure with two subplots: one for time‐domain, one for spectrogram
        self.fig = Figure(figsize=(8, 6))
        super().__init__(self.fig)
        # Top subplot: time‐domain signal
        self.ax_signal = self.fig.add_subplot(2, 1, 1)
        # Bottom subplot: spectrogram, sharing x‐axis with ax_signal
        self.ax_spec = self.fig.add_subplot(2, 1, 2, sharex=self.ax_signal)
        self.setParent(parent)

    def clear(self):
        """
        Clear both subplots.
        """
        self.ax_signal.clear()
        self.ax_spec.clear()
        # It’s often helpful to call draw() after clearing, but GUI will call draw() after next plot.

    def plot_processed_signal(self, signal, fs, label="Processed"):
        # In case signal is None or empty, do nothing
        if signal is None or fs is None or len(signal) == 0:
            return

        t = np.arange(len(signal)) / fs
        self.ax_signal.plot(t, signal, label=label, color='black')
        self.ax_signal.set_ylabel("Amplitude")
        self.ax_signal.legend()
        self.ax_signal.set_title("Time‐domain Signal")
        # Do NOT call self.draw() here; let GUI caller invoke draw() after all plotting steps.

    def plot_extra(self, signal_raw, signal_proc, fs, settings):
        # Clear existing content
        self.clear()

        # --- 1. Time‐domain plotting ---
        do_signal_raw = settings.get("draw_raw", False) and settings.get("mode_raw", "") in ["Signal", "Both"]
        do_signal_proc = settings.get("draw_proc", False) and settings.get("mode_proc", "") in ["Signal", "Both"]

        if do_signal_raw and signal_raw is not None:
            t_raw = np.arange(len(signal_raw)) / fs
            self.ax_signal.plot(t_raw, signal_raw, label="Raw", color='black')

        if do_signal_proc and signal_proc is not None:
            t_proc = np.arange(len(signal_proc)) / fs
            self.ax_signal.plot(t_proc, signal_proc, label="Processed", color='blcak')

        # If neither raw nor processed is drawn, leave the signal plot blank
        if do_signal_raw or do_signal_proc:
            self.ax_signal.legend()
            self.ax_signal.set_ylabel("Amplitude")
            self.ax_signal.set_title("Time‐domain Signal")

        # --- 2. Spectrogram plotting ---
        do_spec_raw = settings.get("draw_raw", False) and settings.get("mode_raw", "") in ["Spectrogram", "Both"]
        do_spec_proc = settings.get("draw_proc", False) and settings.get("mode_proc", "") in ["Spectrogram", "Both"]

        # Determine which signal to use for spectrogram: prefer processed if requested
        if do_spec_proc and signal_proc is not None:
            data_for_spec = signal_proc
        elif do_spec_raw and signal_raw is not None:
            data_for_spec = signal_raw
        else:
            data_for_spec = None

        if data_for_spec is not None:
            self._plot_spectrogram(data_for_spec, fs, settings)

        # Label x‐axis on spec plot
        self.ax_spec.set_xlabel("Time (s)")

        # Finally redraw the canvas
        self.draw()

    def _plot_spectrogram(self, data, fs, settings):
        nperseg = settings.get("nperseg", 1024)
        fmax = settings.get("fmax", fs / 2.0)
        log_scale = settings.get("log_scale", False)

        # Compute spectrogram (f: freq bins, t: time bins, Sxx: power spectral density)
        f, t, Sxx = spectrogram(data, fs=fs, nperseg=nperseg, scaling="density", mode="psd")

        # Truncate to f <= fmax
        mask = f <= fmax
        f = f[mask]
        Sxx = Sxx[mask, :]

        # Normalize by global max
        max_val = np.max(Sxx) if Sxx.size > 0 else 1.0
        if max_val <= 0:
            max_val = 1.0
        Sxx_norm = Sxx / max_val

        # Optional log‐scale (10·log10), then re‐normalize to [0,1]
        if log_scale:
            Sxx_db = 10.0 * np.log10(Sxx_norm + 1e-10)
            # Clip negative infinities, then scale to 0–1
            Sxx_db = np.nan_to_num(Sxx_db)  # replace -inf with very small number
            Sxx_norm = (Sxx_db - np.min(Sxx_db)) / (np.max(Sxx_db) - np.min(Sxx_db) + 1e-12)

        # Get custom colormap
        cmap = self._get_custom_colormap()

        # Plot with pcolormesh
        pcm = self.ax_spec.pcolormesh(t, f, Sxx_norm, shading='auto',
                                      cmap=cmap, vmin=0.0, vmax=1.0)

        self.ax_spec.set_ylabel("Frequency (Hz)")
        self.ax_spec.set_title("Spectrogram")
        # Add colorbar on the right of spectrogram
        self.fig.colorbar(pcm, ax=self.ax_spec, orientation='vertical',
                          label="Normalized Power")

    def _get_custom_colormap(self):
        cdict = {
            'red':   [(0.00, 0.00, 0.00),
                      (0.20, 1.00, 1.00),
                      (0.80, 0.00, 0.00),
                      (1.00, 1.00, 1.00)],
            'green': [(0.00, 0.00, 0.00),
                      (0.20, 1.00, 1.00),
                      (0.80, 1.00, 1.00),
                      (1.00, 0.00, 0.00)],
            'blue':  [(0.00, 0.00, 0.00),
                      (0.20, 0.00, 0.00),
                      (0.80, 0.00, 0.00),
                      (1.00, 0.00, 0.00)]
        }
        return LinearSegmentedColormap('CustomMap', cdict)

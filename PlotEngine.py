# PlotEngine.py

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import spectrogram


class PlotEngine(FigureCanvas):

    def __init__(self, *args, **kwargs):
        # Create a Figure with two subplots: one for time‐domain, one for spectrogram
        self.fig = Figure(figsize=(8, 5),constrained_layout=True)

        gs = self.fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 1], hspace=0.0)

        self.ax_signal = self.fig.add_subplot(gs[0, 0])
        self.ax_spec   = self.fig.add_subplot(gs[1, 0], sharex=self.ax_signal)

        super().__init__(self.fig)

    def clear(self):
        """
        Clear both subplots.
        """
        self.fig.clf()
        gs = self.fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 1], hspace=0.0)
        self.ax_signal = self.fig.add_subplot(gs[0, 0])
        self.ax_spec   = self.fig.add_subplot(gs[1, 0], sharex=self.ax_signal)

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

    def _plot_spectrogram(self, data, fs, settings, global_max=None):
        nperseg   = settings["nperseg"]
        fmin      = settings.get("fmin", 0.0)
        fmax      = settings["fmax"]
        log_scale = settings["log_scale"]

        f, t, Sxx = spectrogram(
            data,
            fs=fs,
            nperseg=nperseg,
            scaling="density",
            mode="psd"
        )
        mask = (f >= fmin) & (f <= fmax)
        f    = f[mask]
        Sxx  = Sxx[mask, :]

        if global_max is None or global_max <= 0:
            base = np.max(Sxx) if Sxx.size > 0 else 1.0
        else:
            base = global_max
        Sxx_norm = Sxx / (base + 1e-20)
        Sxx_norm = np.clip(Sxx_norm, 0.0, 1.0)

        if log_scale:
            eps = 1e-12
            Sxx_db = 10.0 * np.log10(Sxx_norm + eps)
            Sxx_db = np.nan_to_num(Sxx_db)
            min_db = np.min(Sxx_db)
            max_db = np.max(Sxx_db)
            if max_db - min_db < 1e-6:
                Sxx_norm = np.zeros_like(Sxx_db)
            else:
                Sxx_norm = (Sxx_db - min_db) / (max_db - min_db)

        pcm = self.ax_spec.pcolormesh(
            t, f, Sxx_norm,
            shading='auto',
            cmap='jet',
            vmin=0.0, vmax=1.0
        )
        self.ax_spec.set_ylabel("Frequency (Hz)")
        self.ax_spec.set_xlabel("Time (s)")
        self.ax_spec.set_title("Spectrogram")
        self.fig.colorbar(pcm, ax=self.ax_spec, orientation='vertical', label="Normalized Power")
        self.ax_spec.set_xlim(0, t[-1])
        self.ax_spec.set_ylim(fmin, f[-1])

        #cache
        self.last_f       = f.copy()
        self.last_t       = t.copy()
        self.last_Sxx_norm = Sxx_norm.copy()       

    def plot_extra(self, signal_raw, signal_proc, fs, settings, global_max=None):
        self.clear()

        do_sig_raw  = settings["draw_raw"]  and settings["mode_raw"]  in ["Signal", "Both"]
        do_sig_proc = settings["draw_proc"] and settings["mode_proc"] in ["Signal", "Both"]

        if do_sig_raw and signal_raw is not None:
            t_raw = np.arange(len(signal_raw)) / fs
            self.ax_signal.plot(t_raw, signal_raw, color='gray', label='Raw')
        if do_sig_proc and signal_proc is not None:
            t_proc = np.arange(len(signal_proc)) / fs
            self.ax_signal.plot(t_proc, signal_proc, color='black', label='Processed')

        if do_sig_raw or do_sig_proc:
            self.ax_signal.set_ylabel("Amplitude")
            self.ax_signal.legend(loc="upper right")

        do_spec_raw  = settings["draw_raw"]  and settings["mode_raw"]  in ["Spectrogram", "Both"]
        do_spec_proc = settings["draw_proc"] and settings["mode_proc"] in ["Spectrogram", "Both"]

        data_for_spec = None
        if do_spec_proc and signal_proc is not None:
            data_for_spec = signal_proc
        elif do_spec_raw and signal_raw is not None:
            data_for_spec = signal_raw

        if data_for_spec is not None:
            self._plot_spectrogram(data_for_spec, fs, settings, global_max)

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
   
    def detect_power_events(self, threshold_mult):
         if not hasattr(self, "last_Sxx_norm"):
             return []

         S = self.last_Sxx_norm

         if S.size == 0:
             return []
         pwr_seq = np.sum(S, axis=0)

         diff_seq = np.diff(pwr_seq)

         sigma = np.std(diff_seq)

         thr_pos = threshold_mult * sigma
         thr_neg = -threshold_mult * sigma
 
         idx_rise = np.where(diff_seq > thr_pos)[0]
         idx_fall = np.where(diff_seq < thr_neg)[0]
 
         t_rise_all = self.last_t[idx_rise + 1]
         t_fall_all = self.last_t[idx_fall + 1]

         events = []
         j_start = 0
         for tr in t_rise_all:
             while j_start < len(t_fall_all) and t_fall_all[j_start] <= tr:
                 j_start +=  1
             if j_start < len(t_fall_all):
                 tf = t_fall_all[j_start]
                 events.append((tr, tf))
                 j_start +=  1
             else:
                 break
 
         return events

    def plot_detection_lines(self, event_pairs):
        for tr, tf in event_pairs:
            self.ax_signal.axvline(tr, color='blue', linestyle='--', linewidth=1)
            self.ax_signal.axvline(tf, color='blue', linestyle='--', linewidth=1)

            self.ax_spec.axvline(tr, color='blue', linestyle='--', linewidth=1)
            self.ax_spec.axvline(tf, color='blue', linestyle='--', linewidth=1)
        self.draw()
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import spectrogram
from hmmlearn import hmm


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
            vmin=0.0, vmax=1.0,
            zorder=0
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
   
    def detect_bursts_hmm(self, signal, fs, settings):

        if signal is None or len(signal) == 0:
            return []

        # Step 1: Calculate Spectrogram
        f, t, Sxx = spectrogram(
            signal, fs=fs, nperseg=settings['nperseg'], scaling="density", mode="psd"
        )
        if Sxx.size == 0 or t.size < 2:
            return []

        # Step 2: Engineer Features for the HMM
        # Feature 1: Log power in the specified frequency band
        freq_mask = (f >= settings['fmin']) & (f <= settings['fmax'])
        power_feature = np.sum(Sxx[freq_mask, :], axis=0)
        log_power = np.log10(power_feature + 1e-20)

        # Feature 2: Delta of log power (change in power)
        delta_log_power = np.diff(log_power, prepend=log_power[0])

        # Combine features into a single array for the HMM
        # Shape: (n_timesteps, n_features)
        feature_matrix = np.column_stack([log_power, delta_log_power])

        # Step 3: Train a 4-state Gaussian HMM
        model = hmm.GaussianHMM(n_components=4, covariance_type="diag", n_iter=100)
        model.fit(feature_matrix)

        # Step 4: Identify the meaning of the 4 hidden states
        # The HMM assigns states randomly (0, 1, 2, 3). We need to map them to our
        # conceptual states (Baseline, Rising, Plateau, Falling) based on their learned means.
        # model.means_ is a (n_components, n_features) array.
        state_means = model.means_
        
        # Identify Baseline state: has the lowest mean log_power (feature 0)
        baseline_state = np.argmin(state_means[:, 0])
        
        # Identify Rising state: has the highest mean delta_log_power (feature 1)
        rising_state = np.argmax(state_means[:, 1])

        # Identify Falling state: has the lowest mean delta_log_power (feature 1)
        falling_state = np.argmin(state_means[:, 1])

        # Identify Plateau state: The remaining state. It should have high power but low delta.
        plateau_state = list(set(range(4)) - {baseline_state, rising_state, falling_state})[0]

        # Step 5: Decode the most likely state sequence
        hidden_states = model.predict(feature_matrix)

        # Step 6: Convert the state sequence to event timestamps
        events = []
        in_event = False
        start_time = 0.0

        for i in range(1, len(hidden_states)):
            prev_state = hidden_states[i-1]
            curr_state = hidden_states[i]

            # An event starts when we transition from Baseline to the Rising phase
            if not in_event and prev_state == baseline_state and curr_state == rising_state:
                in_event = True
                start_time = t[i]
            
            # An event ends when we transition back to the Baseline from any other state
            elif in_event and curr_state == baseline_state:
                in_event = False
                end_time = t[i]
                # Optional: filter out very short, noisy events
                if end_time - start_time > 0.01: # min duration of 10ms
                    events.append((start_time, end_time))

        # Handle the case where the signal ends while still in an event
        if in_event:
            end_time = t[-1]
            if end_time - start_time > 0.01:
                events.append((start_time, end_time))

        return events

    def plot_detection_lines(self, event_pairs):
        for tr, tf in event_pairs:
            self.ax_signal.axvspan(tr, tf, color='blue', alpha=0.5, zorder=0)
            self.ax_spec.axvspan(tr, tf, color='blue', alpha=0.5, zorder=1)
        self.draw()
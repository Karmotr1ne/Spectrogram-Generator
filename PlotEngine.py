import numpy as np
import warnings
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QCursor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import spectrogram
from hmmlearn import hmm

class PlotEngine(FigureCanvas):

    def __init__(self, *args, **kwargs):
        self.fig = Figure(constrained_layout=True)
        super().__init__(self.fig,)
        self.ax_signal = None
        self.ax_spec = None
        self._create_axes()

        self.model = hmm.GaussianHMM(n_components=4, covariance_type="diag", n_iter=100, random_state=42)
        self.is_model_refined = False
        self.spec_data_source = None
        self.last_fs = None
        self.last_settings = None
        self.last_t = np.array([])
        self.last_f = None
        self.last_Sxx = None
        self.segment_map = []
        self.currently_plotted_items = []

        self.editing_enabled = False
        self.burst_patches = []
        
        # Colors for ROI states
        self.ROI_COLOR = 'blue'
        self.HOVER_COLOR = 'red'
        
        # State variables for the new logic
        self.hovered_patch = None  # Stores the patch currently under the mouse
        self.is_adding = False     # True when drawing a new patch
        self.adding_patch = None   # The visual patch being drawn
        self.press_x = None        # Stores x-coord on left-click press
        
        self.press_cid = self.release_cid = self.motion_cid = None
    
    def _get_correct_xdata(self, event):

        ax = event.inaxes
        if ax is None:
            return None

        if event.xdata is not None:
            return event.xdata

        try:
            inv = ax.transData.inverted()

            xdata, _ = inv.transform((event.x, event.y))
            return xdata
        except Exception:
            return None

    def _create_axes(self):
        gs = self.fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 1]) # 不再有 hspace=0.0
        self.ax_signal = self.fig.add_subplot(gs[0, 0])
        self.ax_spec = self.fig.add_subplot(gs[1, 0], sharex=self.ax_signal)
        
    def clear(self):
        self.burst_patches.clear()
        self.segment_map.clear()
        self.currently_plotted_items.clear()
        self.fig.clf()
        self._create_axes()
        self.last_detected_events = []
        self.last_raw_t = np.array([])
        self.last_fs = None

    def plot_extra(self, signal_raw, signal_proc, fs, settings, global_max=None):
        self.last_fs = fs
        
        if self.ax_signal is None: self._create_axes()
        
        if settings.get("draw_raw") and signal_raw is not None:
            self.ax_signal.plot(np.arange(len(signal_raw))/fs, signal_raw, color='blue', label='Raw')
        if settings.get("draw_proc") and signal_proc is not None:
            self.ax_signal.plot(np.arange(len(signal_proc))/fs, signal_proc, color='black', label='Processed')
        
        if self.ax_signal.has_data():
            self.ax_signal.set_ylabel("Amplitude")
            leg = self.ax_signal.legend(loc="upper right", frameon=True)
            if hasattr(self, 'last_raw_t') and len(self.last_raw_t)>1:
                self.ax_signal.set_xlim(0, self.last_raw_t[-1])
            leg.set_zorder(100)

        source_candidate = None
        if settings["mode_proc"] in ["Spectrogram", "Both"] and signal_proc is not None:
            source_candidate = signal_proc
        elif settings["mode_raw"] in ["Spectrogram", "Both"] and signal_raw is not None:
            source_candidate = signal_raw

        if source_candidate is not None:
            self.spec_data_source = source_candidate
            self.last_fs = fs
            self.last_settings = settings
            self._plot_spectrogram(self.spec_data_source, fs, settings, global_max)

        self.fig.canvas.draw()
        self.fig.canvas.draw()

    def _plot_spectrogram(self, data, fs, settings, global_max=None):
        # This method remains largely the same, but sets self.last_t
        nperseg, fmin, fmax, log_scale = settings["nperseg"], settings["fmin"], settings["fmax"], settings["log_scale"]
        f, t, Sxx = spectrogram(data, fs=fs, nperseg=nperseg, scaling="density", mode="psd")
        mask = (f >= fmin) & (f <= fmax)
        f, Sxx = f[mask], Sxx[mask, :]

        # ADD THIS BLOCK to store data for export
        self.last_f = f.copy()
        self.last_t = t.copy()
        self.last_Sxx = Sxx.copy()
        
        if Sxx.size == 0: 
            self.last_t = np.array([])
            return
        
        base = np.max(Sxx) if global_max is None or global_max <= 0 else global_max
        Sxx_norm = np.clip(Sxx / (base + 1e-20), 0.0, 1.0)
        if log_scale:
            eps = 1e-12; Sxx_db = 10.0 * np.log10(Sxx_norm + eps); Sxx_db = np.nan_to_num(Sxx_db)
            min_db, max_db = np.min(Sxx_db), np.max(Sxx_db)
            Sxx_norm = (Sxx_db - min_db) / (max_db - min_db) if (max_db - min_db) > 1e-6 else np.zeros_like(Sxx_db)
            
        pcm = self.ax_spec.pcolormesh(t, f, Sxx_norm, shading='auto', cmap='jet', vmin=0.0, vmax=1.0, zorder=0)
        self.ax_spec.set_ylabel("Frequency (Hz)"); self.ax_spec.set_xlabel("Time (s)")
        self.fig.colorbar(pcm, ax=self.ax_spec, orientation='vertical', label="Normalized Power")
        if hasattr(self, 'last_raw_t') and len(self.last_raw_t) > 1:
            max_time = max(t[-1], self.last_raw_t[-1])
        else:
            max_time = t[-1]
        self.ax_spec.set_xlim(0, max_time)
        self.ax_spec.set_ylim(fmin, f[-1])
        self.last_t = t.copy()

    def plot_sweeps(self, sweeps_info, settings):
        """
        High-level plotting method that handles signal combination and segment map creation.
        """
        self.clear() # Start fresh
        self.currently_plotted_items = [info['item'] for info in sweeps_info]

        combine = settings.get("combine", False)
        fs0 = sweeps_info[0]['fs'] if sweeps_info else 0

        sig_raw_plot, sig_proc_plot = None, None

        if combine:
            current_time_offset = 0.0
            concatenated_signal = []

            # Determine which signal source is primary (Processed > Raw)
            use_proc = settings.get("draw_proc", True)

            for info in sweeps_info:
                signal_to_use = info['signal_proc'] if use_proc and info['signal_proc'] is not None else info['signal_raw']

                if signal_to_use is None:
                    continue

                duration = len(signal_to_use) / info['fs']
                self.segment_map.append({
                    'start_time_combined': current_time_offset,
                    'end_time_combined': current_time_offset + duration,
                    'source_item': info['item']
                })
                concatenated_signal.append(signal_to_use)
                current_time_offset += duration

            if concatenated_signal:
                final_signal = np.concatenate(concatenated_signal)
                # Assign the concatenated signal to the correct plot variable
                if use_proc and any(info['signal_proc'] is not None for info in sweeps_info):
                    sig_proc_plot = final_signal
                else:
                    sig_raw_plot = final_signal

                self.last_raw_t = np.arange(len(final_signal)) / fs0
                self.last_fs    = fs0

        else: # Not combining
            info = sweeps_info[0]
            sig_raw_plot = info['signal_raw'] if settings.get("draw_raw") else None
            sig_proc_plot = info['signal_proc'] if settings.get("draw_proc") else None

        # Call the lower-level plot function with the prepared data
        self.plot_extra(
            signal_raw=sig_raw_plot, signal_proc=sig_proc_plot, fs=fs0, settings=settings
        )

    def _calculate_features(self, signal, fs=None, settings=None):
        fs       = fs or self.last_fs
        settings = settings or self.last_settings
        f, t, Sxx = spectrogram(signal, fs=fs, nperseg=settings['nperseg'], scaling="density", mode="psd")
        Sxx = np.asarray(Sxx)
        f = np.asarray(f)

        if Sxx.size == 0: return None, None
        
        freq_mask = (f >= settings['fmin']) & (f <= settings['fmax'])
        power_feature = np.sum(Sxx[freq_mask, :], axis=0)
        log_power = np.log10(power_feature + 1e-20)
        delta_log_power = np.diff(log_power, prepend=log_power[0])
        return t, np.column_stack([log_power, delta_log_power])

    def learn_and_detect(self):
        t, features = self._calculate_features(self.spec_data_source)
        print("=== DEBUG ROI time ranges and indices ===")
        for i, (sig_patch, _) in enumerate(self.burst_patches):
            bbox = sig_patch.get_extents()
            start_t, end_t = bbox.x0, bbox.x1
            idx_start = np.searchsorted(t, start_t)
            idx_end   = np.searchsorted(t, end_t)
            print(f"ROI #{i+1}: time [{start_t:.3f}, {end_t:.3f}] s → indices [{idx_start}, {idx_end}]")
        print("t range:", t[0], "~", t[-1], "step", t[1]-t[0])
        
        if self.spec_data_source is None: raise ValueError("Please plot a spectrogram before learning.")
        if not self.burst_patches: raise ValueError("No manual regions provided to learn from.")
        
        print("\n--- [DEBUG] Starting learn_and_detect ---")
        
        t, features = self._calculate_features(self.spec_data_source, self.last_fs, self.last_settings)
        if t is None: 
            print("[DEBUG] Feature calculation returned None. Aborting.")
            return []

        print(f"[DEBUG] Total time points in spectrogram: {len(t)}")
        print(f"[DEBUG] Total features shape: {features.shape}")

        precise_bursts_t = []
        # --- Loop through each manually drawn region (ROI) ---
        for i, patch_pair in enumerate(self.burst_patches):
            bbox = patch_pair[0].get_extents()
            idx0 = int(np.clip(round(min(bbox.x0, bbox.x1)), 0, len(self.last_t)-1))
            idx1 = int(np.clip(round(max(bbox.x0, bbox.x1)), 0, len(self.last_t)-1))
            roi_start_t = self.last_t[idx0]
            roi_end_t   = self.last_t[idx1]
            print(f"\n[DEBUG] Processing ROI #{i+1}: Time range = {roi_start_t:.4f}s to {roi_end_t:.4f}s")

            # Find the indices of the feature array that fall within this time range
            roi_indices = np.where((t >= roi_start_t) & (t <= roi_end_t))[0]
            
            print(f"[DEBUG] Number of time points found in this ROI: {len(roi_indices)}")

            if len(roi_indices) < 2: # A single point has zero variance, let's check for this
                print("[DEBUG] WARNING: ROI contains fewer than 2 data points. Skipping this ROI as it cannot be learned from.")
                continue

            # This is the actual data subset for the HMM
            roi_features = features[roi_indices, :]
            roi_time_subset = t[roi_indices]
            
            print(f"[DEBUG] Shape of features for this ROI: {roi_features.shape}")
            
            # Let's check the variance directly before sending to HMM
            # This is the most critical check
            feature_variances = np.var(roi_features, axis=0)
            print(f"[DEBUG] Variance of features for this ROI: {feature_variances}")

            if np.any(feature_variances <= 0):
                print("[DEBUG] ERROR: Found zero or negative variance in features BEFORE HMM.")
                # You can print the problematic features to inspect them
                # print("[DEBUG] Features with zero variance:\n", roi_features)

            
            # Pass the 'warnings' module to the function call
            precise_times = self._find_burst_in_roi(roi_features, roi_time_subset, warnings)
            if precise_times: 
                print(f"[DEBUG] HMM found a burst in this ROI: {precise_times}")
                precise_bursts_t.append(precise_times)
        
        if not precise_bursts_t: 
            raise ValueError("Could not identify a clear burst in any of the provided regions.")
            
        labels = np.zeros(len(t), dtype=int)
        for start_t, end_t in precise_bursts_t:
            start_idx, end_idx = np.searchsorted(t, start_t), np.searchsorted(t, end_t)
            if start_idx >= end_idx: continue
            labels[start_idx] = 1
            if end_idx > start_idx + 1: labels[start_idx+1:end_idx] = 2
            if end_idx < len(labels):
                labels[end_idx] = 3

        self._train_supervised(features, labels)
        predicted_states = self.model.predict(features)

        events, in_event = [], False
        start_time = 0.0
        for i in range(len(predicted_states)):
            if not in_event and predicted_states[i] in [1, 2]:
                in_event, start_time = True, t[i]
            elif in_event and predicted_states[i] == 0:
                in_event = False
                if t[i] > start_time: events.append((start_time, t[i]))
        if in_event: events.append((start_time, t[-1]))

        print("--- [DEBUG] Finished learn_and_detect ---")
        self.last_detected_events = events
        return events
    
    def _train_supervised(self, features, labels):
        n_states = self.model.n_components
        new_means = []
        new_covars = []

        for i in range(n_states):
            state_features = features[labels == i]
            num_samples = state_features.shape[0]

            if num_samples > 1:
                mean = state_features.mean(axis=0)
                var = state_features.var(axis=0) + 1e-6
                new_means.append(mean)
                new_covars.append(var)

            elif num_samples == 1:
                print(f"[DEBUG] INFO: State {i} has 1 sample. Using its features as the mean.")
                mean = state_features[0]
                var = np.ones(features.shape[1]) * 1e-6
                new_means.append(mean)
                new_covars.append(var)

            else: # num_samples == 0
                print(f"[DEBUG] WARNING: State {i} has 0 samples. Using default parameters.")
                mean = np.zeros(features.shape[1])
                var = np.ones(features.shape[1]) * 1e-6
                new_means.append(mean)
                new_covars.append(var)

        self.model.means_ = np.array(new_means)
        self.model.covars_ = np.array(new_covars)
        
        # 1. Count transitions
        transmat = np.zeros((n_states, n_states))
        for i in range(len(labels) - 1): 
            transmat[labels[i], labels[i+1]] += 1
        
        # 2. Normalize rows that have observed transitions
        sum_of_rows = transmat.sum(axis=1, keepdims=True)
        transmat_prob = np.divide(transmat, sum_of_rows, 
                                out=np.zeros_like(transmat, dtype=float), 
                                where=sum_of_rows != 0)
        
        # 3. For states with no outgoing transitions (rows summing to 0),
        #    set their self-transition probability to 1.0.
        no_transition_states = np.where(sum_of_rows.flatten() == 0)[0]
        for state_idx in no_transition_states:
            transmat_prob[state_idx, state_idx] = 1.0
            print(f"[DEBUG] State {state_idx} has no outgoing transitions. Setting self-transition probability to 1.")

        # 4. Manually enforce the transition from 'falling edge' (State 3) back to 'baseline' (State 0).
        if n_states > 3: # Ensure this logic only applies if there are enough states
            transmat_prob[3, :] = 0.0  # Erase any learned probabilities for State 3
            transmat_prob[3, 0] = 1.0  # Set the probability of (State 3 -> State 0) to 100%


        self.model.transmat_ = transmat_prob

        self.model.startprob_ = np.array([1.0, 0.0, 0.0, 0.0])
        self.is_model_refined = True
    
    def _find_burst_in_roi(self, roi_features, roi_t, warnings):
        if len(roi_features) < self.model.n_components: return None
        
        temp_hmm = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=50, random_state=42)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module='hmmlearn')
                temp_hmm.fit(roi_features)
        except ValueError:
            # This can happen if the data in the ROI is not suitable for the model (e.g., all zeros).
            return None

        means = np.asarray(temp_hmm.means_)
        if means.ndim != 2:
            raise TypeError(f"HMM means has unexpected dimension: {means.ndim}. Expected 2.")

        burst_state = np.argmax(means[:, 0])
        states = temp_hmm.predict(roi_features)
        burst_indices = np.where(states == burst_state)[0]
        if len(burst_indices) == 0: return None
        return roi_t[burst_indices[0]], roi_t[burst_indices[-1]]

    def unsupervised_detect(self):
        if self.spec_data_source is None: raise ValueError("Please plot a spectrogram before detecting.")

        t, features = self._calculate_features(self.spec_data_source, self.last_fs, self.last_settings)
        if t is None or len(t) == 0: return []

        if not self.is_model_refined:
            if len(features) < self.model.n_components:
                raise ValueError("Not enough data to train the model. Signal may be too short.")
            self.model.fit(features)
            
            state_means = np.asarray(self.model.means_)
            baseline_state = np.argmin(state_means[:, 0])
            transmat = self.model.transmat_.copy()
            
            print(f"[DEBUG] Unsupervised: Baseline state identified as #{baseline_state}")
            
            for i in range(self.model.n_components):
                if i == baseline_state: continue
                if transmat[i, baseline_state] < 1e-5:
                    if transmat[i, i] > 0.1:
                        donation = min(transmat[i, i] * 0.05, 0.05)
                        transmat[i, i] -= donation
                        transmat[i, baseline_state] += donation
                        print(f"[DEBUG] Forcing escape route for State {i} to baseline state {baseline_state}.")

            self.model.transmat_ = transmat
                
        hidden_states = self.model.predict(features)

        state_means = np.asarray(self.model.means_)
        if state_means.ndim != 2:
            raise TypeError(f"HMM means has unexpected dimension: {state_means.ndim}. Expected 2.")

        baseline_state = np.argmin(state_means[:, 0])

        events, in_event, start_time = [], False, 0.0

        for i in range(1, len(hidden_states)):
            is_baseline_now = (hidden_states[i] == baseline_state)
            was_baseline_before = (hidden_states[i-1] == baseline_state)

            if not in_event and was_baseline_before and not is_baseline_now:
                in_event = True
                # The event truly starts at the time of the LAST baseline point,
                # which is t[i-1].
                start_time = t[i-1] 

            elif in_event and is_baseline_now and not was_baseline_before:
                in_event = False
                # The burst ends at the time of the LAST non-baseline point before
                # returning to baseline, which is also t[i-1].
                end_time = t[i-1]
                if end_time > start_time:
                    events.append((start_time, end_time))


        # If the signal ends while an event is active, close the event at the very last time point.
        if in_event:
            events.append((start_time, t[-1]))
        self.last_detected_events = events
        return events

    def reset_model(self):
        """Resets the HMM to its initial, untrained state."""
        self.model = hmm.GaussianHMM(n_components=4, covariance_type="diag", n_iter=100, random_state=42)
        self.is_model_refined = False

    def set_editing_enabled(self, enabled):
        if self.press_cid: self.fig.canvas.mpl_disconnect(self.press_cid)
        if self.release_cid: self.fig.canvas.mpl_disconnect(self.release_cid)
        if self.motion_cid: self.fig.canvas.mpl_disconnect(self.motion_cid)

        # Reset all state variables and connection IDs for a clean start.
        self.press_cid = self.release_cid = self.motion_cid = None
        self.is_adding = self.adding_patch = self.press_x = self.hovered_patch = None

        # Now, set the new state and connect if enabled.
        self.editing_enabled = enabled
        if enabled:
            # Connect the event handlers fresh.
            self.press_cid = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
            self.release_cid = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
            self.motion_cid = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_motion(self, event):
        if not self.editing_enabled or event.inaxes not in [self.ax_signal, self.ax_spec] or event.xdata is None:
            if self.hovered_patch:
                self.hovered_patch[0].set_color(self.ROI_COLOR)
                self.hovered_patch[1].set_color(self.ROI_COLOR)
                self.hovered_patch = None
                self.fig.canvas.draw()
            return

        xdata = self._get_correct_xdata(event)
        if xdata is None:
            return

        # If we are in the middle of drawing a new region, update its visual representation
        if self.is_adding and self.press_x is not None:
            if self.adding_patch:
                self.adding_patch[0].remove()
                self.adding_patch[1].remove()
            
            patch_sig = self.ax_signal.axvspan(self.press_x, xdata, color='green', alpha=0.3, zorder=5)
            patch_spec = self.ax_spec.axvspan(self.press_x, xdata, color='green', alpha=0.3, zorder=5)
            self.adding_patch = (patch_sig, patch_spec)
            self.fig.canvas.draw()
            return

        # This block handles highlighting existing patches as the mouse moves over them
        found_patch = None
        for patch_pair in self.burst_patches:
            patch = patch_pair[0] if event.inaxes is self.ax_signal else patch_pair[1]
            contains, _ = patch.contains(event)
            if contains:
                found_patch = patch_pair
                break
        
        if found_patch is not self.hovered_patch:
            # Reset the old hovered patch
            if self.hovered_patch:
                self.hovered_patch[0].set_color(self.ROI_COLOR)
                self.hovered_patch[1].set_color(self.ROI_COLOR)
            
            # Highlight the new one
            if found_patch:
                found_patch[0].set_color(self.HOVER_COLOR)
                found_patch[1].set_color(self.HOVER_COLOR)

            self.hovered_patch = found_patch
            self.fig.canvas.draw()

    def on_press(self, event):
        if not self.editing_enabled or event.inaxes not in [self.ax_signal, self.ax_spec] or event.xdata is None:
            return
        
        press_x = self._get_correct_xdata(event)
        if press_x is None:
            return
        
        if event.button == 3 and self.hovered_patch:
            menu = QtWidgets.QMenu(self.parent())
            delete_action = menu.addAction("Delete") 
            merge_action  = menu.addAction("Merge")
            global_pos = QCursor.pos()

            chosen = menu.exec_(global_pos)

            if chosen == delete_action:
                self.remove_patch(self.hovered_patch)
                self.hovered_patch = None

            elif chosen == merge_action:
                container_ext = self.hovered_patch[0].get_extents()
                contained = [
                    p for p in self.burst_patches
                    if p is not self.hovered_patch
                    and p[0].get_extents().x0 >= container_ext.x0
                    and p[0].get_extents().x1 <= container_ext.x1
                ]
                # 如果没有子 ROI，就不做任何事
                if not contained:
                    return
        
                # 2) 用子 ROI 的最早 start / 最晚 end 计算新范围
                starts = [p[0].get_extents().x0 for p in contained]
                ends   = [p[0].get_extents().x1 for p in contained]
                t_start, t_end = min(starts), max(ends)
        
                # 3) 删除所有子 ROI 和那个“临时容器”ROI
                for p in contained:
                    self.remove_patch(p)
                self.remove_patch(self.hovered_patch)
        
                # 4) 画一个新的、跨越这段区间的 ROI
                new_sig = self.ax_signal.axvspan(t_start, t_end,
                                                color=self.ROI_COLOR,
                                                alpha=0.5, zorder=10)
                new_spec = self.ax_spec.axvspan(t_start, t_end,
                                                color=self.ROI_COLOR,
                                                alpha=0.5, zorder=10)
                self.burst_patches.append((new_sig, new_spec))
        
                # 5) 更新用于 CSV 的事件列表
                self.last_detected_events = [
                    (p.get_extents().x0, p.get_extents().x1)
                    for p, _ in self.burst_patches
                ]
        
                # 6) 重置状态并重绘
                self.hovered_patch = None
                self.fig.canvas.draw()
                return

        if event.button == 1:
            self.is_adding = True
            self.press_x = press_x

    def on_release(self, event):
        xdata = self._get_correct_xdata(event)

        # Check for invalid state or if the release happened outside the axes
        if not self.editing_enabled or not self.is_adding or xdata is None:
            if self.adding_patch:
                self.adding_patch[0].remove(); self.adding_patch[1].remove()
            self.is_adding = self.adding_patch = self.press_x = None
            return

        if self.adding_patch:
            self.adding_patch[0].remove()
            self.adding_patch[1].remove()

        start_x, end_x = self.press_x, xdata
        if hasattr(self, 'last_raw_t') and len(self.last_raw_t)>1:
           min_width = self.last_raw_t[1] - self.last_raw_t[0]
        else:
           min_width = 1.0 / self.last_fs if self.last_fs else 0.01

        if abs(start_x - end_x) >= min_width:
            final_sig = self.ax_signal.axvspan(start_x, end_x, color=self.ROI_COLOR, alpha=0.5, zorder=10)
            final_spec = self.ax_spec.axvspan(start_x, end_x, color=self.ROI_COLOR, alpha=0.5, zorder=10)
            self.burst_patches.append((final_sig, final_spec))
            self.last_detected_events.append((min(start_x, end_x), max(start_x, end_x)))
        self.is_adding = self.adding_patch = self.press_x = None
        self.fig.canvas.draw()

    def remove_patch(self, patch_pair):  
        p_sig, p_spec = patch_pair
        p_sig.remove()
        p_spec.remove()
        if patch_pair in self.burst_patches:
            self.burst_patches.remove(patch_pair)
        self.fig.canvas.draw()

    def plot_detection_lines(self, event_pairs):  
        while self.burst_patches: self.remove_patch(self.burst_patches[0])
        for tr, tf in event_pairs:
            patch_sig = self.ax_signal.axvspan(tr, tf, color=self.ROI_COLOR, alpha=0.5, zorder=10)
            patch_spec = self.ax_spec.axvspan(tr, tf, color=self.ROI_COLOR, alpha=0.5, zorder=10)
            self.burst_patches.append((patch_sig, patch_spec))
        self.fig.canvas.draw() 
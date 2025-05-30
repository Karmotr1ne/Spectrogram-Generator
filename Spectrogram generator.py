import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from pyabf import ABF
from pathlib import Path
import h5py
from matplotlib.backends.backend_pdf import PdfPages
from tkinter import filedialog, Tk, messagebox
from tqdm import tqdm

# Signal Loading Functions
def _load_abf_signal(filepath, channel=0, combine_sweeps=True):
    abf = ABF(filepath)
    sampling_rate = abf.dataRate

    if combine_sweeps:
        data = []
        for i in range(abf.sweepCount):
            abf.setSweep(i, channel=channel)
            data.append(abf.sweepY.copy())
        return (np.hstack(data), sampling_rate)
    else:
        data = []
        for i in range(abf.sweepCount):
            abf.setSweep(i, channel=channel)
            data.append(abf.sweepY.copy())
        return (data, sampling_rate)

def _load_neo_h5_signal(filepath, channel=0, combine_sweeps=True):
    """ adapt GUI """
    all_traces = []
    fs = 10000  # default
    with h5py.File(filepath, "r") as f:
        block_keys = [k for k in f["data"] if k.startswith("neo.block")]
        if not block_keys:
            raise ValueError("No neo.block found.")
        block_key = block_keys[0]

        group_path = f"data/{block_key}/groups"
        for seg_key in f[group_path]:
            arrays_path = f"{group_path}/{seg_key}/data_arrays"
            for da_key in f[arrays_path]:
                da_group = f[f"{arrays_path}/{da_key}"]
                if "data" in da_group:
                    arr = da_group["data"][()]  # numpy array
                    
                    # sampling_period
                    if "sampling_period" in da_group.attrs:
                        dt = da_group.attrs["sampling_period"]
                        fs = int(round(1.0 / dt))

                    # 1D / 2D safe channel
                    if arr.ndim == 2:
                        if arr.shape[1] <= channel:
                            raise IndexError(f"Requested channel {channel} exceeds shape {arr.shape}")
                        trace = arr[:, channel]
                    elif arr.ndim == 1:
                        trace = arr
                    else:
                        continue  # skip invalid shape

                    all_traces.append(trace)

    if not all_traces:
        raise ValueError("No valid analog signals found in file.")

    return (np.concatenate(all_traces), fs) if combine_sweeps else (all_traces, fs)

def load_signal_from_file(filepath, channel=0, combine_sweeps=True):
    if filepath.endswith(".abf"):
        return _load_abf_signal(filepath, channel, combine_sweeps)
    elif filepath.endswith(".h5"):
        # Flat saved files from GUI.py
        try:
            return _load_neo_h5_signal(filepath, combine_sweeps)
        except Exception as e:
            raise ValueError(f"Failed to read HDF5: {e}")
    else:
        raise ValueError(f"Unsupported file type: {filepath}")

# Spectrogram Generation
def generate_raw_spectrogram(signal, fs, nperseg=2048, fmax=80):
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg)
    freq_mask = f <= fmax
    return f[freq_mask], t, Sxx[freq_mask, :]

# Plotting Functions
def plot_multi_page_pdf(sweep_data_list, file_stem, output_dir):
    output_path = os.path.join(output_dir, f"{file_stem}_allsweeps.pdf")
    vmin = min(np.min(Sxx) for (f, t, Sxx) in sweep_data_list)
    vmax = max(np.max(Sxx) for (f, t, Sxx) in sweep_data_list)

    with PdfPages(output_path) as pdf:
        for idx, (f, t, Sxx) in enumerate(sweep_data_list):
            plt.figure(figsize=(10, 6))
            plt.pcolormesh(t, f, Sxx, shading='gouraud', cmap='jet', vmin=vmin, vmax=vmax)
            plt.colorbar(label='Power (uV^2/Hz)')
            plt.title(f"{file_stem} - Sweep {idx + 1}")
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")
            plt.tight_layout()
            pdf.savefig()
            plt.close()

# File Processing
def process_all_files(input_dir, output_dir, channel=0, combine_sweeps=True):
    input_files = sorted(Path(input_dir).glob("*.abf")) + sorted(Path(input_dir).glob("*.h5"))

    for file in tqdm(input_files, desc="Processing files"):
        try:
            results, fs = load_signal_from_file(str(file), channel=channel, combine_sweeps=combine_sweeps)

            if combine_sweeps:
                f, t, Sxx = generate_raw_spectrogram(results, fs)
                pdf_path = Path(output_dir) / f"{file.stem}_spectrogram.pdf"
                with PdfPages(pdf_path) as pdf:
                    plt.figure(figsize=(10, 6))
                    plt.pcolormesh(t, f, Sxx, shading='gouraud', cmap='jet', vmin=np.min(Sxx), vmax=np.max(Sxx))
                    plt.colorbar(label='Power (uV^2/Hz)')
                    plt.title(file.stem)
                    plt.xlabel("Time (s)")
                    plt.ylabel("Frequency (Hz)")
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()
            else:
                sweep_results = []
                for signal in results:
                    sweep_results.append(generate_raw_spectrogram(signal, fs))
                plot_multi_page_pdf(sweep_results, file.stem, output_dir)

        except Exception as e:
            print(f"[ERROR] Skipping {file.name}: {e}")

# Entry Point
def main():
    root = Tk()
    root.withdraw()
    input_dir = filedialog.askdirectory(title="Select Input Directory")
    output_dir = filedialog.askdirectory(title="Select Output Directory")

    if not input_dir or not output_dir:
        print("No directory selected. Exiting.")
        return

    combine = messagebox.askyesno("Sweep Combination", "Combine all sweeps before spectrogram generation?")
    process_all_files(input_dir, output_dir, combine_sweeps=combine)

if __name__ == "__main__":
    main()


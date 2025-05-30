import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from pyabf import ABF
from pathlib import Path
import h5py
from h5py import Dataset
from matplotlib.backends.backend_pdf import PdfPages
from tkinter import filedialog, Tk, messagebox
from tqdm import tqdm

# Signal Loading Functions
def _load_abf_signal(filepath, channel=0, combine_sweeps=True):
    abf = ABF(filepath)
    sampling_rate = abf.dataRate
    if combine_sweeps:
        data = np.hstack([abf.setSweep(i, channel=channel) or abf.sweepY for i in range(abf.sweepCount)])
    else:
        data = [abf.setSweep(i, channel=channel) or abf.sweepY.copy() for i in range(abf.sweepCount)]
    return data if not combine_sweeps else (data, sampling_rate)

def _load_h5_signal(filepath, channel=0, default_sampling_rate=10000, combine_sweeps=True):
    all_traces = []
    with h5py.File(filepath, "r") as h5file:
        block_key = next((k for k in h5file["data"] if k.startswith("neo.block")), None)
        groups = h5file[f"data/{block_key}/groups"]
        for seg_key in groups:
            da_path = f"data/{block_key}/groups/{seg_key}/data_arrays"
            for da_key in h5file[da_path]:
                group = h5file[f"{da_path}/{da_key}"]
                if isinstance(group, h5py.Group) and "data" in group:
                    arr = group["data"][()]
                    if arr.ndim == 2 and arr.shape[1] > channel:
                        all_traces.append(arr[:, channel])
                    elif arr.ndim == 1:
                        all_traces.append(arr)

    if not all_traces:
        raise ValueError("No sweep data found")

    return (np.concatenate(all_traces), default_sampling_rate) if combine_sweeps else (all_traces, default_sampling_rate)

def load_signal_from_file(filepath, channel=0, combine_sweeps=True):
    if filepath.endswith(".abf"):
        return _load_abf_signal(filepath, channel, combine_sweeps)
    elif filepath.endswith(".h5"):
        return _load_h5_signal(filepath, channel, combine_sweeps=combine_sweeps)
    else:
        raise ValueError(f"Unsupported file type: {filepath}")

# Spectrogram Generation
def generate_raw_spectrogram(signal, fs, nperseg=2048, fmax=250):
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


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

def _load_abf_signal(filepath, channel=0, combine_sweeps=True):
    abf = ABF(filepath)
    abf.setSweep(0)
    sampling_rate = abf.dataRate
    if combine_sweeps:
        data = np.hstack([abf.setSweep(i, channel=channel) or abf.sweepY for i in range(abf.sweepCount)])
    else:
        abf.setSweep(0, channel=channel)
        data = abf.sweepY
    return data, sampling_rate

def _load_h5_signal(filepath, channel=0, default_sampling_rate=10000, combine_sweeps=True):
    all_traces = []

    with h5py.File(filepath, "r") as h5file:
        data_group = h5file.get("data")
        if data_group is None:
            raise ValueError("Missing 'data' group in HDF5 file.")

        block_key = next((k for k in data_group if k.startswith("neo.block")), None)
        if block_key is None:
            raise ValueError("No neo.block found under 'data'.")

        groups_path = f"data/{block_key}/groups"
        groups = h5file.get(groups_path)
        if groups is None:
            raise ValueError(f"Missing 'groups' path at {groups_path}")

        for seg_key in groups:
            data_array_path = f"{groups_path}/{seg_key}/data_arrays"
            data_arrays = h5file.get(data_array_path)
            if data_arrays is None:
                continue

            for da_key in data_arrays:
                group = data_arrays[da_key]
                if not isinstance(group, h5py.Group):
                    continue
                if "data" not in group:
                    continue

                dataset = group["data"]
                try:
                    data = dataset[()]
                    if data.ndim == 2:
                        all_traces.append(data[:, channel])
                    elif data.ndim == 1:
                        all_traces.append(data)
                except Exception:
                    continue

    if not all_traces:
        raise ValueError(f"No valid signal data found in {filepath}")

    if combine_sweeps:
        signal = np.concatenate(all_traces)
    else:
        return all_traces, default_sampling_rate
    return signal, default_sampling_rate

def load_signal_from_file(filepath, channel=0, combine_sweeps=True):
    if filepath.endswith(".abf"):
        return _load_abf_signal(filepath, channel, combine_sweeps)
    elif filepath.endswith(".h5"):
        return _load_h5_signal(filepath, channel, combine_sweeps=combine_sweeps)
    else:
        raise ValueError(f"Unsupported file type: {filepath}")

def generate_spectrogram(signal, fs, nperseg=2048, fmax=250):
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg)
    freq_mask = f <= fmax
    return f[freq_mask], t, Sxx[freq_mask, :]

def plot_spectrogram_page(f, t, Sxx, title):
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud', cmap='jet')  # 深蓝→深红
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    plt.title(title)
    plt.colorbar(label="Power [dB]")
    plt.tight_layout()
    return plt

def process_all_files(input_dir, output_dir, channel=0, combine_sweeps=True):
    input_files = sorted(Path(input_dir).glob("*.abf")) + sorted(Path(input_dir).glob("*.h5"))

    for file in tqdm(input_files, desc="Processing files"):
        try:
            results, fs = load_signal_from_file(str(file), channel=channel, combine_sweeps=combine_sweeps)

            if combine_sweeps:
                f, t, Sxx = generate_spectrogram(results, fs)
                pdf_path = Path(output_dir) / (file.stem + "_spectrogram.pdf")
                with PdfPages(pdf_path) as pdf:
                    fig = plot_spectrogram_page(f, t, Sxx, title=file.stem)
                    pdf.savefig(fig.gcf())
                    plt.close()
            else:
                pdf_path = Path(output_dir) / (file.stem + "_allsweeps.pdf")
                with PdfPages(pdf_path) as pdf:
                    for idx, signal in enumerate(results):
                        f, t, Sxx = generate_spectrogram(signal, fs)
                        fig = plot_spectrogram_page(f, t, Sxx, title=f"{file.stem} - Sweep {idx + 1}")
                        pdf.savefig(fig.gcf())
                        plt.close()

        except Exception as e:
            print(f"[ERROR] Skipping {file.name}: {e}")

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

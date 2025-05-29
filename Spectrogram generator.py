import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pyabf import ABF
from tkinter import Tk, filedialog
from tqdm import tqdm
from scipy.signal import spectrogram
from matplotlib.backends.backend_pdf import PdfPages


# Get files
def _load_abf_signal(filepath, channel=0):
    abf = ABF(str(filepath))
    all_sweeps = []
    for sweep in range(abf.sweepCount):
        abf.setSweep(sweep, channel=channel)
        all_sweeps.append(abf.sweepY.copy())
    signal = np.concatenate(all_sweeps)
    sampling_rate = abf.dataRate
    return signal, sampling_rate

def _load_h5_signal(filepath, channel=0, default_sampling_rate=10000):
    all_traces = []
    sampling_rate = default_sampling_rate

    with h5py.File(filepath, 'r') as h5file:
        data_root = h5file.get('data')
        if data_root is None:
            raise ValueError("No 'data' group found in HDF5 file.")

        for block_key in data_root:
            if not block_key.startswith("neo.block"):
                continue
            block = data_root.get(f"{block_key}/groups")
            if block is None:
                continue
            for seg_key in block:
                seg_group = block.get(f"{seg_key}/data_arrays")
                if seg_group is None:
                    continue
                for da_key in seg_group:
                    dataset = seg_group.get(da_key)
                    if dataset is None:
                        continue
                    data = dataset[()]
                    if data.ndim == 2:
                        all_traces.append(data[:, channel])
                    elif data.ndim == 1:
                        all_traces.append(data)

    if not all_traces:
        raise ValueError(f"No valid signal data found in {filepath}")

    signal = np.concatenate(all_traces)
    return signal, sampling_rate

def load_signal_from_file(filepath, channel=0, default_sampling_rate=10000):
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix == ".abf":
        return _load_abf_signal(filepath, channel)
    elif suffix == ".h5":
        return _load_h5_signal(filepath, channel, default_sampling_rate)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

# Processing
def process_data_file(filepath, output_dir, channel=0, default_sampling_rate=10000, nperseg=2048):
    filepath = Path(filepath)
    file_stem = filepath.stem

    try:
        signal, fs = load_signal_from_file(filepath, channel=channel, default_sampling_rate=default_sampling_rate)
    except Exception as e:
        print(f"[ERROR] Skipping {filepath.name}: {e}")
        return

    # cal spectrogram
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=2048)
    power_db = 10 * np.log10(Sxx + 1e-12)

    # save PDF
    pdf_path = Path(output_dir) / f"{file_stem}_spectrogram.pdf"
    with PdfPages(pdf_path) as pdf:
        plt.figure(figsize=(12, 5))
        plt.pcolormesh(t, f, power_db, shading='gouraud')
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title(f"{file_stem} Spectrogram")
        plt.colorbar(label="Power (dB)")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # save CSV
    df = pd.DataFrame(power_db, index=f, columns=t)
    df.index.name = "Frequency_Hz"
    csv_path = Path(output_dir) / f"{file_stem}_spectrogram_data.csv"
    df.to_csv(csv_path)

    print(f"[DONE] Processed: {filepath.name}")

# get folder
root = Tk()
root.withdraw()
input_dir = Path(filedialog.askdirectory(title="Select folder containing data files"))
output_dir = Path(filedialog.askdirectory(title="Select folder to save results"))
output_dir.mkdir(exist_ok=True)

# run
for file in tqdm(list(input_dir.glob("*")), desc="Processing files"):
    if file.suffix.lower() in [".abf", ".h5"]:
        process_data_file(file, output_dir)

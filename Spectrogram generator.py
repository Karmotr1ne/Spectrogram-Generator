import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import spectrogram
from pyabf import ABF
from pathlib import Path
from tkinter import Tk, filedialog
import pandas as pd
from tqdm import tqdm

root = Tk()
root.withdraw()

# Get ABF
input_dir = filedialog.askdirectory(title="Select folder")
if not input_dir:
    raise Exception("Cancel selection")
input_dir = Path(input_dir)

output_dir = filedialog.askdirectory(title="Select folder")
if not input_dir:
    raise Exception("Cancel selection")
output_dir = Path(output_dir)

output_dir.mkdir(parents=True, exist_ok=True)

def process_abf_file(file_path, output_dir):
    abf = ABF(file_path)
    sampling_rate = abf.dataRate
    file_stem = Path(file_path).stem

    # Connect all sweeps
    full_signal = []
    for sweep in range(abf.sweepCount):
        abf.setSweep(sweep)
        full_signal.append(abf.sweepY)
    full_signal = np.concatenate(full_signal)

    # Spectrogram
    f, t, Sxx = spectrogram(full_signal, fs=sampling_rate, nperseg=2048)
    power_db = 10 * np.log10(Sxx + 1e-12)

    # PDF
    pdf_path = output_dir / f"{file_stem}_spectrogram.pdf"
    with PdfPages(pdf_path) as pdf:
        plt.figure(figsize=(12, 5))
        plt.pcolormesh(t, f, power_db, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title(f'{file_stem} - Combined Sweep Spectrogram')
        plt.colorbar(label='Power [dB]')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # CSV
    df = pd.DataFrame(power_db, index=f, columns=t)
    df.index.name = "Frequency_Hz"
    csv_path = output_dir / f"{file_stem}_spectrogram_data.csv"
    df.to_csv(csv_path)
   
# run for all
abf_files = list(input_dir.rglob("*.abf"))

# terminal monitor
for abf_path in tqdm(abf_files, desc="Processing"):
    process_abf_file(abf_path, output_dir)

print(f"Path:{output_dir}")

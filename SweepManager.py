import os
import re
import numpy as np
import h5py
import pyabf

class SweepManager:
    def __init__(self):
        self.data = {}

    def load_file(self, filepath: str):
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".abf":
            return self._load_abf(filepath)
        elif ext == ".h5":
            return self._load_h5(filepath)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _load_abf(self, filepath: str):
        abf = pyabf.ABF(filepath)
        if abf.channelCount < 1:
            raise ValueError("Expected at least 1 channel in ABF file.")

        display_names = []
        base = os.path.splitext(os.path.basename(filepath))[0]

        for i in range(abf.sweepCount):
            abf.setSweep(i, channel=0)
            ch0 = abf.sweepY.copy()
            ch1 = None
            if abf.channelCount > 1:
                abf.setSweep(i, channel=1)
                ch1 = abf.sweepY.copy()

            fs = abf.dataRate
            display_name = f"{base}_sweep{i}"

            raw_sig = None
            if ch0 is not None and np.any(np.abs(ch0) > 0):
                raw_sig = ch0
            elif ch1 is not None and np.any(np.abs(ch1) > 0):
                raw_sig = ch1
            else:
                raw_sig = ch0 if ch0 is not None else np.zeros(1)

            self.data[display_name] = {
                "filepath":  filepath,
                "sweep_idx": i,
                "fs":        fs,
                "raw":       raw_sig,
                "processed": None
            }
            display_names.append(display_name)

        return display_names

    def _load_h5(self, filepath: str):
       
        display_names = []

        import os
        from copy import deepcopy
        from neo.io import NixIO
        import re

        try:
            reader = NixIO(filename=filepath, mode='ro')
            block = reader.read_block(lazy=False)
        except Exception as e:
            raise ValueError(f"Failed to open H5 via NixIO: {e}")

        if not hasattr(block, 'segments') or not block.segments:
            return display_names

        base = os.path.splitext(os.path.basename(filepath))[0]

        for i, seg in enumerate(block.segments):
            raw_sig = None
            proc_sig = None

            for sig in seg.analogsignals:

                name = sig.name.decode() if isinstance(sig.name, bytes) else str(sig.name)
                if name.endswith("_raw"):
                    raw_sig = deepcopy(sig)
                elif name.endswith("_processed") or name == "processed":
                    proc_sig = deepcopy(sig)

            if proc_sig is None and seg.analogsignals:
                proc_sig = deepcopy(seg.analogsignals[0])

            if proc_sig is None:
                continue

            if raw_sig is None:
                raw_sig = deepcopy(proc_sig)

            try:
                fs = float(proc_sig.sampling_rate.rescale("Hz").magnitude)
            except Exception:
                fs = None

            display_name = f"{base}_sweep{i}"

            data_raw = raw_sig.magnitude.copy().reshape(-1)
            data_proc = proc_sig.magnitude.copy().reshape(-1)

            self.data[display_name] = {
                "filepath":  filepath,
                "sweep_idx": i,
                "fs":        fs,
                "raw":       data_raw,
                "processed": data_proc
            }

            display_names.append(display_name)

        return display_names

    def get_signal(self, display_name: str, processed: bool = False):
        if display_name not in self.data:
            raise KeyError(f"{display_name} not found in SweepManager.data")

        entry = self.data[display_name]
        if processed:
            sig = entry.get("processed")
            if sig is None:
                sig = entry.get("raw")
                if sig is None:
                    raise KeyError(f"No 'processed' or 'raw' for {display_name}")
            fs = entry.get("fs")
            if fs is None:
                raise KeyError(f"No sampling rate for {display_name}")
            return sig, fs
        else:
            sig = entry.get("raw")
            if sig is None:
                raise KeyError(f"No 'raw' for {display_name}")
            fs = entry.get("fs")
            if fs is None:
                raise KeyError(f"No sampling rate for {display_name}")
            return sig, fs

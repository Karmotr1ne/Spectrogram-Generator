import os
import re
import numpy as np
import h5py
import pyabf
from neo.io import NixIO

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
                "fs_raw":    fs,
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
        import numpy as np

        def to_str(name):
            """Safely converts a bytes or str object to a standard string."""
            if isinstance(name, bytes):
                return name.decode('utf-8', 'ignore')
            return str(name)

        try:
            reader = NixIO(filename=filepath, mode='ro')
            block = reader.read_block(lazy=False)
        except Exception as e:
            raise ValueError(f"Failed to open H5 via NixIO: {e}")

        if not hasattr(block, 'segments') or not block.segments:
            return display_names

        base = os.path.splitext(os.path.basename(filepath))[0]

        for i, seg in enumerate(block.segments):
            raw_sig_neo, proc_sig_neo = None, None
            fs_raw, fs_proc = None, None
            
            # 1. Use the safe 'to_str' helper for flexible signal identification
            potential_proc = [s for s in seg.analogsignals if 'proc' in to_str(s.name).lower()]
            potential_raw = [s for s in seg.analogsignals if 'raw' in to_str(s.name).lower()]

            if potential_proc:
                proc_sig_neo = potential_proc[0]
            if potential_raw:
                raw_sig_neo = potential_raw[0]

            # 2. Smarter fallback logic
            if proc_sig_neo is None and raw_sig_neo is None and seg.analogsignals:
                print(f"INFO: No signal name containing 'raw' or 'proc' found in {base}_sweep{i}. "
                      f"Defaulting to the first available signal.")
                proc_sig_neo = seg.analogsignals[0]

            if proc_sig_neo is None: proc_sig_neo = raw_sig_neo
            if raw_sig_neo is None: raw_sig_neo = proc_sig_neo

            if proc_sig_neo is None:
                continue

            # 3. Use 'to_str' in print statements for robust error reporting
            try:
                fs_proc = float(proc_sig_neo.sampling_rate.rescale("Hz").magnitude)
            except Exception as e:
                print(f"WARNING: Could not extract Fs from processed signal '{to_str(proc_sig_neo.name)}' "
                      f"in {base}_sweep{i}. Error: {e}")

            if raw_sig_neo is not proc_sig_neo:
                try:
                    fs_raw = float(raw_sig_neo.sampling_rate.rescale("Hz").magnitude)
                except Exception as e:
                    print(f"WARNING: Could not extract Fs from raw signal '{to_str(raw_sig_neo.name)}' "
                          f"in {base}_sweep{i}. Error: {e}")
            else:
                fs_raw = fs_proc

            # 4. Determine authoritative Fs and trigger final warning if needed
            authoritative_fs = fs_proc if fs_proc is not None else fs_raw

            if authoritative_fs is None:
                print(f"ERROR: Failed to find any valid sampling rate for {base}_sweep{i}. This sweep will be skipped.")
                continue

            # 5. Store data
            data_raw = raw_sig_neo.magnitude.copy().reshape(-1)
            data_proc = proc_sig_neo.magnitude.copy().reshape(-1)
            display_name = f"{base}_sweep{i}"
            
            self.data[display_name] = {
                "filepath":  filepath,
                "sweep_idx": i,
                "fs":        authoritative_fs,
                "fs_raw":    fs_raw,
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
                # Fallback: if no processed signal, use raw signal instead
                sig = entry.get("raw")
                if sig is None:
                    raise KeyError(f"No 'processed' or 'raw' signal for {display_name}")
                # Use the raw fs since we are falling back to the raw signal
                fs = entry.get("fs_raw", entry.get("fs"))
            else:
                # Standard case: return processed signal with its fs
                fs = entry.get("fs")

            if fs is None:
                raise KeyError(f"No sampling rate for processed signal of {display_name}")
            return sig, fs
        
        else: # Requesting the raw signal
            sig = entry.get("raw")
            if sig is None:
                raise KeyError(f"No 'raw' signal for {display_name}")
            
            # Key Fix: Return the raw signal with its specific sampling rate 'fs_raw'.
            # Fall back to the main 'fs' if 'fs_raw' doesn't exist (for older file types like abf).
            fs = entry.get("fs_raw", entry.get("fs"))
            
            if fs is None:
                raise KeyError(f"No sampling rate for raw signal of {display_name}")
            return sig, fs

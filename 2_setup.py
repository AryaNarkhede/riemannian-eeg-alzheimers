import pandas as pd
import numpy as np
import mne
from mne_bids import BIDSPath, read_raw_bids

def ext_eeg(dp):
    df = pd.read_csv(f"{dp}/participants.tsv", sep='\t')
    ch = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
    xl = []
    yl = []
    gl = []

    for _, r in df.iterrows():
        s = r['participant_id'].replace('sub-', '')
        g = r['Group']
        if g not in ['A', 'C']:
            continue
        
        print(f"P: {s}")
        bp = BIDSPath(subject=s, task='eyesclosed', datatype='eeg', root=dp)
        
        try:
            raw = read_raw_bids(bids_path=bp, verbose=False)
            raw.load_data(verbose=False)
            raw.pick(ch)
            raw.filter(l_freq=0.5, h_freq=45.0, verbose=False)
            ev = mne.make_fixed_length_events(raw, duration=4.0)
            ep = mne.Epochs(raw, ev, tmin=0, tmax=4.0, baseline=None, preload=True, verbose=False)
            
            xd = ep.get_data()
            xl.append(xd)
            yl.extend([1 if g == 'A' else 0] * len(xd))
            gl.extend([s] * len(xd))
        except Exception as e:
            print(f"E {s}: {e}")

    if not xl:
        return np.array([]), np.array([]), np.array([])
    return np.vstack(xl), np.array(yl), np.array(gl)


# Assuming your function is named ext_eeg
X_data, y_labels, groups = ext_eeg('data/ds004504')

np.save('X_data.npy', X_data)
np.save('y_labels.npy', y_labels)
np.save('groups.npy', groups)

import numpy as np
from mne.time_frequency import psd_array_welch
from mne.filter import filter_data
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.preprocessing import Normalizer

xd = np.load('X_data.npy').astype(np.float64)
sf = 200.0

xd = xd / (np.std(xd, axis=(1, 2), keepdims=True) + 1e-12)

p, f = psd_array_welch(xd, sf, fmin=0.5, fmax=45.0, n_fft=256, verbose=False)
xp = p.reshape(p.shape[0], -1)
xp = Normalizer().fit_transform(np.log10(np.maximum(xp, 1e-12)))

xc = Covariances(estimator='lwf').fit_transform(xd)

bnds = [(0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)]
xt_list = []

for l_f, h_f in bnds:
    xb = filter_data(xd, sfreq=sf, l_freq=l_f, h_freq=h_f, verbose=False, n_jobs=-1)
    cb = Covariances(estimator='lwf').fit_transform(xb)
    tb = TangentSpace(metric='riemann').fit_transform(cb)
    xt_list.append(tb)

xt = np.hstack(xt_list)

np.save('X_psd.npy', xp)
np.save('X_cov.npy', xc)
np.save('X_ts.npy', xt)

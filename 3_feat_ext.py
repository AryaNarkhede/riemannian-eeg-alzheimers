import numpy as np
from mne.time_frequency import psd_array_welch
from pyriemann.estimation import Covariances

xd = np.load('X_data.npy')
sf = 200.0

p, f = psd_array_welch(xd, sf, fmin=0.5, fmax=45.0, n_fft=256, verbose=False)
xp = p.reshape(p.shape[0], -1)

ce = Covariances(estimator='lwf')
xc = ce.fit_transform(xd)

np.save('X_psd.npy', xp)
np.save('X_cov.npy', xc)

from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt
import numpy as np
from torchvision import transforms


class FastICATransform:
    def __init__(self, n_components=None, random_state=0):
        self.fast_ica = FastICA(n_components=n_components, random_state=random_state)
        self.mixing_matrix = None

    def __call__(self, X):
        import copy
        if len(X.shape) == 3:
            N, S, C = X.shape
            X_ICA = copy.deepcopy(X)
            X_ICA = X_ICA.reshape(-1, C)
            X_ICA = self.fast_ica.fit_transform(X_ICA)
            self.mixing_matrix = self.fast_ica.mixing_
            return np.stack([X.reshape(N, S, C), X_ICA.reshape(N, S, C)], axis=0)
        else:
            X_ICA = copy.deepcopy(X)
            X_ICA = self.fast_ica.fit_transform(X_ICA)
            self.mixing_matrix = self.fast_ica.mixing_
            return np.stack([X, X_ICA], axis=0)

class StandardScalerTransform:
    def __init__(self):
        self.scaler = StandardScaler()

    def __call__(self, X):
        if len(X.shape) == 3:
            N, S, C = X.shape
            X = X.reshape(-1, C)
            X = self.scaler.fit_transform(X)
            return X.reshape(N, S, C)
        else:
            return self.scaler.fit_transform(X)

class ReshapeTransform:
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, X):
        return X.reshape(-1, self.shape)
    
class FilterTransform:
    def __init__(self, fs, notch_freq=50, lowcut=10, highcut=450, Q=30):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.Q = Q
        self.notch_freq = notch_freq
    
    def __call__(self, X):
        return self._filter_data(X)
    
    def _filter_data(self, X):
        
        if len(X.shape) == 3:
            N, S, C = X.shape
            X = X.reshape(-1, C)
            return self._filter_data(X)

        # Calculate the normalized frequency and design the notch filter
        w0 = self.notch_freq / (self.fs / 2)
        b_notch, a_notch = iirnotch(w0, self.Q)

        #calculate the normalized frequencies and design the highpass filter
        cutoff = self.lowcut / (self.fs / 2)
        sos = butter(5, cutoff, btype='highpass', output='sos')

        # apply filters using 'filtfilt' to avoid phase shift
        data = sosfiltfilt(sos, data, axis=0, padtype='even')
        data = filtfilt(b_notch, a_notch, data)

        return data

def make_transform(cfg):
    transform = []
    if cfg.DATA.NORMALIZE:
        transform.append(StandardScalerTransform())
    if cfg.DATA.FILTER_DATA:
        transform.append(FilterTransform(fs=cfg.DATA.EMG.SAMPLING_RATE,
                                         notch_freq=cfg.DATA.EMG.NOTCH_FREQ,
                                         lowcut=cfg.DATA.EMG.LOW_FREQ,
                                         highcut=cfg.DATA.EMG.HIGH_FREQ,
                                         Q=cfg.DATA.EMG.Q))
    if cfg.DATA.ICA:
        transform.append(FastICATransform(n_components=cfg.DATA.EMG.NUM_CHANNELS))

    if len(transform) == 0:
        return None
    else:
        return transforms.Compose(transform)
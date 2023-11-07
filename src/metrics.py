from pesq import pesq_batch
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
import librosa
import numpy as np

def calculate_pesq(F_ref, F_deg, fs=16000):
    # TODO: remove hardcoded values
    ref = np.zeros((F_ref.shape[0], 160000))
    deg = np.zeros((F_deg.shape[0], 160000))
    for i in range(F_ref.shape[0]):
        ref[i] = librosa.istft(F_ref[i].detach().cpu().numpy().T, n_fft=512, hop_length=256)
        deg[i] = librosa.istft(F_deg[i].detach().cpu().numpy().T, n_fft=512, hop_length=256)
    return np.mean(pesq_batch(fs, ref, deg, 'wb'))

def calculate_sisdr(F_ref, F_deg):
    si_sdr = ScaleInvariantSignalDistortionRatio()
    return si_sdr(F_deg, F_ref)

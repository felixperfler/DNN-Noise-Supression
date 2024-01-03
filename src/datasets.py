import os
import random
import torch
from torch.utils.data import Dataset
import numpy as np
import soundfile
from scipy import signal
import librosa

from src.audio_helpers import segmental_snr_mixer


# set seed
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)
np.random.seed(0)


class DNSChallangeDataset(Dataset):
    def __init__(self,
                 datapath:str,
                 datapath_clean_speach:str,
                 sig_length:int=10,
                 fs:int=16000):

        self.sig_length = sig_length # in seconds
        self.fs = fs

        # get all clean speech signals
        self.clean_speech_signals = []
        for root, _, files in os.walk(f"{datapath_clean_speach}"):
            for file in files:
                if file.endswith(".flac"):
                    self.clean_speech_signals.append(os.path.join(root, file))

        self.clean_speech_signals = self.clean_speech_signals[:10]

        # scrmabel the list
        np.random.shuffle(self.clean_speech_signals)
        
        # get all noise signals
        noise_signals = []
        for root, _, files in os.walk(f"{datapath}/noise"):
            for file in files:
                if file.endswith(".wav"):
                    noise_signals.append(os.path.join(root, file))

        self.noise_signals = noise_signals

        # get all impulse responses
        impulse_responses = []
        for root, _, files in os.walk(f"{datapath}/impulse_responses"):
            for file in files:
                if file.endswith(".wav"):
                    impulse_responses.append(os.path.join(root, file))

        self.impulse_responses = impulse_responses

    def __len__(self):
        return len(self.clean_speech_signals)

    def __getitem__(self, idx):
        clean_speech, fs_signal = soundfile.read(self.clean_speech_signals[idx])
        noise, fs_noise = soundfile.read(random.choice(self.noise_signals))
        rir, fs_ir = soundfile.read(random.choice(self.impulse_responses))

        if fs_signal != self.fs:
            clean_speech = librosa.resample(clean_speech, orig_sr=fs_signal, target_sr=self.fs)
        if fs_noise != self.fs:
            noise = librosa.resample(noise, orig_sr=fs_noise, target_sr=self.fs)
        if fs_ir != self.fs:
            rir = librosa.resample(rir, orig_sr=fs_ir, target_sr=self.fs)

        if len(clean_speech.shape) > 1:
            clean_speech = clean_speech[:,0]
        if len(noise.shape) > 1:
            noise = noise[:,0]
        if len(rir.shape) > 1:
            rir = rir[:,0]

        while len(clean_speech) < self.sig_length*self.fs:
            clean_speech = np.append(clean_speech, clean_speech)

        while len(noise) < self.sig_length*self.fs:
            noise = np.append(noise, noise)

        params = {'target_level_lower':-35,'target_level_upper':-15, 'snr':np.random.uniform(-6, 9)}
        # add reverb
        reverb_speech = signal.fftconvolve(clean_speech, rir, mode="full")
        reverb_speech = reverb_speech[0 : clean_speech.shape[0]]

        clean_snr, _, noisy_snr, _ = segmental_snr_mixer(params=params,
                                                            clean=reverb_speech,
                                                            noise=noise,
                                                            snr=params['snr'],)

        clean_snr = clean_snr[:self.sig_length*self.fs]
        noisy_snr = noisy_snr[:self.sig_length*self.fs]

        return {
            'target_signal':clean_snr.astype(np.float32),
            'noisy_signal':noisy_snr.astype(np.float32),
        }

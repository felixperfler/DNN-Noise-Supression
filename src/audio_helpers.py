# from https://github.com/microsoft/DNS-Challenge/blob/master/audiolib.py
import numpy as np

EPS = np.finfo(float).eps

def normalize_segmental_rms(audio, rms, target_level=-25):
    '''Normalize the signal to the target level
    based on segmental RMS'''
    scalar = 10 ** (target_level / 20) / (rms+EPS)
    audio = audio * scalar
    return audio

def is_clipped(audio, clipping_threshold=0.99):
    return any(abs(audio) > clipping_threshold)

def active_rms(clean, noise, fs=16000, energy_thresh=-50):
    '''Returns the clean and noise RMS of the noise calculated only in the active portions'''
    window_size = 100 # in ms
    window_samples = int(fs*window_size/1000)
    sample_start = 0
    noise_active_segs = []
    clean_active_segs = []

    while sample_start < len(noise):
        sample_end = min(sample_start + window_samples, len(noise))
        noise_win = noise[sample_start:sample_end]
        clean_win = clean[sample_start:sample_end]
        noise_seg_rms = (noise_win**2).mean()**0.5
        # Considering frames with energy
        if noise_seg_rms > energy_thresh:
            noise_active_segs = np.append(noise_active_segs, noise_win)
            clean_active_segs = np.append(clean_active_segs, clean_win)
        sample_start += window_samples

    if len(noise_active_segs)!=0:
        noise_rms = (noise_active_segs**2).mean()**0.5
    else:
        noise_rms = EPS

    if len(clean_active_segs)!=0:
        clean_rms = (clean_active_segs**2).mean()**0.5
    else:
        clean_rms = EPS

    return clean_rms, noise_rms

def segmental_snr_mixer(params, clean, noise, snr, target_level=-25, clipping_threshold=0.99):
    '''Function to mix clean speech and noise at various segmental SNR levels'''
    if len(clean) > len(noise):
        noise = np.append(noise, np.zeros(len(clean)-len(noise)))
    else:
        clean = np.append(clean, np.zeros(len(noise)-len(clean)))
    clean = clean/(max(abs(clean))+EPS)
    noise = noise/(max(abs(noise))+EPS)
    rmsclean, rmsnoise = active_rms(clean=clean, noise=noise)
    clean = normalize_segmental_rms(clean, rms=rmsclean, target_level=target_level)
    noise = normalize_segmental_rms(noise, rms=rmsnoise, target_level=target_level)
    # Set the noise level for a given SNR
    noisescalar = rmsclean / (10**(snr/20)) / (rmsnoise+EPS)
    noisenewlevel = noise * noisescalar

    # Mix noise and clean speech
    noisyspeech = clean + noisenewlevel
    # Randomly select RMS value between -15 dBFS and -35 dBFS and normalize noisyspeech with that value
    # There is a chance of clipping that might happen with very less probability, which is not a major issue.
    noisy_rms_level = np.random.randint(params['target_level_lower'], params['target_level_upper'])
    rmsnoisy = (noisyspeech**2).mean()**0.5
    scalarnoisy = 10 ** (noisy_rms_level / 20) / (rmsnoisy+EPS)
    noisyspeech = noisyspeech * scalarnoisy
    clean = clean * scalarnoisy
    noisenewlevel = noisenewlevel * scalarnoisy
    # Final check to see if there are any amplitudes exceeding +/- 1. If so, normalize all the signals accordingly
    if is_clipped(noisyspeech):
        noisyspeech_maxamplevel = max(abs(noisyspeech))/(clipping_threshold-EPS)
        noisyspeech = noisyspeech/noisyspeech_maxamplevel
        clean = clean/noisyspeech_maxamplevel
        noisenewlevel = noisenewlevel/noisyspeech_maxamplevel
        noisy_rms_level = int(20*np.log10(scalarnoisy/noisyspeech_maxamplevel*(rmsnoisy+EPS)))

    return clean, noisenewlevel, noisyspeech, noisy_rms_level

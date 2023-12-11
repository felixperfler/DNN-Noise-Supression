import io
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToTensor
import PIL.Image

def gen_plots(noisy_signal, target_signal, output_signal, encoder_filterbank, epoch, fs, type):
    """Create a pyplot plot and save to buffer."""

    t = np.arange(0,len(noisy_signal)) / fs

    plt.figure()
    plt.plot(t, noisy_signal, label='noisy signal')
    plt.plot(t, target_signal, label='target signal')
    plt.plot(t, output_signal, label='enhanced signal')
    plt.title(f'Signals in the time domain at epoch {epoch} during {type}')
    plt.xlabel('Time in [s]')
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)

    image = PIL.Image.open(buf)
    image_signals = ToTensor()(image)

    plt.figure()
    plt.imshow(encoder_filterbank, origin='lower')
    plt.title(f'Encoder Filterbank at epoch {epoch} during {type}')

    buf2 = io.BytesIO()
    plt.savefig(buf2, format='jpeg')
    buf2.seek(0)

    image_filterbank = PIL.Image.open(buf2)
    image_filterbank = ToTensor()(image_filterbank)

    return image_signals, image_filterbank
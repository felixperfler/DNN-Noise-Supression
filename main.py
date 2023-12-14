import os
import datetime
import random
import torch
from torch.utils.data import DataLoader
from torchaudio.transforms import Spectrogram
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
from pesq import pesq_batch
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

from src.datasets import DNSChallangeDataset
from conv_tasnet import TasNet
from src.losses import ComplexCompressedMSELoss
from src.fb_utils import fir_tightener3000

# set seed
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)
np.random.seed(0)

EPOCHS = 300
VAL_EVERY = 3
KAPPA_BETA = 1e-3
# KAPPA_BETA = None
BATCH_SIZE = 16
NUM_WORKERS = 4
MODEL_FILE = None
FS = 16000
LOGGING_DIR = f"{os.getcwd()}/runs_kappa/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}" if KAPPA_BETA != None else\
        f"{os.getcwd()}/runs/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
DATASET = "/users/felix.perfler/LibriSpeech"
USE_FIR_TIGTHENER3000 = True
SIGNAL_LENGTH = 5

def calculate_condition_number(w):
    w_hat = torch.sum(torch.abs(torch.fft.fft(w,dim=1))**2,dim=0)
    B = torch.max(w_hat,dim=0).values
    A = torch.min(w_hat,dim=0).values

    return  B/A

def main():

    device = torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu")
    print(f"Using device: {device}")

    model = TasNet(
        enc_dim=256,
        kernel=3,
        feature_dim=128,
        sr=FS,
        win=2,
        layer=8,
        stack=4,
        num_spk=2,
        causal=False
    )

    if MODEL_FILE != None:
        checkpoint = torch.load(MODEL_FILE)
        # TODO: check if device is correct
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        epoch = checkpoint['epoch']
    else:
        epoch = 0

    if USE_FIR_TIGTHENER3000:
        # get encoder weights
        encoder_filterbank = model.encoder.weight.squeeze(1).cpu().detach().numpy()
        # pad encoder weights to signal length at the end
        encoder_filterbank = np.pad(encoder_filterbank, ((0,0),(0,SIGNAL_LENGTH*FS-encoder_filterbank.shape[1])), 'constant', constant_values=0)
        # get tightener weights
        tightener_filterbank = fir_tightener3000(encoder_filterbank, model.win, eps=1.02)
        tightener_filterbank = torch.tensor(tightener_filterbank[:,:model.win], dtype=torch.float32).unsqueeze(1)
        # set encoder weights to tightener weights
        model.encoder.weight = torch.nn.Parameter(tightener_filterbank)

    model.to(device)

    print("#params of model: ", sum(p.numel() for p in model.parameters()))

    if KAPPA_BETA == None:
        loss_fn = ComplexCompressedMSELoss()
    else:
        loss_fn = ComplexCompressedMSELoss(beta=KAPPA_BETA)
        

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    
    if MODEL_FILE != None:
        checkpoint = torch.load(MODEL_FILE)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    dataset_train = DNSChallangeDataset(datapath=DATASET,
                                        datapath_clean_speach=DATASET + "/train-clean-360",
                                        fs=FS,
                                        sig_length=SIGNAL_LENGTH)
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    dataset_val = DNSChallangeDataset(datapath=DATASET,
                                        datapath_clean_speach=DATASET + "/test-clean",
                                        fs=FS,
                                        sig_length=SIGNAL_LENGTH)
    dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # init tensorboard
    writer = SummaryWriter(f"{LOGGING_DIR}")

    specgram = Spectrogram(n_fft=512, win_length=512, hop_length=128, power=None).to(device)

    while epoch < EPOCHS:
        running_loss = 0
        model.train()
        for batch in tqdm(dataloader_train):
            noisy_signal = batch['noisy_signal'].to(device)
            target_signal = batch['target_signal'].to(device)

            optimizer.zero_grad()

            output_signal = model(noisy_signal)[:,0,:]
            
            output_signal_fft = specgram(output_signal)
            target_signal_fft = specgram(target_signal)
            if KAPPA_BETA != None:
                # get encoder weights for optimization
                encoder_filterbank = model.encoder.weight.squeeze(1)
                base_loss, loss = loss_fn(output_signal_fft, target_signal_fft, encoder_filterbank)
                running_loss += base_loss.item()
            else:
                loss = loss_fn(output_signal_fft, target_signal_fft)
                running_loss += loss.item()
            
            loss.backward()

            optimizer.step()


        if epoch % VAL_EVERY == 0:
            running_val_loss = 0
            pesq = 0
            sisdr = 0
            model.eval()
            with torch.no_grad():
                for batch in tqdm(dataloader_val):
                    noisy_signal = batch['noisy_signal'].to(device)
                    target_signal = batch['target_signal'].to(device)

                    output_signal = model(noisy_signal)[:,0,:]

                    output_signal_fft = specgram(output_signal)
                    target_signal_fft = specgram(target_signal)

                    loss = loss_fn(output_signal_fft, target_signal_fft)
                    running_val_loss += loss.item()

                    pesq += np.mean(pesq_batch(FS, np.array(target_signal.cpu().detach().numpy()), np.array(output_signal.cpu().detach().numpy()), 'wb'))
                    sisdr += ScaleInvariantSignalDistortionRatio()(torch.abs(torch.fft.rfft(output_signal.cpu().detach())), torch.abs(torch.fft.rfft(target_signal.cpu().detach())))

            
            writer.add_audio('Prediction Val', output_signal[0], epoch, sample_rate=FS)
            writer.add_audio('Target Val', target_signal[0], epoch, sample_rate=FS)
            writer.add_audio('Noisy Val', noisy_signal[0], epoch, sample_rate=FS)

            writer.add_scalar('PESQ Val', pesq / len(dataloader_val), epoch)
            writer.add_scalar('SISDR Val', sisdr / len(dataloader_val), epoch)

            writer.add_scalars('Loss', {'Train': running_loss / len(dataloader_train),
                                        'Val': running_val_loss / len(dataloader_val)}, epoch)
            writer.add_scalar('Condition Number', calculate_condition_number(model.encoder.weight.squeeze(1)), epoch)
        else:
            writer.add_scalars('Loss', {'Train': running_loss / len(dataloader_train),}, epoch)
            writer.add_scalar('Condition Number', calculate_condition_number(model.encoder.weight.squeeze(1)), epoch)

        # check if model save dir exists
        if not os.path.exists(f"{LOGGING_DIR}/models"):
            os.makedirs(f"{LOGGING_DIR}/models")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, f"{LOGGING_DIR}/models/model_{epoch}.pth")
        
        epoch += 1

if __name__ == "__main__":
    main()

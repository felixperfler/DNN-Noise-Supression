import os
import datetime
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import librosa
import numpy as np
from pesq import pesq_batch
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

from src.datasets import DNSChallangeDataset
from conv_tasnet import TasNet
from src.losses import ComplexCompressedMSELoss

# set seed
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)
np.random.seed(0)

EPOCHS = 300
VAL_EVERY = 3
KAPPA_BETA = 0.1
BATCH_SIZE = 8
NUM_WORKERS = 2
MODEL_FILE = None
FS = 16000

def main():

    device = torch.device("mps")  if torch.backends.mps.is_available() \
            else torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu")
    
    # if KAPPA_BETA is not None and DEVICE is mps switch to cpu (because of fft)
    if KAPPA_BETA != None and device == torch.device("mps"):
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model = TasNet(
        enc_dim=128,
        kernel=3,
        feature_dim=64,
        sr=FS,
        win=2,
        layer=8,
        stack=3,
        num_spk=2,
        causal=False
    )

    if MODEL_FILE != None:
        checkpoint = torch.load(MODEL_FILE)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
    else:
        epoch = 0
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

    dataset_train = DNSChallangeDataset(datapath=f"{os.getcwd()}/datasets",
                                    split="train", fs=FS)
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    dataset_val = DNSChallangeDataset(datapath=f"{os.getcwd()}/datasets",
                                    split="val", fs=FS)
    dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # init tensorboard
    writer = SummaryWriter(f"{os.getcwd()}/runs/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")

    while epoch < EPOCHS:
        running_loss = 0
        model.train()
        for batch in tqdm(dataloader_train):
            noisy_signal = batch['noisy_signal'].to(device)
            target_signal = batch['target_signal'].to(device)

            optimizer.zero_grad()

            output_signal = model(noisy_signal)[:,0,:]


            # import matplotlib.pyplot as plt 

            # t = np.arange(0,noisy_signal.cpu().detach().numpy().shape[1]) / FS

            # plt.figure()
            # plt.plot(t, noisy_signal.cpu().detach().numpy()[0], label='noisy signal')
            # plt.plot(t, target_signal.cpu().detach().numpy()[0], label='target signal')
            # plt.plot(t, output_signal.cpu().detach().numpy()[0], label='enhanced signal')
            # plt.title(f'Signals in the time domain at epoch {epoch}')
            # plt.xlabel('Time in [s]')
            # plt.legend()

            # plt.figure()
            # plt.imshow(model.encoder.weight.cpu().detach().numpy().squeeze(1), origin='lower')
            # plt.title(f'Encoder Filterbank at epoch {epoch}')

            # plt.show()

            output_signal_fft = torch.fft.rfft(output_signal)
            target_signal_fft = torch.fft.rfft(target_signal)

            if KAPPA_BETA != None:

                # get encoder weights for optimization
                encoder_filterbank = model.encoder.weight.squeeze(1)
                base_loss, loss = loss_fn(output_signal_fft, target_signal_fft, encoder_filterbank)
            else:
                loss = loss_fn(output_signal_fft, target_signal_fft)
            
            running_loss += loss.item()
            loss.backward()

            optimizer.step()

        # prediction_sample_time_domain = librosa.istft(output_signal[0].detach().cpu().numpy().T, n_fft=512, hop_length=256)
        # writer.add_audio('Prediction Train', prediction_sample_time_domain, epoch, sample_rate=16000)
        # target_sample_time_domain = librosa.istft(target_signal[0].detach().cpu().numpy().T, n_fft=512, hop_length=256)
        # writer.add_audio('Target Train', target_sample_time_domain, epoch, sample_rate=16000)
        # noisy_sample_time_domain = librosa.istft(noisy_signal[0].detach().cpu().numpy().T, n_fft=512, hop_length=256)
        # writer.add_audio('Noisy Train', noisy_sample_time_domain, epoch, sample_rate=16000)

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

                    output_signal_fft = torch.fft.rfft(output_signal)
                    target_signal_fft = torch.fft.rfft(target_signal)

                    loss = loss_fn(output_signal_fft, target_signal_fft)
                    running_val_loss += loss.item()

                    pesq += np.mean(pesq_batch(FS, np.array(target_signal), np.array(output_signal), 'wb'))
                    sisdr += ScaleInvariantSignalDistortionRatio()(torch.abs(torch.fft.rfft(output_signal)), torch.abs(torch.fft.rfft(target_signal)))

            writer.add_audio('Prediction Val', output_signal[0], epoch, sample_rate=FS)
            writer.add_audio('Target Val', target_signal[0], epoch, sample_rate=FS)
            writer.add_audio('Noisy Val', noisy_signal[0], epoch, sample_rate=FS)

            writer.add_scalar('PESQ Val', pesq / len(dataloader_val), epoch)
            writer.add_scalar('SISDR Val', sisdr / len(dataloader_val), epoch)

            writer.add_scalars('Loss', {'Train': running_loss / len(dataloader_train),
                                        'Val': running_val_loss / len(dataloader_val)}, epoch)
        else:
            writer.add_scalars('Loss', {'Train': running_loss / len(dataloader_train),}, epoch)

        # if f"{os.getcwd()}/runs/models" not in os.listdir():
        #     os.mkdir(f"{os.getcwd()}/runs/models")

        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     }, f"{os.getcwd()}/runs/models/model_{epoch}.pth")
        
        epoch += 1

if __name__ == "__main__":
    main()

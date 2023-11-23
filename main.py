from cProfile import label
import os
import datetime
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import librosa
import numpy as np

from src.datasets import DNSChallangeDataset
from conv_tasnet import TasNet
from src.losses import ComplexCompressedMSELoss
from src.metrics import calculate_pesq, calculate_sisdr

# set seed
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)
np.random.seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("mps")
EPOCHS = 300
VAL_EVERY = 3
KAPPA_BETA = 0.3
BATCH_SIZE = 4
NUM_WORKERS = 2
MODEL_FILE = None

def main():

    model = TasNet(
        enc_dim=256,
        kernel=3,
    )

    if MODEL_FILE != None:
        checkpoint = torch.load(MODEL_FILE)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
    else:
        epoch = 0
    model.to(DEVICE)

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
                                    split="train")
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    dataset_val = DNSChallangeDataset(datapath=f"{os.getcwd()}/datasets",
                                    split="val")
    dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # init tensorboard
    writer = SummaryWriter(f"{os.getcwd()}/runs/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")

    while epoch < EPOCHS:
        running_loss = 0
        model.train()
        for batch in tqdm(dataloader_train):
            noisy_signal = batch['noisy_signal'].to(DEVICE)
            target_signal = batch['target_signal'].to(DEVICE)

            optimizer.zero_grad()

            output_signal = model(noisy_signal)[:,0,:]


            import matplotlib.pyplot as plt 

            t = np.arange(0,noisy_signal.shape[1]) / 16000

            plt.figure()
            plt.plot(t, noisy_signal[0], label='noisy signal')
            plt.plot(t, target_signal[0], label='target signal')
            plt.plot(t, output_signal.detach().numpy()[0], label='enhanced signal')
            plt.title(f'Signals in the time domain at epoch {epoch}')
            plt.xlabel('Time in [s]')
            plt.legend()

            plt.figure()
            plt.imshow(model.encoder.weight.detach().numpy().squeeze(1), origin='lower')
            plt.title(f'Encoder Filterbank at epoch {epoch}')

            plt.show()

            if KAPPA_BETA != None:

                # get encoder weights for optimization
                encoder_filterbank = model.encoder.weight.squeeze(1)
                base_loss, loss = loss_fn(output_signal, target_signal, encoder_filterbank)
            else:
                loss = loss_fn(output_signal, target_signal)
            
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
                    noisy_signal = batch['noisy_signal'].to(DEVICE)
                    target_signal = batch['target_signal'].to(DEVICE)

                    output_signal = model(noisy_signal)[:,0,:]

                    loss = loss_fn(output_signal, target_signal)
                    running_val_loss += loss.item()

                    pesq += calculate_pesq(target_signal, output_signal)

                    sisdr += calculate_sisdr(torch.abs(torch.swapaxes(target_signal.detach().cpu(), 1, 2)),
                                             torch.abs(torch.swapaxes(output_signal.detach().cpu(), 1, 2)))

            prediction_sample_time_domain = librosa.istft(output_signal[0].detach().cpu().numpy().T, n_fft=512, hop_length=256)
            writer.add_audio('Prediction Val', prediction_sample_time_domain, epoch, sample_rate=16000)
            target_sample_time_domain = librosa.istft(target_signal[0].detach().cpu().numpy().T, n_fft=512, hop_length=256)
            writer.add_audio('Target Val', target_sample_time_domain, epoch, sample_rate=16000)
            noisy_sample_time_domain = librosa.istft(noisy_signal[0].detach().cpu().numpy().T, n_fft=512, hop_length=256)
            writer.add_audio('Noisy Val', noisy_sample_time_domain, epoch, sample_rate=16000)

            writer.add_scalar('PESQ Val', pesq / len(dataloader_val), epoch)
            writer.add_scalar('SISDR Val', sisdr / len(dataloader_val), epoch)

            writer.add_scalars('Loss', {'Train': running_loss / len(dataloader_train),
                                        'Val': running_val_loss / len(dataloader_val)}, epoch)
        else:
            writer.add_scalars('Loss', {'Train': running_loss / len(dataloader_train),}, epoch)

        if f"{os.getcwd()}/runs/models" not in os.listdir():
            os.mkdir(f"{os.getcwd()}/runs/models")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, f"{os.getcwd()}/runs/models/model_{epoch}.pth")
        
        epoch += 1

if __name__ == "__main__":
    main()

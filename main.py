import os
import datetime
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import librosa
import numpy as np

from src.datasets import DNSChallangeDataset
from src.model import DNN
from src.losses import ComplexCompressedMSELoss
from src.metrics import calculate_pesq, calculate_sisdr

# set seed
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)
np.random.seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 300
VAL_EVERY = 3

def main():

    model = DNN()
    model.to(DEVICE)

    print("#params of model: ", sum(p.numel() for p in model.parameters()))

    loss_fn = ComplexCompressedMSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    dataset_train = DNSChallangeDataset(datapath="datasets",
                                    split="train")
    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=4)

    dataset_val = DNSChallangeDataset(datapath="datasets",
                                    split="val")
    dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=False, num_workers=4)

    # init tensorboard
    writer = SummaryWriter(f"{os.getcwd()}/runs/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")

    for epoch in range(EPOCHS):
        running_loss = 0
        model.train()
        for batch in tqdm(dataloader_train):
            noisy_signal = batch['noisy_signal'].to(DEVICE)
            target_signal = batch['target_signal'].to(DEVICE)

            optimizer.zero_grad()

            noisy_signal_log = torch.log(torch.abs(noisy_signal).to(torch.float32) +\
                                      torch.ones(noisy_signal.shape, dtype=torch.float32).to(DEVICE) * 1e-8)

            output_mask = model(noisy_signal_log)
            output_signal = torch.mul(noisy_signal, output_mask.to(noisy_signal.dtype))

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

                    noisy_signal_log = torch.log(torch.abs(noisy_signal).to(torch.float32) +\
                                            torch.ones(noisy_signal.shape, dtype=torch.float32).to(DEVICE) * 1e-8)
                    output_mask = model(noisy_signal_log)
                    output_signal = torch.mul(noisy_signal, output_mask.to(noisy_signal.dtype))

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
        torch.save(model.state_dict(), f"{os.getcwd()}/runs/models/model_{epoch}.pth")

if __name__ == "__main__":
    main()

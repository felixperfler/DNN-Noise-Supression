import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from src import sdr

from src import models


# Conv-TasNet
class TasNet(nn.Module):
    def __init__(self, enc_dim=512, feature_dim=128, sr=16000, win: float=2., layer: int=8, stack: int=3, 
                 kernel=3, num_spk=2, causal=False, kappa3000=False):
        super(TasNet, self).__init__()
        
        # hyper parameters
        self.num_spk = num_spk

        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        
        self.win = int(sr*win/1000)
        self.stride = self.win // 2
        
        self.layer = layer
        self.stack = stack
        self.kernel = kernel

        self.causal = causal

        self.kappa3000 = kappa3000
        
        # input encoder
        self.encoder = nn.Conv1d(1, self.enc_dim, self.win, bias=False, stride=self.stride)
        # self.encoder = nn.Conv1d(1, self.enc_dim, self.win, bias=False, stride=0, padding_mode='circular')
        
        # TCN separator
        self.TCN = models.TCN(self.enc_dim, self.enc_dim*self.num_spk, self.feature_dim, self.feature_dim*4,
                              self.layer, self.stack, self.kernel, causal=self.causal)

        self.receptive_field = self.TCN.receptive_field
        
        # output decoder
        self.decoder = nn.ConvTranspose1d(self.enc_dim, 1, self.win, bias=False, stride=self.stride)

    def pad_signal(self, input):

        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")
        
        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nsample = input.size(2)
        rest = self.win - (self.stride + nsample % self.win) % self.win
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, 1, rest)).type(torch.float32).to(input.device)
            input = torch.cat([input, pad], 2)
        
        pad_aux = Variable(torch.zeros(batch_size, 1, self.stride)).type(torch.float32).to(input.device)
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest
        
    def forward(self, input):
        
        # padding
        output, rest = self.pad_signal(input)
        batch_size = output.size(0)
        
        # waveform encoder
        enc_output = self.encoder(output)  # B, N, L

        # generate masks
        masks = torch.sigmoid(self.TCN(enc_output)).view(batch_size, self.num_spk, self.enc_dim, -1)  # B, C, N, L
        masked_output = enc_output.unsqueeze(1) * masks  # B, C, N, L
        
        # # waveform decoder
        if self.kappa3000:
            output = F.conv_transpose1d(masked_output.view(batch_size*self.num_spk, self.enc_dim, -1),
                            nn.Parameter(self.encoder.weight.detach().clone()),
                            stride=self.stride)
            output = output[:,:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L
            output = output.view(batch_size, self.num_spk, -1)  # B, C, T
        else:
            output = self.decoder(masked_output.view(batch_size*self.num_spk, self.enc_dim, -1))  # B*C, 1, L
            output = output[:,:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L
            output = output.view(batch_size, self.num_spk, -1)  # B, C, T
        
        return output

def test_conv_tasnet():
    x = torch.rand(2, 32000)
    nnet = TasNet()
    x = nnet(x)
    s1 = x[0]
    print(s1.shape)


if __name__ == "__main__":
    test_conv_tasnet()

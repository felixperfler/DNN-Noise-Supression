import torch

class DNN(torch.nn.Module):
    # https://arxiv.org/pdf/2009.12286.pdf
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.ff1 = torch.nn.Linear(257, 400)
        self.gru1 = torch.nn.GRU(400, 400, batch_first=True)
        self.gru2 = torch.nn.GRU(400, 400, batch_first=True)
        self.ff2 = torch.nn.Linear(400, 600)
        self.ff3 = torch.nn.Linear(600, 600)
        self.ff4 = torch.nn.Linear(600, 257)

    def forward(self, x):
        x = torch.relu(self.ff1(x))
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x = torch.relu(self.ff2(x))
        x = torch.relu(self.ff3(x))
        x = torch.sigmoid(self.ff4(x))

        return x

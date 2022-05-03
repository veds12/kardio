import torch
import torch.nn as nn

def conv_block(in_channels, out_channels, stride):
    block = nn.Sequential(
        nn.BatchNorm1d(in_channels),
        nn.ReLU(),
        nn.Dropout(),
        nn.Conv1d(in_channels, out_channels, kernel_size=32, stride=stride, bias=False),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.Dropout(),
        nn.Conv1d(out_channels, out_channels, kernel_size=32, stride=stride, bias=False),
        nn.MaxPool1d(kernel_size=2)
    )

    return block

class CNNModule(nn.Module):
    def __init__(
        self,
        in_channels=1,
    ):
        super(CNNModule, self).__init__()
        self.block1 = conv_block(in_channels, 64, 1)
        self.block2 = conv_block(64, 64, 1)
        # self.block3 = conv_block(64, 64, 1)
        
        self.output_c = nn.Sequential(
            nn.Linear(1344, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
    
    def forward(self, x):
        x = x.permute(1, 2, 0)
        x = self.block1(x)
        x = self.block2(x)
        # x = self.block3(x)
        x = nn.Flatten()(x)
        out = self.output_c(x)

        return out
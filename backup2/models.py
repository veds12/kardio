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

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.block1 = conv_block(1, 64, 1)
        self.block2 = conv_block(64, 64, 1)

        # self.output_c = nn.Sequential(
        #     nn.Linear(1344, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 1),
        # )

        self.fc1 = nn.Linear(1344, 256)
        self.fc2 = nn.Linear(256, 27)

        self.fc3 = nn.Linear(in_features=27, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=256)
        self.fc5 = nn.Linear(in_features=256, out_features=1)

    def forward(self, x):
        x = x.permute(1, 2, 0)
        x = self.block1(x)
        x = self.block2(x)
        x = nn.Flatten()(x)
        # x = self.output_c(x)

        x = self.fc1(x)
        x = nn.ReLU()(x)
        features = self.fc2(x)

        x = nn.ReLU()(features)
        x = self.fc3(x)
        x = nn.ReLU()(x)
        x = self.fc4(x)
        x = nn.ReLU()(x)
        x = self.fc5(x)
    
        return x, features



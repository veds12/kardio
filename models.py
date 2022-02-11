import torch
import torch.nn as nn

def VanillaMLP(
    layer_sizes,
    activation,
):

    if activation == "relu":
        _activation = nn.ReLU()
    elif activation == "tanh":
        _activation = nn.Tanh()
    else:
        raise NotImplementedError

    _layers = [_activation]

    for i in range(len(layer_sizes) - 2):
        _layers += [nn.Linear(layer_sizes[i], layer_sizes[i + 1]), _activation]

    _layers += [nn.Linear(layer_sizes[-2], layer_sizes[-1])]

    return nn.Sequential(*_layers)

class LSTMModule(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_size=32,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0,
        bidirectional=False,
        proj_size=0,
        out_size=6,
    ):
        super(LSTMModule, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            proj_size=proj_size,
        )

        if proj_size == 0:
            self.proj_size = hidden_size
        else:
            self.proj_size = proj_size

        self.output = nn.Linear(self.proj_size, out_size)

    def forward(self, x, h_0=None, c_0=None):
        if h_0 == None or c_0 == None:
            output, _ = self.lstm(x)
        else:
            output, _ = self.lstm(x, (h_0, c_0))
        output = self.output(output[-1])
        output = nn.Softmax()(output)

        return output

class CNNModule(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_size=6,
    ):
        super(CNNModule, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )
        
        self.output = nn.Sequential(
            nn.Linear(3968, 512),
            nn.ReLU(),
            nn.Linear(512, out_size)
        )
    
    def forward(self, x):
        x = x.permute(1, 2, 0)
        x = self.conv(x).squeeze(1)
        x = self.output(x)
        x = nn.Softmax()(x)

        return x

class MLPModule(nn.Module):
    def __init__(
        self,
        in_length=128,
        hidden_layers=[64, 64],
        out_size=6,
    ):
        super(MLPModule, self).__init__()
        layer_sizes = [in_length, *hidden_layers, out_size]
        self.mlp = VanillaMLP(layer_sizes, "relu")

    def forward(self, x):
        x = x.permute(1, 2, 0).squeeze(1)
        x = self.mlp(x)
        x = nn.Softmax()(x)

        return x

MODELS = {
    'lstm': LSTMModule,
    'cnn': CNNModule,
    'mlp': MLPModule,
}

def get_model(model_name):
    return MODELS[model_name]

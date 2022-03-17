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

class LSTMModule(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=2,
        bias=True,
        batch_first=False,
        dropout=0,
        bidirectional=False,
        proj_size=0,
        out_sizes=[2, 4, 6, 4, 6, 5],
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

        self.heads = nn.ModuleList([nn.Linear(self.proj_size, out_size) for out_size in out_sizes])

    def forward(self, x, h_0=None, c_0=None):
        if h_0 == None or c_0 == None:
            output, _ = self.lstm(x)
        else:
            output, _ = self.lstm(x, (h_0, c_0))
        
        outputs = [nn.Softmax(-1)(self.heads[i](output[-1])) for i in range(len(self.heads))]

        return outputs

class CNNSimple(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_sizes=[2, 4, 6, 4, 6, 5],
    ):
        super(CNNSimple, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=32),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )
        
        self.output_c = nn.Sequential(
            nn.Linear(7552, 512),
            nn.ReLU(),
        )

        self.heads = nn.ModuleList([nn.Linear(512, out_size) for out_size in out_sizes])
    
    def forward(self, x):
        x = x.permute(1, 2, 0)
        x = self.conv(x).squeeze(1)
        x = self.output_c(x)
        
        outputs = [nn.Softmax(-1)(self.heads[i](x)) for i in range(len(self.heads))]

        return outputs

class CNNModule(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_sizes=[2, 4, 6, 4, 6, 5],
    ):
        super(CNNModule, self).__init__()
        self.block1 = conv_block(in_channels, 64, 1)
        self.block2 = conv_block(64, 64, 1)
        # self.block3 = conv_block(64, 64, 1)
        
        self.output_c = nn.Sequential(
            nn.Linear(1344, 512),
            nn.ReLU(),
        )

        self.heads = nn.ModuleList([nn.Linear(512, out_size) for out_size in out_sizes])
    
    def forward(self, x):
        x = x.permute(1, 2, 0)
        x = self.block1(x)
        x = self.block2(x)
        # x = self.block3(x)
        x = nn.Flatten()(x)
        x = self.output_c(x)

        outputs = [nn.Softmax(-1)(self.heads[i](x)) for i in range(len(self.heads))]

        return outputs

class MLPModule(nn.Module):
    def __init__(
        self,
        in_length=2700,
        hidden_layers=[256, 512, 64],
        out_sizes=[2, 4, 6, 4, 6, 5],
    ):
        super(MLPModule, self).__init__()
        layer_sizes = [in_length, *hidden_layers]
        self.mlp = VanillaMLP(layer_sizes, "relu")

        self.heads = nn.ModuleList([nn.Linear(hidden_layers[-1], out_size) for out_size in out_sizes])

    def forward(self, x):
        x = x.permute(1, 2, 0).squeeze(1)
        x = self.mlp(x)

        outputs = [nn.Softmax(-1)(self.heads[i](x)) for i in range(len(self.heads))]

        return outputs

MODELS = {
    'lstm': LSTMModule,
    'cnn': CNNModule,
    'mlp': MLPModule,
    'cnn_simple': CNNSimple,
}

def get_model(model_name):
    return MODELS[model_name]

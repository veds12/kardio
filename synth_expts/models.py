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

        self.output_1 = nn.Linear(self.proj_size, out_size)
        self.output_2 = nn.Linear(self.proj_size, out_size)

    def forward(self, x, h_0=None, c_0=None):
        if h_0 == None or c_0 == None:
            output, _ = self.lstm(x)
        else:
            output, _ = self.lstm(x, (h_0, c_0))
        
        output_1 = self.output_1(output[-1])
        output_1 = nn.Softmax()(output_1)

        output_2 = self.output_2(output[-1])
        output_2 = nn.Softmax()(output_2)

        return output_1, output_2

class CNNModuleBranch(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_sizes=[6, 6],
        neural_branch=False,
    ):
        super(CNNModuleBranch, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )
        # self.block3 = conv_block(64, 64, 1)
        self.neural_branch = neural_branch
        
        self.output_c = nn.Sequential(
            nn.Linear(3968, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
        )

        self.heads = nn.ModuleList([nn.Linear(128, out_size) for out_size in out_sizes])

        if neural_branch:
            self.neural_branch = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 4),
            )
    
    def forward(self, x):
        x = x.permute(1, 2, 0)
        x = self.conv(x).squeeze(1)
        # x = self.block3(x)
        x = self.output_c(x)

        feature_outputs = [nn.Softmax(-1)(self.heads[i](x)) for i in range(len(self.heads))]

        if self.neural_branch:
            neural_output = self.neural_branch(x)
        else:
            neural_output = None


        return feature_outputs, neural_output

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
        
        self.output_c = nn.Sequential(
            nn.Linear(3968, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
        )

        self.output_1 = nn.Linear(128, out_size)
        self.output_2 = nn.Linear(128, out_size)
    
    def forward(self, x):
        x = x.permute(1, 2, 0)
        x = self.conv(x).squeeze(1)
        x = self.output_c(x)
        
        output_1 = self.output_1(x)
        output_1 = nn.Softmax()(output_1)

        output_2 = self.output_2(x)
        output_2 = nn.Softmax()(output_2)

        return output_1, output_2

class MLPModule(nn.Module):
    def __init__(
        self,
        in_length=128,
        hidden_layers=[64, 128],
        out_size=6,
    ):
        super(MLPModule, self).__init__()
        layer_sizes = [in_length, *hidden_layers]
        self.mlp = VanillaMLP(layer_sizes, "relu")

        self.output_1 = nn.Linear(hidden_layers[-1], out_size)
        self.output_2 = nn.Linear(hidden_layers[-1], out_size)

    def forward(self, x):
        x = x.permute(1, 2, 0).squeeze(1)
        x = self.mlp(x)

        output_1 = self.output_1(x)
        output_1 = nn.Softmax()(output_1)

        output_2 = self.output_2(x)
        output_2 = nn.Softmax()(output_2)

        return output_1, output_2

MODELS = {
    'lstm': LSTMModule,
    'cnn': CNNModuleBranch,
    'mlp': MLPModule,
}

def get_model(model_name):
    return MODELS[model_name]

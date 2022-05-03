import torch
import torch.nn as nn

class Encoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder, self).__init__()

    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=False
    )
    
    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=False
    )

  def forward(self, x):
    # x = x.reshape((1, self.seq_len, self.n_features))

    x, (_, _) = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)

    return hidden_n

class Decoder(nn.Module):

  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(Decoder, self).__init__()

    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features

    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=False
    )

    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=False
    )

    self.output_layer = nn.Linear(self.hidden_dim, n_features)

  def forward(self, x):
    x = x.repeat(self.seq_len, 1, 1)

    x, (hidden_n, cell_n) = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)

    return self.output_layer(x)

class RecurrentAutoencoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(RecurrentAutoencoder, self).__init__()

    self.encoder = Encoder(seq_len, n_features, embedding_dim)
    self.decoder = Decoder(seq_len, embedding_dim, n_features)

  def forward(self, x):
    x_emb = self.encoder(x)
    x = self.decoder(x_emb)

    return x, x_emb

def conv_block(in_channels, out_channels, stride, pool=True):
  if pool:
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
  else:
    block = nn.Sequential(
        nn.BatchNorm1d(in_channels),
        nn.ReLU(),
        nn.Dropout(),
        nn.Conv1d(in_channels, out_channels, kernel_size=32, stride=stride, bias=False),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.Dropout(),
        nn.Conv1d(out_channels, out_channels, kernel_size=32, stride=stride, bias=False),
    )

  return block

def transpose_conv_block(in_channels, out_channels, stride):
  block = nn.Sequential(
    nn.BatchNorm1d(in_channels),
    nn.ReLU(),
    nn.Dropout(),
    nn.ConvTranspose1d(in_channels, out_channels, kernel_size=32, stride=stride, bias=False),
    nn.BatchNorm1d(out_channels),
    nn.ReLU(),
    nn.Dropout(),
    nn.ConvTranspose1d(out_channels, out_channels, kernel_size=32, stride=stride, bias=False),
  )

  return block

class ConvolutionalAutoencoder(nn.Module):
  def __init__(self,
              seq_len,
              n_features,
              embedding_dim=64,
              ):
    super(ConvolutionalAutoencoder, self).__init__()

    self.encoder = nn.ModuleList([
      conv_block(n_features, 64, 2, False),
      # conv_block(32, 64, 1, False),
      ])
    
    self.encoder_project = nn.Sequential(
      nn.Flatten(),
      # nn.Linear(9344, 1024),
      # nn.Linear(1024, emb_size)
      nn.Linear(2880, 512),
      nn.Linear(512, embedding_dim)
      )
    
    self.decode_project = nn.Sequential(
      # nn.Linear(emb_size, 1024),
      # nn.Linear(1024, 9344),
      nn.Linear(embedding_dim, 512),
      nn.Linear(512, 2880),
      nn.ReLU()
      )

    self.decoder = nn.ModuleList([
      # transpose_conv_block(64, 32, 1),
      transpose_conv_block(64, n_features, 2),
    ])

  def forward(self, x , decode=True):
    x = x.permute(1, 2, 0)
    x_emb = x
    for layer in self.encoder:
      x_emb = layer(x_emb)

    shape = x_emb.shape

    x_emb = self.encoder_project(x_emb)
        
    if decode:
      x = x_emb
      x = self.decode_project(x)

      x = x.view(shape)
      
      for layer in self.decoder:
          x = layer(x)
      
      x = x.permute(2, 0, 1)
  
      return x, x_emb

models = {
  'lstm': RecurrentAutoencoder,
  'cnn': ConvolutionalAutoencoder,
}

def get_model(model_name, seq_len=270, n_features=1, embedding_dim=64):
  model = models[model_name]
  
  return model(seq_len=seq_len, n_features=n_features, embedding_dim=embedding_dim)
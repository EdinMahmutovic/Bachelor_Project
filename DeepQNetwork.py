import torch
import torch.nn as nn
import torch.optim as optim


class DeepQNetwork(nn.Module):
    def __init__(self, input_size, lr, hidden_size, num_layers, n_actions, batch_first=False):
        super(DeepQNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=batch_first)

        self.fc1 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.to(self.device)

    def forward(self, x, seq_length=-1, batch_index=0):
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out[seq_length, batch_index, :])
        out = self.fc2(out)

        return out

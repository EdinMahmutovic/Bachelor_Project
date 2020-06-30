import torch.nn as nn
import torch.optim as optim
import torch


class AutoEncoder(nn.Module):
    def __init__(self, n_input, n1, n2, n3, n4, n5, n6, n7, n_dim, lr):
        super(AutoEncoder, self).__init__()

        self.fc1 = nn.Linear(in_features=n_input, out_features=n1)
        self.fc2 = nn.Linear(in_features=n1, out_features=n2)
        self.fc3 = nn.Linear(in_features=n2, out_features=n3)
        self.fc4 = nn.Linear(in_features=n3, out_features=n4)
        self.fc5 = nn.Linear(in_features=n4, out_features=n5)
        self.fc6 = nn.Linear(in_features=n5, out_features=n6)
        self.fc7 = nn.Linear(in_features=n6, out_features=n7)
        self.fc8 = nn.Linear(in_features=n7, out_features=n_dim)

        self.fc9 = nn.Linear(in_features=n_dim, out_features=n7)
        self.fc10 = nn.Linear(in_features=n7, out_features=n6)
        self.fc11 = nn.Linear(in_features=n6, out_features=n5)
        self.fc12 = nn.Linear(in_features=n5, out_features=n4)
        self.fc13 = nn.Linear(in_features=n4, out_features=n3)
        self.fc14 = nn.Linear(in_features=n3, out_features=n2)
        self.fc15 = nn.Linear(in_features=n2, out_features=n1)
        self.fc16 = nn.Linear(in_features=n1, out_features=n_input)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss()
        self.to(self.device)

    def encoder(self, x):

        out = self.fc1(x)
        out = torch.tanh(out)

        out = self.fc2(out)
        out = torch.tanh(out)

        out = self.fc3(out)
        out = torch.relu(out)

        out = self.fc4(out)
        out = torch.relu(out)

        out = self.fc5(out)
        out = torch.tanh(out)

        out = self.fc6(out)
        out = torch.relu(out)

        out = self.fc7(out)
        out = torch.relu(out)

        out = self.fc8(out)
        out = torch.tanh(out)

        return out

    def decoder(self, x):
        out = self.fc9(x)
        out = torch.tanh(out)

        out = self.fc10(out)
        out = torch.relu(out)

        out = self.fc11(out)
        out = torch.relu(out)

        out = self.fc12(out)
        out = torch.tanh(out)

        out = self.fc13(out)
        out = torch.tanh(out)

        out = self.fc14(out)
        out = torch.relu(out)

        out = self.fc15(out)
        out = torch.relu(out)

        out = self.fc16(out)
        out = torch.softmax(out, dim=1)

        return out

import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pickle


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
        out = torch.relu(out)

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
        out = torch.relu(out)

        out = self.fc13(out)
        out = torch.tanh(out)

        out = self.fc14(out)
        out = torch.relu(out)

        out = self.fc15(out)
        out = torch.relu(out)

        out = self.fc16(out)

        return out


def train_autoencoder(max_size, autoencoder, batch_size, encoder_type):
    accuracy_threshold = 0.9995
    test_accuracies = []
    while True:
        n_samples = 10 ** 5
        x = np.eye(max_size)[np.random.choice(max_size, n_samples)]
        x_train, x_val = train_test_split(x, test_size=0.2)
        x_train, x_test = train_test_split(x_train, test_size=0.2)

        x_test = torch.Tensor(x_test).to(autoencoder.device)
        x_val = torch.Tensor(x_val).to(autoencoder.device)
        data_loader = DataLoader(dataset=x_train, batch_size=batch_size, shuffle=True,
                                 pin_memory=True)

        for i, x_train in enumerate(iter(data_loader)):
            autoencoder.train()
            x_train = x_train.to(autoencoder.device)
            autoencoder.optimizer.zero_grad()
            decoded_x = autoencoder.encoder(x=x_train.float())
            encoded_x = autoencoder.decoder(x=decoded_x)

            loss = autoencoder.loss(encoded_x, torch.argmax(x_train, dim=1))
            loss.backward()
            autoencoder.optimizer.step()

            if i % 1000 == 0:
                autoencoder.eval()
                with torch.no_grad():
                    x_test = x_test.to(autoencoder.device)
                    x_encoded = autoencoder.encoder(x=x_test)
                    x_decoded = autoencoder.decoder(x=x_encoded)

                    max_index = x_decoded.max(dim=1)[1]
                    test_accuracy = (max_index == torch.argmax(x_test, dim=1)).sum()
                    test_accuracy = test_accuracy.item() / x_test.shape[0]
                    test_accuracies.append(test_accuracy)

        autoencoder.eval()
        with torch.no_grad():
            x_encoded = autoencoder.encoder(x=x_val)
            x_decoded = autoencoder.decoder(x=x_encoded)

            max_index = x_decoded.max(dim=1)[1]
            val_accuracy = (max_index == torch.argmax(x_val, dim=1)).sum()
            val_accuracy = val_accuracy.item() / x_val.shape[0]

            if val_accuracy > accuracy_threshold:
                torch.save(autoencoder.state_dict(), "auto_encoder_" + encoder_type + str(max_size) + ".pt")
                with open(encoder_type + '_accuracy' + str(max_size) + '.pickle', 'wb') as b:
                    pickle.dump(test_accuracies, b)

                break

    return autoencoder

import torch.nn.functional as F
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, input_size, input_num_chanels):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.dense1 = nn.Linear(in_features=input_size * input_num_chanels, out_features=128)
        self.batch1 = nn.BatchNorm1d(num_features=128)
        self.dense2 = nn.Linear(in_features=128, out_features=128)
        self.batch2 = nn.BatchNorm1d(num_features=128)
        self.dense3 = nn.Linear(in_features=128, out_features=128)
        self.batch3 = nn.BatchNorm1d(num_features=128)
        self.dense4 = nn.Linear(in_features=128, out_features=128)
        self.batch4 = nn.BatchNorm1d(num_features=128)

        self.bottleneck = nn.Linear(in_features=128, out_features=8)

        self.dense5 = nn.Linear(in_features=8, out_features=128)
        self.batch5 = nn.BatchNorm1d(num_features=128)
        self.dense6 = nn.Linear(in_features=128, out_features=128)
        self.batch6 = nn.BatchNorm1d(num_features=128)
        self.dense7 = nn.Linear(in_features=128, out_features=128)
        self.batch7 = nn.BatchNorm1d(num_features=128)
        self.dense8 = nn.Linear(in_features=128, out_features=128)
        self.batch8 = nn.BatchNorm1d(num_features=128)

        self.output = nn.Linear(in_features=128, out_features=input_size * input_num_chanels)

        self.input_size = input_size
        self.input_num_chanels = input_num_chanels

    def encode(self, x):
        h1 = F.relu(self.batch1(self.dense1(x)))
        h2 = F.relu(self.batch1(self.dense2(h1)))
        h3 = F.relu(self.batch1(self.dense3(h2)))
        h4 = F.relu(self.batch1(self.dense4(h3)))
        h5 = F.relu(self.bottleneck(h4))
        return h5

    def decode(self, z):
        h1 = F.relu(self.batch1(self.dense5(z)))
        h2 = F.relu(self.batch1(self.dense6(h1)))
        h3 = F.relu(self.batch1(self.dense7(h2)))
        h4 = F.relu(self.batch1(self.dense8(h3)))
        h5 = F.relu(self.output(h4))
        return h5

    def forward(self, x):
        x_reshaped = x.reshape(x.shape[0], -1)
        z = self.encode(x_reshaped)
        decoded = self.decode(z)
        return decoded.reshape(x.shape)

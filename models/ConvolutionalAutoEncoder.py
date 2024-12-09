import torch.nn.functional as F
from torch import nn


class ConvolutionalAutoEncoder(nn.Module):
    def __init__(self, input_size, input_num_chanels):
        super(ConvolutionalAutoEncoder, self).__init__()
        start_num_channels = 2
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=start_num_channels, kernel_size=3, stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=start_num_channels, out_channels=start_num_channels, kernel_size=3, stride=1,
                               padding=1)
        self.conv3 = nn.Conv2d(in_channels=start_num_channels, out_channels=start_num_channels*2, kernel_size=3, stride=1,
                               padding=1)
        self.conv4 = nn.Conv2d(in_channels=start_num_channels*2, out_channels=start_num_channels*2, kernel_size=3, stride=1,
                               padding=1)
        self.conv5 = nn.Conv2d(in_channels=start_num_channels*2, out_channels=start_num_channels*4, kernel_size=3, stride=1,
                               padding=1)
        self.conv6 = nn.Conv2d(in_channels=start_num_channels*4, out_channels=start_num_channels*4, kernel_size=3, stride=1,
                               padding=1)
        self.conv7 = nn.Conv2d(in_channels=start_num_channels*4, out_channels=start_num_channels*8, kernel_size=3, stride=1,
                               padding=1)
        self.conv8 = nn.Conv2d(in_channels=start_num_channels*8, out_channels=start_num_channels*8, kernel_size=3, stride=1,
                               padding=1)
        self.conv9 = nn.Conv2d(in_channels=start_num_channels*8, out_channels=start_num_channels*16, kernel_size=3, stride=1,
                               padding=1)
        self.conv10 = nn.Conv2d(in_channels=start_num_channels*16, out_channels=start_num_channels*16, kernel_size=3, stride=1,
                                padding=1)

        # Decoder
        self.conv11 = nn.Conv2d(in_channels=start_num_channels*16, out_channels=start_num_channels*32, kernel_size=3, stride=1,
                                padding=1)
        self.conv12 = nn.Conv2d(in_channels=start_num_channels*32, out_channels=start_num_channels*32, kernel_size=3, stride=1,
                                padding=1)
        self.conv13 = nn.Conv2d(in_channels=start_num_channels*32, out_channels=start_num_channels*16, kernel_size=3, stride=1,
                                padding=1)
        self.conv14 = nn.Conv2d(in_channels=start_num_channels*16, out_channels=start_num_channels*16, kernel_size=3, stride=1,
                                padding=1)
        self.conv15 = nn.Conv2d(in_channels=start_num_channels*16, out_channels=start_num_channels*8, kernel_size=3, stride=1,
                                padding=1)
        self.conv16 = nn.Conv2d(in_channels=start_num_channels*8, out_channels=start_num_channels*8, kernel_size=3, stride=1,
                                padding=1)
        self.conv17 = nn.Conv2d(in_channels=start_num_channels*8, out_channels=start_num_channels*4, kernel_size=3, stride=1,
                                padding=1)
        self.conv18 = nn.Conv2d(in_channels=start_num_channels*4, out_channels=start_num_channels*4, kernel_size=3, stride=1,
                                padding=1)
        self.conv19 = nn.Conv2d(in_channels=start_num_channels*4, out_channels=start_num_channels*2, kernel_size=3, stride=1,
                                padding=1)
        self.conv20 = nn.Conv2d(in_channels=start_num_channels*2, out_channels=start_num_channels*2, kernel_size=3, stride=1,
                                padding=1)
        self.conv21 = nn.Conv2d(in_channels=start_num_channels*2, out_channels=start_num_channels, kernel_size=3, stride=1,
                                padding=1)
        self.conv22 = nn.Conv2d(in_channels=start_num_channels, out_channels=1, kernel_size=3, stride=1,
                                padding=1)

        self.input_size = input_size
        self.input_num_chanels = input_num_chanels

    def encode(self, x):
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h2))
        h4 = F.relu(self.conv4(h3))
        h5 = nn.MaxPool2d(2)(h4)
        h6 = F.relu(self.conv5(h5))
        h7 = F.relu(self.conv6(h6))
        h8 = nn.MaxPool2d(2)(h7)
        h9 = F.relu(self.conv7(h8))
        h10 = F.relu(self.conv8(h9))
        h11 = nn.MaxPool2d(2)(h10)
        h12 = F.relu(self.conv9(h11))
        h13 = F.relu(self.conv10(h12))
        h14 = nn.MaxPool2d(2)(h13)
        return h14

    def decode(self, z):
        h1 = F.relu(self.conv11(z))
        h2 = F.relu(self.conv12(h1))
        h3 = F.relu(self.conv13(h2))
        h4 = F.relu(self.conv14(h3))
        h5 = nn.Upsample(scale_factor=2, mode='bilinear')(h4)
        h6 = F.relu(self.conv15(h5))
        h7 = F.relu(self.conv16(h6))
        h8 = nn.Upsample(scale_factor=2, mode='bilinear')(h7)
        h9 = F.relu(self.conv17(h8))
        h10 = F.relu(self.conv18(h9))
        h11 = nn.Upsample(scale_factor=2, mode='bilinear')(h10)
        h12 = F.relu(self.conv19(h11))
        h13 = F.relu(self.conv20(h12))
        h14b = nn.Upsample(size=(self.input_num_chanels, self.input_size), mode='bilinear')(h13)
        h15 = F.relu(self.conv21(h14b))
        h16 = self.conv22(h15)
        return h16

    def forward(self, x):
        x_reshaped = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        z = self.encode(x_reshaped)
        decoded = self.decode(z)
        return decoded.reshape(x.shape)

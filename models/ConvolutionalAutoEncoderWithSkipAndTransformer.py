import torch
import torch.nn.functional as F
from torch import nn


class ConvolutionalAutoEncoderWithSkipAndTransformer(nn.Module):
    def __init__(self, input_size, input_num_chanels):
        super(ConvolutionalAutoEncoderWithSkipAndTransformer, self).__init__()
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

        self.conv_trans1 = nn.ConvTranspose2d(in_channels=start_num_channels*16, out_channels=start_num_channels*16, stride=2, kernel_size=2)
        self.conv_trans2 = nn.ConvTranspose2d(in_channels=start_num_channels*8, out_channels=start_num_channels*8, stride=2, kernel_size=2)
        self.conv_trans3 = nn.ConvTranspose2d(in_channels=start_num_channels*4, out_channels=start_num_channels*4, stride=2, kernel_size=2)
        self.conv_trans4 = nn.ConvTranspose2d(in_channels=start_num_channels*2, out_channels=start_num_channels*2, stride=2, kernel_size=2)

        self.batch1 = nn.BatchNorm2d(num_features=start_num_channels * 2)
        self.batch2 = nn.BatchNorm2d(num_features=start_num_channels * 4)
        self.batch3 = nn.BatchNorm2d(num_features=start_num_channels * 8)
        self.batch4 = nn.BatchNorm2d(num_features=start_num_channels * 16)

        self.batch5 = nn.BatchNorm2d(num_features=start_num_channels * 16)
        self.batch6 = nn.BatchNorm2d(num_features=start_num_channels * 8)
        self.batch7 = nn.BatchNorm2d(num_features=start_num_channels * 4)
        self.batch8 = nn.BatchNorm2d(num_features=start_num_channels * 2)

        self.transformer_encode = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=500, nhead=10, batch_first=True), num_layers=1)  # nn.MultiheadAttention(500, 10, batch_first=True)
        self.transformer_decode = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=500, nhead=10, batch_first=True), num_layers=1)  # nn.MultiheadAttention(500, 10, batch_first=True)

        self.weights_skip_encode = nn.Linear(in_features=50, out_features=50)
        self.weights_skip_decode = nn.Linear(in_features=50, out_features=50)

        self.input_size = input_size
        self.input_num_chanels = input_num_chanels

    def encode(self, x):
        h1 = F.leaky_relu(self.conv1(x))
        h2 = F.leaky_relu(self.conv2(h1))
        h3 = F.leaky_relu(self.conv3(h2 + h1))
        h4 = F.leaky_relu(self.conv4(h3))
        skip1 = torch.zeros(h4.shape, device=x.device)
        skip1[:, :x.shape[1], :, :] = x
        h5 = nn.MaxPool2d(2)(self.batch1(h4 + h3 + skip1))
        h6 = F.leaky_relu(self.conv5(h5))
        h7 = F.leaky_relu(self.conv6(h6))
        skip2 = torch.zeros(h7.shape, device=x.device)
        skip2[:, :h5.shape[1], :, :] = h5
        h8 = nn.MaxPool2d(2)(self.batch2(h7 + h6 + skip2))
        h9 = F.leaky_relu(self.conv7(h8))
        h10 = F.leaky_relu(self.conv8(h9))
        skip3 = torch.zeros(h10.shape, device=x.device)
        skip3[:, :h8.shape[1], :, :] = h8
        h11 = nn.MaxPool2d(2)(self.batch3(h10 + h9 + skip3))
        h12 = F.leaky_relu(self.conv9(h11))
        h13 = F.leaky_relu(self.conv10(h12))
        skip4 = torch.zeros(h13.shape, device=x.device)
        skip4[:, :h11.shape[1], :, :] = h11
        last = h13
        last_flatten = last.flatten(2, 3)
        last_attention = self.transformer_encode(last_flatten)
        last = last_attention.reshape(last.shape)
        h14 = nn.MaxPool2d(2)(self.batch4((last + F.leaky_relu(self.weights_skip_encode(h12 + skip4)))))
        return h14, last

    def decode(self, z, last_encoder_layer):
        h1 = F.leaky_relu(self.conv11(z))
        h2 = F.leaky_relu(self.conv12(h1))
        h3 = F.leaky_relu(self.conv13(h2 + h1))
        h4 = F.leaky_relu(self.conv14(h3))
        h5 = self.conv_trans1(self.batch5(h4 + h3 + z))
        h5_flatten = h5.flatten(2, 3)
        last_attention = self.transformer_decode(h5_flatten, last_encoder_layer.flatten(2, 3))
        last = last_attention.reshape(h5.shape)
        h6 = F.leaky_relu(self.conv15(last))
        h7 = F.leaky_relu(self.conv16(h6))
        skip2 = last[:, :h7.shape[1], :, :]
        h8 = self.conv_trans2(self.batch6(h7 + F.leaky_relu(self.weights_skip_decode(h6 + skip2))))
        h9 = F.leaky_relu(self.conv17(h8))
        h10 = F.leaky_relu(self.conv18(h9))
        skip3 = h8[:, :h10.shape[1], :, :]
        h11 = self.conv_trans3(self.batch7(h10 + h9 + skip3))
        h12 = F.leaky_relu(self.conv19(h11))
        h13 = F.leaky_relu(self.conv20(h12))
        skip4 = h11[:, :h13.shape[1], :, :]
        h14a = self.conv_trans4(self.batch8(h13 + h12 + skip4))
        h14b = nn.Upsample(size=(self.input_num_chanels, self.input_size), mode='bilinear')(h14a)
        h15 = F.leaky_relu(self.conv21(h14b))
        skip5 = h14b[:, :h15.shape[1], :, :]
        h16 = self.conv22(h15 + skip5)
        return h16

    def forward(self, x):
        x_reshaped = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        z, last = self.encode(x_reshaped)
        decoded = self.decode(z, last)
        return decoded.reshape(x.shape)

import torch
import numpy as np


class InceptionBlock(torch.nn.Module):
    def __init__(self, in_channels, name):
        super().__init__()

        self.branch_0 = InceptionBlockBranch0(in_channels, name)
        self.branch_1 = InceptionBlockBranch1(in_channels, name)
        self.branch_2 = InceptionBlockBranch2(in_channels, name)
        self.branch_3 = InceptionBlockBranch3(in_channels, name)


    def forward(self, input):
        branch_0 = self.branch_0(input)
        branch_1 = self.branch_1(input)
        branch_2 = self.branch_2(input)
        branch_3 = self.branch_3(input)

        return torch.cat([branch_0, branch_1, branch_2, branch_3], dim=1)


class InceptionBlockBranch0(torch.nn.Module):
    def __init__(self, in_channels, name):
        super().__init__()
        out_channels_choices = {
            'Mixed_3b': 64,
            'Mixed_3c': 128,
            'Mixed_4b': 192,
            'Mixed_4c': 160,
            'Mixed_4d': 128,
            'Mixed_4e': 112,
            'Mixed_4f': 256,
            'Mixed_5b': 256,
            'Mixed_5c': 384}
        out_channels = out_channels_choices[name]
        kernel = (1, 1, 1)
        stride = (1, 1, 1)
        self.Conv3d_0a_1x1 = Conv3dBlock(in_channels, out_channels,
                                         kernel, stride)

    def forward(self, input):
        return self.Conv3d_0a_1x1(input)


class InceptionBlockBranch1(torch.nn.Module):
    def __init__(self, in_channels, name):
        super().__init__()
        out_channels_choices = {
            'Mixed_3b': 96,
            'Mixed_3c': 128,
            'Mixed_4b': 96,
            'Mixed_4c': 112,
            'Mixed_4d': 128,
            'Mixed_4e': 144,
            'Mixed_4f': 160,
            'Mixed_5b': 160,
            'Mixed_5c': 192}
        out_channels = out_channels_choices[name]
        kernel = (1, 1, 1)
        stride = (1, 1, 1)
        self.Conv3d_0a_1x1 = Conv3dBlock(in_channels, out_channels,
                                         kernel, stride)
        in_channels = out_channels

        out_channels_choices = {
            'Mixed_3b': 128,
            'Mixed_3c': 192,
            'Mixed_4b': 208,
            'Mixed_4c': 224,
            'Mixed_4d': 256,
            'Mixed_4e': 288,
            'Mixed_4f': 320,
            'Mixed_5b': 320,
            'Mixed_5c': 384}
        out_channels = out_channels_choices[name]
        kernel = (3, 3, 3)
        stride = (1, 1, 1)
        self.Conv3d_0b_3x3 = Conv3dBlock(in_channels, out_channels,
                                         kernel, stride)

    def forward(self, input):
        out = self.Conv3d_0a_1x1(input)
        return self.Conv3d_0b_3x3(out)


class InceptionBlockBranch2(torch.nn.Module):
    def __init__(self, in_channels, name):
        super().__init__()
        out_channels_choices = {
            'Mixed_3b': 16,
            'Mixed_3c': 32,
            'Mixed_4b': 16,
            'Mixed_4c': 24,
            'Mixed_4d': 24,
            'Mixed_4e': 32,
            'Mixed_4f': 32,
            'Mixed_5b': 32,
            'Mixed_5c': 48}
        out_channels = out_channels_choices[name]
        kernel = (1, 1, 1)
        stride = (1, 1, 1)
        self.Conv3d_0a_1x1 = Conv3dBlock(in_channels, out_channels,
                                         kernel, stride)
        in_channels = out_channels

        out_channels_choices = {
            'Mixed_3b': 32,
            'Mixed_3c': 96,
            'Mixed_4b': 48,
            'Mixed_4c': 64,
            'Mixed_4d': 64,
            'Mixed_4e': 64,
            'Mixed_4f': 128,
            'Mixed_5b': 128,
            'Mixed_5c': 128}
        out_channels = out_channels_choices[name]
        kernel = (3, 3, 3)
        stride = (1, 1, 1)
        self.Conv3d_0b_3x3 = Conv3dBlock(in_channels, out_channels,
                                         kernel, stride)

    def forward(self, input):
        out = self.Conv3d_0a_1x1(input)
        return self.Conv3d_0b_3x3(out)


class InceptionBlockBranch3(torch.nn.Module):
    def __init__(self, in_channels, name):
        super().__init__()

        kernel = (3, 3, 3)
        stride = (1, 1, 1)
        padding = 1  # same
        self.MaxPool3d_0a_3x3 = torch.nn.MaxPool3d(kernel, stride, padding)

        out_channels_choices = {
            'Mixed_3b': 32,
            'Mixed_3c': 64,
            'Mixed_4b': 64,
            'Mixed_4c': 64,
            'Mixed_4d': 64,
            'Mixed_4e': 64,
            'Mixed_4f': 128,
            'Mixed_5b': 128,
            'Mixed_5c': 128}
        out_channels = out_channels_choices[name]
        kernel = (1, 1, 1)
        stride = (1, 1, 1)
        self.Conv3d_0b_1x1 = Conv3dBlock(in_channels, out_channels,
                                         kernel, stride)
        in_channels = out_channels

    def forward(self, input):
        out = self.MaxPool3d_0a_3x3(input)
        return self.Conv3d_0b_1x1(out)


class Conv3dBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 use_bias=False, use_batch_norm=True,
                 activation_fn=torch.nn.ReLU(), padding='same',
                 name='Conv3dBlock'):
        super().__init__()

        self.activation_fn = activation_fn

        if use_batch_norm:
            self.batch_norm = torch.nn.BatchNorm3d(out_channels)
        else:
            self.batch_norm = None

        if padding.lower() == 'same':
            kernel_size_arr = np.array(kernel_size)
            padding = np.floor(kernel_size_arr / 2.).astype(np.int64)
            padding = tuple([int(pad) for pad in padding])
        self.conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size,
                                      stride, padding=padding,
                                      bias=use_bias)

    def forward(self, input):
        out = self.conv3d(input)
        if self.batch_norm is not None:
            out = self.batch_norm(out)
        if self.activation_fn is not None:
            out = self.activation_fn(out)
        return out


class InceptionI3d(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        in_channels = 3

        out_channels = 64
        kernel = (7, 7, 7)
        stride = (2, 2, 2)
        self.Conv3d_1a_7x7 = Conv3dBlock(in_channels, out_channels, kernel,
                                         stride)
        in_channels = out_channels

        kernel = (1, 3, 3)
        stride = (1, 2, 2)
        padding = (0, 1, 1)  # same
        self.MaxPool3d_2a_3x3 = torch.nn.MaxPool3d(kernel, stride, padding)

        out_channels = 64
        kernel = (1, 1, 1)
        stride = (1, 1, 1)
        self.Conv3d_2b_1x1 = Conv3dBlock(in_channels, out_channels, kernel,
                                         stride)
        in_channels = out_channels

        out_channels = 192
        kernel = (3, 3, 3)
        stride = (1, 1, 1)
        self.Conv3d_2c_3x3 = Conv3dBlock(in_channels, out_channels, kernel,
                                         stride)
        in_channels = out_channels

        kernel = (1, 3, 3)
        stride = (1, 2, 2)
        padding = (0, 1, 1)  # same
        self.MaxPool3d_3a_3x3 = torch.nn.MaxPool3d(kernel, stride, padding)

        self.Mixed_3b = InceptionBlock(in_channels, 'Mixed_3b')
        in_channels = 256

        self.Mixed_3c = InceptionBlock(in_channels, 'Mixed_3c')
        in_channels = 480

        kernel = (3, 3, 3)
        stride = (2, 2, 2)
        padding = 1  # same
        self.MaxPool3d_4a_3x3 = torch.nn.MaxPool3d(kernel, stride, padding)

        self.Mixed_4b = InceptionBlock(in_channels, 'Mixed_4b')
        in_channels = 512

        self.Mixed_4c = InceptionBlock(in_channels, 'Mixed_4c')
        in_channels = 512

        self.Mixed_4d = InceptionBlock(in_channels, 'Mixed_4d')
        in_channels = 512

        self.Mixed_4e = InceptionBlock(in_channels, 'Mixed_4e')
        in_channels = 528

        self.Mixed_4f = InceptionBlock(in_channels, 'Mixed_4f')
        in_channels = 832

        kernel = (2, 2, 2)
        stride = (2, 2, 2)
        padding = 0  # same
        self.MaxPool3d_5a_2x2 = torch.nn.MaxPool3d(kernel, stride, padding)

        self.Mixed_5b = InceptionBlock(in_channels, 'Mixed_5b')
        in_channels = 832

        self.Mixed_5c = InceptionBlock(in_channels, 'Mixed_5c')
        in_channels = 1024

        kernel = (2, 7, 7)
        stride = (1, 1, 1)
        padding = (0, 0, 0)  # valid
        self.AvgPool3d_6a_7x7 = torch.nn.MaxPool3d(kernel, stride, padding)

        out_channels = num_classes
        kernel = (1, 1, 1)
        stride = (1, 1, 1)
        self.Conv3d_6b_1x1 = Conv3dBlock(in_channels, out_channels, kernel,
                                         stride, use_bias=True,
                                         use_batch_norm=False,
                                         activation_fn=None)
        self.dropout = torch.nn.Dropout3d(p=1.)
        self.softmax = torch.nn.Softmax()


    def forward(self, input):
        """The following comments gives the shape of the output assuming the
        input has the following shape:
            (3, 79, 224, 224) (3 channels, 79 timesteps, 224x224 frame)
        As provided by the `evaluate_sample` in the kinetics-i3d repo."""
        out = self.Conv3d_1a_7x7(input)  # (64, 40, 112, 112)

        out = self.MaxPool3d_2a_3x3(out)  # (64, 40, 56, 56)

        out = self.Conv3d_2b_1x1(out)  # (64, 40, 56, 56)
        out = self.Conv3d_2c_3x3(out)  # (192, 40, 56, 56)

        out = self.MaxPool3d_3a_3x3(out)  # (192, 40, 28, 28)
        out = self.Mixed_3b(out)  # (256, 40, 28, 28)
        out = self.Mixed_3c(out)  # (480, 40, 28, 28)

        out = self.MaxPool3d_4a_3x3(out)  # (512, 20, 14, 14)
        out = self.Mixed_4b(out)  # (512, 20, 14, 14)
        out = self.Mixed_4c(out)  # (512, 20, 14, 14)
        out = self.Mixed_4d(out)  # (512, 20, 14, 14)
        out = self.Mixed_4e(out)  # (528, 20, 14, 14)
        out = self.Mixed_4f(out)  # (832, 20, 14, 14)

        out = self.MaxPool3d_5a_2x2(out)  # (832, 10, 7, 7)
        out = self.Mixed_5b(out)  # (832, 10, 7, 7)
        out = self.Mixed_5c(out)  # (1024, 10, 7, 7)

        out = self.AvgPool3d_6a_7x7(out)  # (1024, 9, 1, 1)
        out = self.dropout(out)
        out = self.Conv3d_6b_1x1(out)  # (NUM_CLASSES, 9, 1, 1)

        out = torch.squeeze(out, dim=4)  # removing width  (NUM_CLASSES, 9, 1)
        out = torch.squeeze(out, dim=3)  # removing height (NUM_CLASSES, 9)
        out = torch.mean(out, dim=2)  # averaging across time. (NUM_CLASSES)
        out = self.softmax(out)

        return out


model = InceptionI3d(num_classes=400)
model.cuda()

input = np.random.random([1, 3, 79, 224, 224])
input_pt = torch.FloatTensor(input)
input_pt = torch.autograd.Variable(input_pt).cuda()

la = model(input_pt)

import pdb; pdb.set_trace()

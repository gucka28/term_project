import torch.nn as nn


""" Optional conv block """
def conv_block(in_channels, out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

""" Define your own model """
class FewShotModel(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.conv1 = conv_block(x_dim, hid_dim)
        self.conv2 = conv_block(hid_dim, hid_dim)
        self.conv3 = conv_block(hid_dim, hid_dim)
        self.conv4 = conv_block(hid_dim, z_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = nn.MaxPool2d(5)(x)
        embedding_vector = x.view(x.size(0),-1)
        return embedding_vector

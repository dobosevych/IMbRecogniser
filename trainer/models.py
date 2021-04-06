import torch.nn as nn
import torch.nn.functional as F
import torch

class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        self.gru_hidden_size = 256
        self.gru_num_layers = 2
        self.cnn_output_height = 4
        num_classes = 4
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3))
        self.norm1 = nn.InstanceNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2))
        self.norm2 = nn.InstanceNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.norm3 = nn.InstanceNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(5, 1), stride=(2, 2))
        self.norm4 = nn.InstanceNorm2d(64)
        self.gru_input_size = self.cnn_output_height * 32
        self.gru = nn.GRU(self.gru_input_size, self.gru_hidden_size, self.gru_num_layers,
                          batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.gru_hidden_size * 2, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.leaky_relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.leaky_relu(out)
        out = self.conv3(out)
        out = self.norm3(out)
        out = F.leaky_relu(out)
        out = self.conv4(out)
        out = self.norm4(out)
        out = F.leaky_relu(out)
        out = out.reshape(-1, batch_size, self.gru_input_size)
        out, _ = self.gru(out)
        out = torch.stack([F.log_softmax(self.fc(out[i]), dim=-1) for i in range(out.shape[0])])
        return out
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
        self.norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3))
        self.norm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2))
        self.norm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 1), stride=(2, 2))
        self.norm4 = nn.BatchNorm2d(64)
        self.gru_input_size = self.cnn_output_height * 32
        self.gru = nn.GRU(self.gru_input_size, self.gru_hidden_size, self.gru_num_layers,
                          batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.gru_hidden_size * 2, num_classes + 1)

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.norm3(out)
        out = F.relu(out)
        out = self.conv4(out)
        out = self.norm4(out)
        out = F.relu(out)
        out = out.reshape(-1, batch_size, self.gru_input_size)
        out, _ = self.gru(out)
        #print(out.shape)
        out = torch.stack([self.fc(out[i]) for i in range(out.shape[0])])
        return out

class BarcodeRecognizer(nn.Module):
    def __init__(self, chars=10, languages=0):
        super(BarcodeRecognizer, self).__init__()
        self.chars = chars
        self.languages = languages
        self.conv0 = nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1)
        self.conv5 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv8 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.conv10 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.conv13 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.conv16 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)
        self.conv18 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)
        self.conv21 = nn.Conv2d(128, 256, kernel_size=(3, 1), padding=1)
        self.conv24 = nn.Conv2d(256, 256, kernel_size=(3, 1), padding=1) # 128
        self.conv26 = nn.Conv2d(256, 256, kernel_size=(3, 1), padding=1)
        self.conv29 = nn.Conv2d(256, 512, kernel_size=(2, 1), padding=(0, 1))
        self.conv32 = nn.Conv2d(512, 512, kernel_size=(1, 1), padding=(0, 2))
        # 7 3
        self.conv36 = nn.Conv2d(512, chars + languages, kernel_size=(1, 1), padding=(0, 4))

        self.bn7 = nn.BatchNorm2d(64, affine=False)
        self.bn15 = nn.BatchNorm2d(128, affine=False)
        self.bn23 = nn.BatchNorm2d(256, affine=False)
        self.bn31 = nn.BatchNorm2d(512, affine=False)
        self.bn34 = nn.BatchNorm2d(512, affine=False)
        self.dp35 = nn.Dropout2d(0.2)

    def forward(self, x):
        # print(x.size())
        x = F.leaky_relu(self.conv0(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        x = F.max_pool2d(x, kernel_size=(2, 2))
        x = F.leaky_relu(self.conv5(x), negative_slope=0.1)
        x = self.bn7(x)
        x = F.leaky_relu(self.conv8(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv10(x), negative_slope=0.1)
        x = F.max_pool2d(x, kernel_size=(2, 1))
        x = F.leaky_relu(self.conv13(x), negative_slope=0.1)
        x = self.bn15(x)
        x = F.leaky_relu(self.conv16(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv18(x), negative_slope=0.1)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 1))
        x = F.leaky_relu(self.conv21(x), negative_slope=0.1)
        x = self.bn23(x)
        x = F.leaky_relu(self.conv24(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv26(x), negative_slope=0.10000000149)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 1))
        x = F.leaky_relu(self.conv29(x), negative_slope=0.1)
        x = self.bn31(x)
        x = F.leaky_relu(self.conv32(x), negative_slope=0.1)
        x = self.bn34(x)
        x = self.dp35(x)
        x = self.conv36(x)
        # B x |A| x 1 x W
        if self.languages > 0:
            x = x[:, self.languages:]
        x = x.permute(3, 0, 2, 1)
        x = x.squeeze(2)
        out_ctc = F.log_softmax(x, dim=2)
        return out_ctc
import torch
import torch.nn as nn
import torch.nn.functional as F


class SERModel(nn.Module):
    def __init__(
        self,
        input_dim=77,   # mel(64) + mfcc(13)
        num_classes=5,
        cnn_out_channels=64,
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.5
    ):
        super(SERModel, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(cnn_out_channels),
            nn.ReLU(),
            nn.Conv2d(cnn_out_channels, cnn_out_channels, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(cnn_out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        lstm_input_dim = cnn_out_channels * (input_dim // 2)

        self.bilstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        self.attention = nn.Linear(2 * lstm_hidden, 1)

        self.fc = nn.Sequential(
            nn.Linear(2 * lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)

        B, C, T, F_a = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, C * F_a)

        x, _ = self.bilstm(x)

        attn_weights = F.softmax(self.attention(x), dim=1)
        x = torch.sum(attn_weights * x, dim=1)

        out = self.fc(x)
        return out

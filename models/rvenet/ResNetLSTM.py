import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNetLSTM(nn.Module):
    def __init__(self, hidden_size=128, num_layers=2):
        super().__init__()
        self.loss_criterion = nn.MSELoss(reduction='mean')

        self.resnet = resnet18(weights='DEFAULT')
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove final layer

        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.regressor = nn.Linear(hidden_size, 1) # Convert classification to reg

    def forward(self, x):
        batch_size, sequence_length, C, H, W = x.shape

        x = x.view(batch_size * sequence_length, C, H, W)
        features = self.resnet(x)
        features = features.view(batch_size, sequence_length, -1)  # Reshape for LSTM

        lstm_out, _ = self.lstm(features)
        final_output = lstm_out[:, -1, :]

        # Regression
        output = self.regressor(final_output)
        return output

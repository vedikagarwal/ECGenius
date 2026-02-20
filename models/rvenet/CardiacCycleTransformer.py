import torch
import torch.nn as nn
import torchvision.models as models

class CardiacCycleTransformer(nn.Module):
    def __init__(self, transformer_hidden_size=128, num_heads=4, num_layers=2, dropout=0.1):
        super(CardiacCycleTransformer, self).__init__()

        # Use a pre-trained ResNet for feature extraction
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        feature_size = resnet.fc.in_features

        # Transformer Encoder Layer
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )

        # Fully connected layer to perform regression
        self.fc = nn.Linear(feature_size, 1)

        self.loss_criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        batch_size, num_cycles, channels, height, width = x.size()

        # Extract features using the ResNet backbone
        features = []
        for cycle in range(num_cycles):
            cycle_features = self.feature_extractor(x[:, cycle])
            cycle_features = cycle_features.view(batch_size, -1)  # Flatten to (batch_size, feature_size)
            features.append(cycle_features)

        features = torch.stack(features, dim=1)  # Shape: (batch_size, num_cycles, feature_size)

        # Apply Transformer Encoder
        transformer_out = self.transformer_encoder(features.transpose(0, 1))  # Transpose to (num_cycles, batch_size, feature_size)

        # Take the output from the last cycle and pass it through the fully connected layer
        output = self.fc(transformer_out[-1, :, :])  # Shape: (batch_size, 1)

        return output

    def compute_loss(self, predictions, targets):
        return self.loss_criterion(predictions, targets)

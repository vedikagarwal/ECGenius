import torch
import torch.nn as nn
import torchvision.models as models


class ResNextTemporal(nn.Module):
    def __init__(self, backbone="resnext"):
        super().__init__()
        self.loss_criterion = nn.MSELoss(reduction='mean')
        # Feature extractor backbone
        if backbone == "resnext":
            base_model = models.resnext50_32x4d(weights='DEFAULT')
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])
            self.feature_dim = 2048
        else:  # shufflenet
            base_model = models.shufflenet_v2_x1_0(weights='DEFAULT')
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])
            self.feature_dim = 232
            
        # Temporal convolution - input: [1, 20*2048, 7, 7]
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(20*self.feature_dim, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        # Output: [1, 256, 7, 7]
        
        # Fully connected layers
        self.fc1 = nn.Linear(256*7*7, 1024)
        self.fc2 = nn.Linear(1024, 1)
        
    def forward(self, x):
        # Input: [batch, 20, 3, 224, 224]
        batch_size = x.size(0)
        
        # Reshape for feature extraction
        x = x.view(-1, 3, 224, 224)  # [batch*20, 3, 224, 224]
        
        # Extract features
        x = self.feature_extractor(x)  # [batch*20, feature_dim, 7, 7]
        
        # Reshape to [batch, 20*feature_dim, 7, 7]
        x = x.view(batch_size, 20*self.feature_dim, 7, 7)
        
        # Temporal convolution
        x = self.temporal_conv(x)  # [batch, 256, 7, 7]
        
        # Flatten
        x = x.view(batch_size, -1)  # [batch, 256*7*7]
        
        # Final fully connected layers
        x = self.fc1(x)  # [batch, 1024]
        x = self.fc2(x)  # [batch, 1]
        
        return x
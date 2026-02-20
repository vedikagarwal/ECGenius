import torch
import torch.nn as nn
import torchvision.models as models

class CardiacCycleRNN(nn.Module):
    def __init__(self, rnn_hidden_size=128, num_rnn_layers=1, num_augmented_features=0):
        super(CardiacCycleRNN, self).__init__()
        resnet = models.resnet18(weights='DEFAULT')
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        feature_size = resnet.fc.in_features

        self.rnn = nn.LSTM(input_size=feature_size, 
                           hidden_size=rnn_hidden_size, 
                           num_layers=num_rnn_layers, 
                           batch_first=True)
        

        self.fc = nn.Linear(rnn_hidden_size, 1)
        if num_augmented_features > 0:
            self.final_dense_1 = nn.Linear(1 + num_augmented_features, 5)
            self.final_dense_2 = nn.Linear(5, 3)
            self.final_dense_3 = nn.Linear(3, 1)
        
        self.loss_criterion = nn.MSELoss(reduction='mean')
        
    def forward(self, x, augmented_features=None):

        batch_size, num_cycles, channels, height, width = x.size()
 
        features = []
        for cycle in range(num_cycles):
            cycle_features = self.feature_extractor(x[:, cycle])
            cycle_features = cycle_features.view(batch_size, -1)
            features.append(cycle_features)
        
        features = torch.stack(features, dim=1)

        rnn_out, _ = self.rnn(features)

        output = self.fc(rnn_out[:, -1, :]) 
        
        if augmented_features is not None:
            combined_input = torch.cat((output, augmented_features), dim=1)
            output = self.final_dense_1(combined_input)
            output = self.final_dense_2(output)
            output = self.final_dense_3(output)
        
        return output

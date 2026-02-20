############ Standard: No need to change ############

import os
import sys
module_path = os.path.dirname(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)

############ Hyperparameters ############

learning_rate = 1e-3
weight_decay = 1e-7
num_epochs = 50

load_model_from_disk = False
save_model_to_disk = True
num_augmented_features = 2

############ Model Selection ############

from models.rvenet.CardiacCycleRNN import CardiacCycleRNN
model = CardiacCycleRNN(num_augmented_features=num_augmented_features)

############ Standard: No need to change ############

import torch
import numpy as np

import json
with open(os.path.join(module_path, 'setup', 'environment.json'), 'r') as f:
    environment = json.load(f)

# Might be faster because tmp dir is not in NFS
os.environ['TEMP'] = environment['new_temp_folder']
os.environ['TEMPDIR'] = environment['new_temp_folder']

# autoreload imports
# %load_ext autoreload
# %autoreload 2

is_cuda = torch.cuda.is_available()

from multiprocessing import util
util.get_temp_dir()

############ Standard: No need to change ############

from runner import Trainer

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
trainer = Trainer(
    environment=environment,
    model=model,
    optimizer=optimizer,
    model_dir=os.path.join(environment['model_checkpoint_path'], model.__class__.__name__, f'augmented_features_{num_augmented_features}'),
    train_data_transforms=None,
    val_data_transforms=None,
    batch_size=8,
    load_from_disk=load_model_from_disk,
    cuda=is_cuda,
    num_augmented_features=num_augmented_features,
)
trainer.run_training_loop(num_epochs=num_epochs)
trainer.plot_loss_history()
trainer.plot_predictions()

if save_model_to_disk:
    trainer.save_model()

############ Standard: No need to change ############

trainer.model.eval()

predicted_values = []
actual_values = []

with torch.no_grad():
    for (dictionary, label) in trainer.val_loader:
        x=dictionary['video_tensor']
        y=label
        if trainer.cuda:
            x = x.cuda()
            y = y.cuda()

        if trainer.num_augmented_features > 0:
            extra_features=dictionary['extra_features']
            if trainer.cuda:
                extra_features=extra_features.cuda()

            predictions = model(x,extra_features)
        else:
            predictions = model(x)

        predicted_values.extend(predictions.cpu().numpy())
        actual_values.extend(y.cpu().numpy())

actual_values = np.array(actual_values).squeeze()
predicted_values = np.array(predicted_values).squeeze()

# Show some values for comparison
show_values = 10
random_idx = np.random.randint(0, len(predicted_values), show_values, dtype=int)

import pandas as pd
print(pd.DataFrame({"Actual":actual_values[random_idx],"Predicted":predicted_values[random_idx]}))


############ Hyperparameters ############

num_augmented_features = 0

############ Model Selection ############

from models.rvenet.CardiacCycleRNN import CardiacCycleRNN
model = CardiacCycleRNN(num_augmented_features=num_augmented_features)

############ Standard: No need to change ############

from runner import Trainer

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
trainer = Trainer(
    environment=environment,
    model=model,
    optimizer=optimizer,
    model_dir=os.path.join(environment['model_checkpoint_path'], model.__class__.__name__, f'augmented_features_{num_augmented_features}'),
    train_data_transforms=None,
    val_data_transforms=None,
    batch_size=8,
    load_from_disk=load_model_from_disk,
    cuda=is_cuda,
    num_augmented_features=num_augmented_features,
)
trainer.run_training_loop(num_epochs=num_epochs)
trainer.plot_loss_history()
trainer.plot_predictions()

if save_model_to_disk:
    trainer.save_model()

############ Standard: No need to change ############

trainer.model.eval()

predicted_values = []
actual_values = []

with torch.no_grad():
    for (dictionary, label) in trainer.val_loader:
        x=dictionary['video_tensor']
        y=label
        if trainer.cuda:
            x = x.cuda()
            y = y.cuda()

        if trainer.num_augmented_features > 0:
            extra_features=dictionary['extra_features']
            if trainer.cuda:
                extra_features=extra_features.cuda()

            predictions = model(x,extra_features)
        else:
            predictions = model(x)

        predicted_values.extend(predictions.cpu().numpy())
        actual_values.extend(y.cpu().numpy())

actual_values = np.array(actual_values).squeeze()
predicted_values = np.array(predicted_values).squeeze()

# Show some values for comparison
show_values = 10
random_idx = np.random.randint(0, len(predicted_values), show_values, dtype=int)

import pandas as pd
print(pd.DataFrame({"Actual":actual_values[random_idx],"Predicted":predicted_values[random_idx]}))

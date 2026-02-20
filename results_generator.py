############ Standard: No need to change ############

import os
import sys
# module_path = os.path.dirname(os.getcwd())
module_path = os.getcwd()
if module_path not in sys.path:
    sys.path.append(module_path)

############ Hyperparameters ############

learning_rate = 1e-3
weight_decay = 1e-7
num_epochs = 10

load_model_from_disk = False
save_model_to_disk = True

############ Model Selection ############

fail = False
if len(sys.argv) > 1:
    if sys.argv[1] == "CardiacCycleRNN":
        from models.rvenet.CardiacCycleRNN import CardiacCycleRNN
        model = CardiacCycleRNN()
    elif sys.argv[1] == "ResNetLSTM":
        from models.rvenet.ResNetLSTM import ResNetLSTM
        model = ResNetLSTM()
    elif sys.argv[1] == "ResNextTemporal":
        from models.rvenet.ResNextTemporal import ResNextTemporal
        model = ResNextTemporal()
    else:
        fail = True
else:
    fail = True

if fail:
    print(f"Run as python3 {sys.argv[0]} <model_name>")
    print("<model_name>: CardiacCycleRNN / ResNetLSTM / ResNextTemporal")
    sys.exit(1)

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
    model_dir=os.path.join(environment['model_checkpoint_path'], model.__class__.__name__),
    train_data_transforms=None,
    val_data_transforms=None,
    batch_size=32,
    load_from_disk=load_model_from_disk,
    cuda=is_cuda,
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
    for (x, y) in trainer.val_loader:
        if trainer.cuda:
            x = x.cuda()
            y = y.cuda()

        predictions = trainer.model(x)

        predicted_values.extend(predictions.cpu().numpy())
        actual_values.extend(y.cpu().numpy())

actual_values = np.array(actual_values).squeeze()
predicted_values = np.array(predicted_values).squeeze()

# Show some values for comparison
show_values = 10
random_idx = np.random.randint(0, len(predicted_values), show_values, dtype=int)

import pandas as pd
print(pd.DataFrame({"Actual":actual_values[random_idx],"Predicted":predicted_values[random_idx]}))

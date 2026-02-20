from typing import Union

from models.rvenet.CardiacCycleRNN import CardiacCycleRNN
from models.rvenet.ResNetLSTM import ResNetLSTM
from models.rvenet.ResNextTemporal import ResNextTemporal
from models.rvenet.CardiacCycleTransformer import CardiacCycleTransformer

model_union = Union[CardiacCycleRNN, ResNetLSTM, ResNextTemporal,CardiacCycleTransformer]

# ECGenius

## Steps to run

1. Run `./setup_env.sh`
2. Make sure all dependencies in requirements.txt are installed

## File descriptions

`dicom_loader.py`: Implements a dataloader over torch.utils.data.Dataset for converting Echo-Cardiogram videos (*.dcm files) to torch tensors, passing them through preprocessing

`preprocessing.py`: Implements transforms over tensors read obtained from dcm files for remaining superfluous information

`requirements.txt`: Packages needed for running ECGenius

`results_generator.py`: Script version of python notebooks for running models for a longer duration

`runner.py`: Implements generic Trainer class to abstract out training and evaluation process

`setup_env.sh`: Creates symlinks for data and validation directories, creates environment.json for various system-specific parameters that would otherwise need to be hardcoded

`checkpoints/`: Saves model weights and loss history plots

`docs/`: Code for GitHub page for the project

`models/`: Contains code for all the models tested until now

- `models/auxiliary/`: Models related to the 2nd dataset (Heart failure clinical records) and insights from the RVENet dataset
  - `models/auxiliary/RandomForestRegressor.py`: Identify importance of age, sex, patient group and heart rate for prediction of RVEF
  - `models/auxiliary/text_preprocessing.py`: Preprocesses and removed outliers from heart failure clinical records dataset
  - `models/auxiliary/heuristic_concatenation.py`: Concatenates cardiovascular and RVENet dataset by bucketizing age groups in 0-100
  - `models/auxiliary/merge_datasets.py`: Tries pandas merge + iterative imputation to concatenate cardiovascular and RVENet dataset
  - `models/auxiliary/merge_dataset_knn.py`: Tries k-nearest neighbors to concatenate cardiovascular and RVENet dataset
  - `models/auxiliary/*.csv`: Corresponding merged dataset outputs
  - `models/auxiliary/get_summary_llm.py`: Get LLM summary for the patient personal merged from 2 datasets
- `models/rvenet/`: Deep neural networks implemented for the RVENet dataset for predicting RVEF (Right ventricle ejection fraction) from echo-cardiogram videos
  - `models/rvenet/all_models.py`: Saves union of all models implemented so that it doesn't need to be hardcoded
  - `models/rvenet/CardiacCycleRNN.py`: Resnet18 + LSTM + Linear layer + Feature Augmentation
  - `models/rvenet/CardiacCycleTransformer.py`: Resnet18 + Transformer Encoder + Linear layer
  - `models/rvenet/ResNetLSTM.py`: Resnet18 + LSTM + Linear layer with different loss function
  - `models/rvenet/ResNextTemporal.py`: Temporal CNN with ResNext as backbone

`notebooks/`: Cleaned up versions of notebooks used during experimentation in the project
`notebooks/feature_augmentation_benefits.ipynb`: Notebook to experiment if feature augmentation helps in CariacCycleRNN

`setup/`: Directory storing environment.json and containing python script for generating it

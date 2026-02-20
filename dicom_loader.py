"""
Script with Pytorch's dataloader class
"""

import glob
import os
from typing import Dict, List, Tuple

import torch
import torch.utils.data as data
import torchvision
from PIL import Image
import csv
import pandas as pd

from preprocessing import get_preprocessed_frames


class DicomLoader(data.Dataset):
    """Class for data loading"""

    def __init__(
        self,
        environment: dict,
        split: str = "train",
        transform: torchvision.transforms.Compose = None,
        deal_with_pt: bool = False,
    ):
        """Initialize the dataloader and set `curr_folder` for the corresponding data split.

        Args:
            root_dir: the dir path which contains the train and test folder
            split: 'test' or 'validation' split
            transform: the composed transforms to be applied to the data
        """
        self.transform = transform
        self.split = split
        self.deal_with_pt = deal_with_pt

        if split == "train":
            self.data_folder = environment["train_folder"]
        elif split == "validation":
            self.data_folder = environment["validation_folder"]
        else:
            raise ValueError(f"Invalid split: {split}")

        self.ground_truth_csv_path = environment["codebook_path"]
        self.dataset = self.load_dicom_paths_with_targets()

    def load_dicom_paths_with_targets(self) -> List[Tuple[str, float,float,float]]:
        """Fetches all (image path,label) pairs in the dataset.

        Args:
            class_labels: the class labels dictionary, with keys being the classes in this dataset
        Returns:
            list[(filepath, float)]: a list of filepaths and their RVEF values
        """

        df = pd.read_csv(self.ground_truth_csv_path)

        img_paths = []  # a list of (filename, target,)

        # See all dicom files in self.data_folder
        dicom_files = glob.glob(os.path.join(self.data_folder, "*.dcm"))

        # Get the corresponding target value for each dicom file
        for dicom_file in dicom_files:
            filename = os.path.basename(dicom_file)

            # remove extension
            filename = filename.split('.')[0]

            filename_series = df[df['FileName'] == filename]

            # assert only one match
            assert len(filename_series['RVEF'].values) == 1

            # verify no data leak
            assert filename_series['Split'].values[0] == self.split

            rvef_value = filename_series['RVEF'].values[0]
            age = filename_series['Age'].values[0]
            hr = filename_series['HR'].values[0]

            img_paths.append((dicom_file, rvef_value,age,hr))

        return img_paths

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Fetches the item (image, label) at a given index.

        Note: Do not forget to apply the transforms, if they exist

        Hint:
        1) get info from self.dataset
        2) use load_img_from_path
        3) apply transforms if valid

        Args:
            index: Index
        Returns:
            img: image of shape (H,W)
            class_idx: index of the ground truth class for this image
        """

        idx_data = self.dataset[index]
        target = torch.tensor([idx_data[1]], dtype=torch.float32)
        extra_features=torch.tensor([idx_data[2],idx_data[3]], dtype=torch.float32)
        pt_name = idx_data[0].split(".")[:-1]
        pt_name = "".join(pt_name) + ".pt"

        if os.path.exists(pt_name) and self.deal_with_pt:
            video_tensor = torch.load(pt_name,weights_only=True)
            assert video_tensor.shape == (20, 3, 224, 224)
            
            tensor_dict={"video_tensor":video_tensor,
                "extra_features":extra_features}
            return tensor_dict, target

        video_tensor = get_preprocessed_frames(idx_data[0])
        # keep only one heart cycle
        # video_tensor = video_tensor[0]

        # Heuristic: keep only the first heart cycle
        video_tensor = video_tensor[0]

        # convert to float tensor
        video_tensor = video_tensor.type(torch.float32)

        assert video_tensor.shape == (20, 3, 224, 224)

        if self.transform is not None:
            video_tensor = self.transform(video_tensor)

        if self.deal_with_pt:
            torch.save(video_tensor, pt_name)
            
        tensor_dict={"video_tensor":video_tensor,
                "extra_features":extra_features}

        return tensor_dict,target

    def __len__(self) -> int:
        """Returns the number of items in the dataset.

        Returns:
            l: length of the dataset
        """

        return len(self.dataset)

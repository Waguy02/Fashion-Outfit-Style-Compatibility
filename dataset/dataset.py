import json
import os
from unicodedata import category
import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm
from constants import DATA_DIR, DISJOINT_DATA_DIR, MAX_OUTFIT_SIZE

from my_utils import *
from enum import Enum


class DatasetType(Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"
    FITB="fitb"


class CustomDataset(Dataset):
    def __init__(self, type):
        self.type = type
        self.init_transforms()
        self.dataset_file = os.path.join(DISJOINT_DATA_DIR,
                                         "train.json" if self.type == DatasetType.TRAIN else "valid.json")
        self.load_data()

    def init_transforms(self):
        """
        Initialize transforms.Might be different for each dataset type
        """
        if self.type == DatasetType.TRAIN:
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def load_data(self):
        """
        Load data from the data items if necessary
        Returns:

        """
        all_datas = read_json(self.dataset_file)
        ##TODO Next : Use visual semantic embeddings to combine cnn features and text embedding from title
        self.datas = []
        for outfit in all_datas:
            outfit_data = []
            items = outfit["items"]
            for item in items:
                outfit_data.append(item["item_id"])

            self.datas.append(outfit_data)



    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        """
        How to retrieve item from the dataset
        Args:
            idx:
        Returns:
        """
        sample = self.datas[idx][:MAX_OUTFIT_SIZE]
        sample_data = [self.transforms(Image.open(get_item_image(item))) for item in sample]



        ##Add start and end token
        input_shape = sample_data[0].shape
        sample_data = [get_start_token(input_shape)] + sample_data + [get_end_token(input_shape)]

        ##PAD THE SEQUENCE IF NECESSERAY
        if len(sample_data) < MAX_OUTFIT_SIZE+2:
            sample_data += [torch.zeros(input_shape)] * (MAX_OUTFIT_SIZE+2 - len(sample_data))

        return torch.stack(sample_data)


class EvalFITBDataset(Dataset):
    """
    Compute the dataset for FITB evaluation task

    """

    def __init__(self,nb_proposals=3):
        self.type = DatasetType.FITB
        self.init_transforms()
        self.dataset_file = os.path.join(DISJOINT_DATA_DIR, "valid.json")

        self.load_data()
        self.nb_proposals=nb_proposals

    def init_transforms(self):
        """
        Initialize transforms.Might be different for each dataset type
        """
        if self.type == DatasetType.TRAIN:
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    def load_data(self):
        """
        Load data from the data items if necessary
        Returns:

        """
        all_datas = read_json(self.dataset_file)
        self.datas = []
        for outfit in all_datas:
            outfit_data = []
            items = outfit["items"]
            for item in items:
                outfit_data.append(item["item_id"])

            self.datas.append(outfit_data)

        self.items_categories = get_item_categories()
        self.all_items=list(self.items_categories.keys())

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        """
        How to retrieve item from the dataset
        Args:
            idx:
        Returns:
        """
        sample = self.datas[idx][:MAX_OUTFIT_SIZE]
        sample_data = [self.transforms(Image.open(get_item_image(item))) for item in sample]

        # ##PAD THE SEQUENCE IF NECESSERAY
        # if len(sample_data) < MAX_OUTFIT_SIZE:
        #     sample_data += [sample_data[-1]] * (MAX_OUTFIT_SIZE - len(sample_data))

        ##Add start and end token
        input_shape = sample_data[0].shape
        sample_data = [get_start_token(input_shape)] + sample_data + [get_end_token(input_shape)]

        ##Mask a position of the item in the outfit
        mask_idx = np.random.randint(len(sample))
        masked_item=sample[mask_idx]

        mask_idx+=1
        candidates=[item for item in self.all_items if item!=masked_item and self.items_categories[item]==\
                    self.items_categories[masked_item]]
        proposals=np.random.choice(candidates,self.nb_proposals,replace=False)
        proposals=list(proposals)+[masked_item]

        """
        Return the left part of the outfit, the masked item and the right part of the outfit, and the list of proposals
        """
        left=sample_data[:mask_idx]
        right=sample_data[mask_idx+1:]

        ##Pad left with start token
        left=[get_start_token(input_shape)]*(MAX_OUTFIT_SIZE+1-len(left))+left
        right=right+[get_end_token(input_shape)]*(MAX_OUTFIT_SIZE+1-len(right))

        masked_item_data=sample_data[mask_idx]
        proposals=torch.stack([self.transforms(Image.open(get_item_image(item))) for item in proposals])




        return {"left":torch.stack(left),"right":torch.stack(right),"masked_item":masked_item_data,"proposals":proposals,"mask_idx":mask_idx}



def collate_fn(batch):
    """
    Collate function to bad batchs
    """
    return batch


def collate_fn_fitb(batch):
    """
    Collate function to bad batchs
    """
    return tuple(zip(*batch))



def create_dataloader(type, batch_size=1, shuffle=False, num_workers=0):
    """
    Create dataloader for the dataset
    """
    if type!=DatasetType.FITB:
        dataset = CustomDataset(type)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    else :
        dataset = EvalFITBDataset()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


def get_start_token(input_shape):
    """
    Get start token for the input size
    """
    return torch.zeros(input_shape)


def get_end_token(input_shape):
    """
    Get end token for the input size
    """
    return torch.ones(input_shape)




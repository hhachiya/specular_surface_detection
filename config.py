import os
from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import ConcatDataset

from dataset import SSF_SH_Dataset


def parse_args():
    '''コマンドライン引数'''
    parser = ArgumentParser()
    parser.add_argument('-dataset_path', type=str, default="data/plastic_mold_dataset",help="dataset path") 
    parser.add_argument('-test_mold_type', type=str, default=None, help="test mold type.")
    parser.add_argument('-mode', type=str, choices=['rccl', 'ssf', 'sh', 'refine', 'pmd'], required=True, help="select learning network component.")
    parser.add_argument('-write_dir', type=str, required=True, help="output directory")
    # learning setting
    parser.add_argument('-epochs', type=int, default=150, help='defalut:150')
    parser.add_argument('-batch_size', type=int, default=5, help='Defalut:5')
    parser.add_argument('-resize', type=int, default=416 , help='Default:416')
    parser.add_argument("-patient", type=int, default=10, help="Early Stopping . the number of epoch. defalut 10")

    return parser.parse_args()


def getTrainTestCounts(dataset):
    train_size = int(dataset.__len__() * 0.8) 
    val_size   = dataset.__len__() - train_size
    return train_size, val_size


def mold_dataset(args):
    assert os.path.exists(args.dataset_path), f"Not found {args.dataset_path}"
    assert args.test_mold_type is not None, "No setting test mold type."
    print("mold dataset make:\n")
    
    train_dataset = []
    val_dataset = []
    
    for mold_type in os.listdir(args.dataset_path):
        mold_type_dir = os.path.join(args.dataset_path, mold_type)
        img_dir = os.path.join(mold_type_dir, "image")
        mask_dir = os.path.join(mold_type_dir, "mask")

        if args.test_mold_type == mold_type:
            test_dataset = SSF_SH_Dataset(img_dir, mask_dir)
            print(f"{mold_type} <- test: {len(test_dataset)}")
        else:
            dataset = SSF_SH_Dataset(img_dir, mask_dir)
            train_size, val_size = getTrainTestCounts(dataset)
            t_dataset, v_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
            train_dataset.append(t_dataset)
            val_dataset.append(v_dataset)
            print(f"{mold_type} <- learning, train: {train_size}, val: {val_size}")
    
    train_dataset = ConcatDataset(train_dataset)
    val_dataset = ConcatDataset(val_dataset)
    print(f"train: {len(train_dataset)}, valid: {len(val_dataset)}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0, pin_memory=True)
    
    return train_loader, val_loader, test_loader

def spherical_mirror_dataset(args):
    assert os.path.exists(args.dataset_path), f"Not found {args.dataset_path}"
    train_path = os.path.join(args.dataset_path, "train")
    test_path = os.path.join(args.dataset_path, "test")
    train_img_path = os.path.join(train_path, "image")
    train_mask_path = os.path.join(train_path, "mask")
    test_img_path = os.path.join(test_path, "image")
    test_mask_path = os.path.join(test_path, "mask")

    train_dataset = SSF_SH_Dataset(train_img_path, train_mask_path)
    train_size, val_size = getTrainTestCounts(train_dataset)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    test_dataset = SSF_SH_Dataset(test_img_path, test_mask_path)
    print(f"train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0, pin_memory=True)
    
    return train_loader, val_loader, test_loader
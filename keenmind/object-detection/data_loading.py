import glob
import random
import os
import warnings
import io
import logging

import boto3
from botocore.config import Config
import numpy as np
from PIL import Image
from PIL import ImageFile
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class ImageFolder(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.transform = transform

    def __getitem__(self, index):

        img_path = self.files[index % len(self.files)]
        img = np.array(
            Image.open(img_path).convert('RGB'), 
            dtype=np.uint8)

        # Label Placeholder
        boxes = np.zeros((1, 5))

        # Apply transforms
        if self.transform:
            img, _ = self.transform((img, boxes))

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    """
    Load images and labels from a text file.

    Each line in the text file should contain paths to images. Each image
    should be in a folder called images, and each image should have a
    corresponding label file located in a labels folder. The following
    gives an example of the required folder structure:

        |---images
        |       |---image1.jpg
        |       |---image2.jpg
        |
        |---labels
                |---label1.txt
                |---label2.txt

    Parameters
    ----------
    list_path: (str) Path to a file contating the locations of the image files
    img_size: (int) The size that the images should be resized to
    transform (torch.Sequential) A Sequential module of image transforms
    """
    def __init__(self, list_path, img_size=416, transform=None):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]

        self.img_size = img_size
        self.batch_count = 0
        self.transform = transform

    def __getitem__(self, index):

        try:
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception as e:
            logger.info(f"Could not read image '{img_path}'.")
            return

        try:
            label_path = self.label_files[index % len(self.img_files)].rstrip()
            boxes = np.loadtxt(label_path).reshape(-1, 5)
        except Exception as e:
            logger.info(f"Could not read label '{label_path}'.")
            return
        
        if self.transform:
            try:
                img, boxes = self.transform((img, boxes))
            except:
                logger.info(f"Could not apply transform.")
                return

        return img_path, img, boxes

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]
        paths, imgs, bb_targets = list(zip(*batch))

    
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)
        
        return paths, imgs, bb_targets

    def __len__(self):
        return len(self.img_files)


def download_s3_folder(bucket_name, s3_folder, local_dir=None):
    """
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)

if __name__ == "__main__":
    
    s3 = boto3.resource('s3')
    test_aws = download_s3_folder('keenmind-od-data', 'data/', local_dir='data/')
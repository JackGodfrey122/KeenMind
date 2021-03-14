import os
import io
import logging

import boto3
from botocore.config import Config
import numpy as np
from PIL import Image
from PIL import ImageFile
import torch
from torch.utils.data import Dataset

from utils import resize


ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


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

        # load image
        try:
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception as e:
            logger.warning(f"Could not read image '{img_path}'.")
            return

        # load label
        try:
            label_path = self.label_files[index % len(self.img_files)].rstrip()
            boxes = np.loadtxt(label_path).reshape(-1, 5)
        except Exception as e:
            logger.warning(f"Could not read label '{label_path}'.")
            return
        
        # do transforms
        try:
            img, boxes = self.transform((img, boxes))
        except:
            logger.warning(f"Could not apply transform.")
            return

        return img_path, img, boxes

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]
        paths, imgs, bb_targets = list(zip(*batch))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Add sample index to targets that will be used in NMS much later on
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

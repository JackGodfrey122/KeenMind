import os
import sys
import time
import datetime
import argparse
import logging
import yaml

from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import numpy as np
import random

from data_loading import ImageFolder
from transforms import DEFAULT_TRANSFORMS, Resize
from utils import non_max_suppression, rescale_boxes, parse_detections

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')


# load config
CONFIG_FILE = sys.argv[1]
with open(CONFIG_FILE, 'r') as stream:
    raw_config = yaml.safe_load(stream)
config = {k: v['value'] for k, v in raw_config.items()}
logger.info('Using {} as config file'.format(CONFIG_FILE))


# dataloading parameters
class_names = config['class_names']
img_size = config['img_size']
num_workers = config['num_workers']
loading_batch_size = config['loading_batch_size']
image_folder = config['image_folder']
pred_folder = config['pred_folder']

# NMS parameters
iou_thres = config['iou_threshold']
conf_thres = config['conf_threshold']
nms_thres = config['nms_threshold']


model_path = config['model_path']
model = torch.load(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info('Using device: {}'.format(str(device)))
model.eval()  # Set in evaluation mode


dataloader = DataLoader(
    ImageFolder(image_folder, transform= \
        transforms.Compose([DEFAULT_TRANSFORMS, Resize(img_size)])),
    batch_size=loading_batch_size,
    shuffle=False,
    num_workers=num_workers,
)

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
imgs = []  # Stores image paths
img_detections = []  # Stores detections for each image index

logger.info("\nPerforming object detection:")
prev_time = time.time()
for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    # Configure input
    input_imgs = Variable(input_imgs.type(Tensor))

    # Get detections
    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, conf_thres, nms_thres)
        detections = parse_detections(*detections, class_names)

    # Log progress
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    logger.info("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

    # Save image and detections
    imgs.extend(img_paths)
    img_detections.append(detections)
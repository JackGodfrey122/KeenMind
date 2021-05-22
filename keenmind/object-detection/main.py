import os
import sys
import time
import datetime
import argparse
import logging
import yaml
from typing import Optional
import io

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
from fastapi import FastAPI, File, UploadFile


from data_loading import ImageFolder
from transforms import DEFAULT_TRANSFORMS, Resize
from utils import non_max_suppression, rescale_boxes, parse_detections
import config as cf

app = FastAPI()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')

# load model
model_path = cf.settings.model_path
model = torch.load(model_path, map_location=torch.device('cpu'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info('Using device: {}'.format(str(device)))

TRANSFORMS = transforms.Compose([DEFAULT_TRANSFORMS, Resize(cf.settings.img_size)])

@app.post("/predict/")
def detect(img_file: UploadFile = File(...)):
    prev_time = time.time()
    input_img = np.array(Image.open(img_file.file).convert('RGB'), dtype=np.uint8)
    boxes = np.zeros((1, 5))  # place holder box so transforms code is reusable

    # transforms
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    input_img, _ = TRANSFORMS((input_img, boxes))
    input_img = input_img.unsqueeze(0)
    input_img = Variable(input_img.type(Tensor))
    
    model.eval()  # Set in evaluation mode
    logger.info("Performing object detection:")

    # Get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, cf.settings.conf_thres, cf.settings.nms_thres)
        detections = parse_detections(*detections, cf.settings.class_names)

    # Log progress
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    logger.info("Inference Time: %s" % (inference_time))

    # Save image and detections
    return detections
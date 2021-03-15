import time
import sys
import yaml
from datetime import datetime
import logging

import torch
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
import wandb

from model import YoloNetV3
from transforms import DEFAULT_TRANSFORMS
from data_loading import ListDataset
from evaluate import evaluate, METRICS


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')


# load config
# CONFIG_FILE = sys.argv[1]
CONFIG_FILE = '/home/jack/keenmind/keenmind/object-detection/configs/config.yaml'
with open(CONFIG_FILE, 'r') as stream:
    raw_config = yaml.safe_load(stream)
config = {k: v['value'] for k, v in raw_config.items()}
logger.info('Using {} as config file'.format(CONFIG_FILE))


# dataloading parameters
train_path = config['train_path']
val_path = config['val_path']
class_names = config['class_names']
img_size = config['img_size']
num_classes = config['num_classes']
num_workers = config['num_workers']
loading_batch_size = config['loading_batch_size']

# training parameters
epochs = config['epochs']
effective_batch_size = config['effective_batch_size']
eval_interval = config['eval_interval']

# NMS parameters
iou_thres = config['iou_threshold']
conf_thres = config['conf_threshold']
nms_thres = config['nms_threshold']

# wandb parameters
wandb_project = config['wandb_project']
wandb_entity = config['wandb_entity']

model_name = config['model_path'] + 'keenmind-od-' + str(datetime.now())

# initialise wandb
run = wandb.init(project=wandb_project, entity=wandb_entity, config=config)
logger.info('Tracking wandb run at {}:{}'.format(wandb_entity, wandb_project))

train_dataset = ListDataset(train_path, img_size=img_size, transform=DEFAULT_TRANSFORMS)
val_dataset = ListDataset(val_path, img_size=img_size, transform=DEFAULT_TRANSFORMS)


train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=loading_batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=train_dataset.collate_fn,
    )


val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=loading_batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=val_dataset.collate_fn,
    )


model = YoloNetV3(num_classes, img_size)
model.cuda()

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
optimizer = torch.optim.Adam(model.parameters())
device = "cuda"

total_batches = len(train_dataset) // loading_batch_size
logging.info('Total batches : {}'.format(total_batches))

logging.info('Starting model training')
for epoch in range(epochs):
    model.train()
    start_time = time.time()
    for batch_ti, (_, imgs, targets) in enumerate(train_dataloader):
        batches_done = batch_ti * epoch
        
        imgs = torch.Tensor(imgs).to(device)
        targets = torch.Tensor(targets).to(device)

        log_string = "Epoch: {} | Batch {} / {} ".format(epoch, batch_ti, total_batches)
        logger.info(log_string)

        outputs, loss, metrics = model(imgs, targets)
        loss.backward()

        if batches_done % effective_batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # logging
        metrics.update({'total loss': loss.data})
        wandb.log(metrics)

    # evaluating
    if epoch % eval_interval == 0:
        logging.info('Evaluating after epoch {}'.format(epoch))

        eval_metrics = evaluate(
            model,
            val_dataloader,
            iou_thres,
            conf_thres,
            nms_thres,
            img_size,
            loading_batch_size)
        
        if eval_metrics is not None:
            eval_ap_metrics = {}
            precision, recall, AP, f1, ap_class = eval_metrics
            for i, c in enumerate(ap_class):
                name_key = 'validation/' + class_names[i] + '_AP'
                eval_ap_metrics.update({name_key: ap_class})
            wandb.log(eval_ap_metrics)
            wandb.log({
            "validation/precision": precision.mean(),
            "validation/recall": recall.mean(),
            "validation/mAP": AP.mean(),
            "validation/f1": f1.mean()
            })

        logging.info('Saving model to {}'.format(model_name))  
        torch.save(model, model_name)

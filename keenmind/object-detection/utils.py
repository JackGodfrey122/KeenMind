import torch
import numpy as np


def xywh2xyxy_np(x):
    """
    Transforms center coords and width/height to (x1, y1) (x2, y2) coords.

    (x1, y1) is upper left coord of box and (x2, y2) is lower right of box.

    Parameters
    ----------
    x : (torch.Tensor) Torch Tensor where the indices are as follows:
        0: center x value
        1: center y value
        2: width of box
        3: height of box

    Returns
    -------
    (numpy array) Numpy array with dims (nB, 4)
    """
    out = np.zeros_like(x)
    out[..., 0] = x[..., 0] - x[..., 2] / 2
    out[..., 1] = x[..., 1] - x[..., 3] / 2
    out[..., 2] = x[..., 0] + x[..., 2] / 2
    out[..., 3] = x[..., 1] + x[..., 3] / 2
    return out


def to_cpu(tensor):
    """
    Moves CUDA Tensor to cpu

    Parameters
    ----------
    tensor : (torch.Tensor)

    Returns
    -------
    (torch.Tensor)
    
    """
    return tensor.detach().cpu()


def bbox_wh_iou(wh1, wh2):
    """
    Calculates the IoU between two bounding boxes parameterized by width and
    height.

    Parameters
    ----------
    wh1 : (torch.Tensor) Bounding box 1
    wh2 : (torch.Tensor) Bounding box 2

    Returns
    -------
    (torch.Tensor) Tensor containing IoU score
    """
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes. By default, these boxes are
    exprected to be parameterized by upper left and lower right xy coords.

    Parameters
    ----------
    box1: (torch.Tensor) Bounding box 1
    box2: (torch.Tensor) Bounding box 2
    x1y1x2y2 (bool): If False, assumes the boxes are parameterized by
    center,width,height coords, and converts to exact coords before
    proceeding with IoU calculation.

    Returns
    -------
    (torch.Tensor) Tensor containing IoU score
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    """
    Builds scaled targets given raw input targets.

    Parameters
    ----------
    pred_boxes: (torch.Tensor) Tensor of dims (nB, nA, nG, nG, 4). This
        represents the bounding box prediction for each cell per anchor
        per image in batch.
    pred_cls: (torch.Tensor) Tensor of dims (nB, nA, nG, nG, nC). This
        represents the class prediction for each cell per anchor per image
        in batch.
    target: (torch.Tensor) Tensor of dims (nB, 6). These are the raw targets
        that were inputted into the model. 6 Comes from the following:
            - batch index
            - cls index
            - bbox coords parameterized by upper left and lower right xy
                coords
    anchors: (torch.Tensor) Tensor of dims (3, 2). 3 anchors per scale, and
        each anchor is parameterized by 2 values (unsure what these represent)
    ignore_thres: (float) If bbox IoU is below this value, don't consider a
        valid prediction.

    Returns
    -------
    iou_scores: (torch.Tensor) Tensor of dims (nB, nA, nG, nG). IoU for
        each target and bounding box combination.
    class_mask: (torch.Tensor) Tensor of dims (nB, nA, nG, nG). Used to
        mask class predictions.
    obj_mask: (torch.Tensor) Tensor of dims (nB, nA, nG, nG). Used to
        mask predictions which contains an object.
    noobj_mask: (torch.Tensor) Tensor of dims (nB, nA, nG, nG). Used to
        mask predictions which do not contain an object.
    tx: (torch.Tensor) Tensor of dims (nB, nA, nG, nG). Predictions for
        center x value.
    ty: (torch.Tensor) Tensor of dims (nB, nA, nG, nG). Predictions for
        center y value.
    tw: (torch.Tensor) Tensor of dims (nB, nA, nG, nG). Predictions for
        center width of bounding box.
    th: (torch.Tensor) Tensor of dims (nB, nA, nG, nG). Predictions for
        center height of bounding box.
    tcls: (torch.Tensor) Tensor of dims (nB, nA, nG, nG, nC). Used to
        store class predictions.
    tconf: (torch.Tensor) Tensor of dims (nB, nA, nG, nG). Represents
        if an object is present.
    """

    BoolTensor = torch.cuda.BoolTensor if pred_boxes.is_cuda else torch.BoolTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)  # batch size
    nA = pred_boxes.size(1)  # num anchors per scale
    nC = pred_cls.size(-1)   # num classes
    nG = pred_boxes.size(2)  # grid size

    # Output tensors
    obj_mask = BoolTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = BoolTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # Convert to position relative to box

    # target has original  dim of (nB, 6). 6 comes from index, cls, x, y, w, h
    # multiply by grid size to account for upscaling
    # WE ARE UPSCALING THE ORIGINAL TARGET, NOT THE PREDICTION
    # WE UNSCALE LATER ON see the lines with .floor() in them
    target_boxes = target[:, 2:6] * nG  # dim is (nB, 4) 4 comes from x, y, w, h
    gxy = target_boxes[:, :2]  # scaled anchor x, y coords
    gwh = target_boxes[:, 2:]  # scaled anchor w, h values

    # Get anchors with best iou
    # anchors here are just the scaled anchors from the yolo layer
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])  # dim is (3, nB)
    
    # best n is the anchor index
    best_ious, best_n = ious.max(0)  # one for each image in batch
    # Separate target values

    # we transpose all of the below to make concat easier
    b, target_labels = target[:, :2].long().t()  # batch index and target cls index
    gx, gy = gxy.t()  # scaled x and y
    gw, gh = gwh.t()  # scaled w and h
    gi, gj = gxy.long().t()  # scaled indexes of grid cell
    
    # For each image in batch, for best anchor, for best grid cell, there is a cls there
    obj_mask[b, best_n, gj, gi] = 1

    # same as above but set to 0 for no object
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    # this is a bit more filtering for low iou scores
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()  # removes the scale factor
    ty[b, best_n, gj, gi] = gy - gy.floor() # removes the scale factor
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)  # makes learning more stable
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)  # makes learning more stable
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1

    # Compute label correctness and iou at best anchor
    # used for metric logging
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf

import torch
import torch.nn.functional as F
import numpy as np


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def xywh2xyxy(in_tensor):
    """
    Given a Tensor with dimensions (nB, N, nP) where:
        - nB: batch size
        - N: Number of predictions
        - nP: Number of predictables (usually 5 + number of classes)
    and assuming that the center x, center y, width and height occupy the
    first 4 indexes of nP, then this function will convert those first 4
    indexes to x1, y1, x2, y2, where (x1, y1) represent the upper left
    corner of the bounding box, and (x2, y2) represent the lower right
    corner of the box.

    Parameters
    ----------
    in_tensor : (torch.Tensor) Usually the output of a YoloV3 model

    Returns
    -------
    (torch.Tensor) A copy of in_tensor with modified coordinate values
    """
    out_tensor = in_tensor.new(in_tensor.shape)
    out_tensor[..., 0] = in_tensor[..., 0] - in_tensor[..., 2] / 2
    out_tensor[..., 1] = in_tensor[..., 1] - in_tensor[..., 3] / 2
    out_tensor[..., 2] = in_tensor[..., 0] + in_tensor[..., 2] / 2
    out_tensor[..., 3] = in_tensor[..., 1] + in_tensor[..., 3] / 2
    return out_tensor


def xywh2xyxy_np(in_tensor):
    """
    Given a Tensor with dimensions (nB, N, nP) where:
        - nB: batch size
        - N: Number of predictions
        - nP: Number of predictables (usually 5 + number of classes)
    and assuming that the center x, center y, width and height occupy the
    first 4 indexes of nP, then this function will convert those first 4
    indexes to x1, y1, x2, y2, where (x1, y1) represent the upper left
    corner of the bounding box, and (x2, y2) represent the lower right
    corner of the box, and return as a numpy array.

    Parameters
    ----------
    in_tensor : (torch.Tensor) Usually the output of a YoloV3 model

    Returns
    -------
    (numpy.Array) A numpy array with modified coordinate values
    """
    out = np.zeros_like(in_tensor)
    out[..., 0] = in_tensor[..., 0] - in_tensor[..., 2] / 2
    out[..., 1] = in_tensor[..., 1] - in_tensor[..., 3] / 2
    out[..., 2] = in_tensor[..., 0] + in_tensor[..., 2] / 2
    out[..., 3] = in_tensor[..., 1] + in_tensor[..., 3] / 2
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


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Applies Non-Max Supression to bounding box predictions.

    It is assumed that the input is a Tensor with dimensions (nB, N, nP)
    where:
        - nB: batch size
        - N: Number of predictions
        - nP: Number of predictables (usually 5 + number of classes)
    and that the center x, center y, width and height occupy the
    first 4 indexes of nP.

    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction): # for each image in batch
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres  # keep boxes with large overlap
            label_match = detections[0, -1] == detections[:, -1]  # track predictions with correct label predictions
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output


def get_batch_statistics(outputs, targets, iou_threshold):
    """
    Compute true positives, predicted scores and predicted labels per sample

    Parameters
    ----------
    outputs : (list) list of tensors which have been processed by non max
        supression. Tensors will be dim (preds_after_NMS, num_class)
    targets : (torch.Tensor) Tensor of targets from the DataLoader
    iou_threshold : (float) Between 0-1

    Returns
    -------
    (list) List of lists which contain:
        [true positives, prediction scores, prediction labels]
        - true postives (int)
        - prediction scores (torch.Tensor) Dimension == number of predictions
            in output
        - prediction labels (torch.Tensor) Dimension == number of predictions
            in output
    """
    batch_metrics = []
    for sample_i in range(len(outputs)):  # for each prediction

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]  # get output

        # unpack all
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        # init tp array of size == number of predictions after nms
        true_positives = np.zeros(pred_boxes.shape[0])

        # get predictables for target
        annotations = targets[targets[:, 0] == sample_i][:, 1:]

        # get label from target
        target_labels = annotations[:, 0] if len(annotations) else []

        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]  # get bbox predictables

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue
                
                # get IoU
                iou, box_index = bbox_iou(
                    pred_box.unsqueeze(0),
                    target_boxes).max(0)

                # TP if above IoU thresh and has not been already predicted
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]

        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def ap_per_class(tp, conf, pred_cls, target_cls):
    """
    Compute the average precision, given the recall and precision curves.
    
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    
    Parameters
    ----------
    tp: (list) True positives
    conf: (list) Objectness value from 0-1
    pred_cls: (list) Predicted object classes
    target_cls: (list) True object classes
    
    Returns
    -------
    (tuple) Tuple of numpy array representing:
        - precession
        - recall
        - average precision
        - f1 score
    """
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """
    Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    
    Parameters
    ----------
    recall: () The recall curve (list).
    precision: ()The precision curve (list).
    
    Returns
    -------
    () The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
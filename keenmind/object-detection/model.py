import torch
import torch.nn as nn
from torch.nn import MSELoss, BCELoss
import torch.nn.functional as F

from utils import build_targets, to_cpu


class ConvLayer(nn.Module):
    """
    Mostly generalised convolutional block with batchnorm and a leakyrelu
    activation.

    Note taken from Conv2d docs (https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html):
    Depending of the size of your kernel, several (of the last) columns of the
    input might be lost, because it is a valid cross-correlation, and not a
    full cross-correlation. It is up to the user to add proper padding.

    yolov3 uses size 3 kernels throughout, resulting in padding that produces
    dimensions that are .5 above the desired dimensions. However, rounding
    down solves this issue. This note was more for my sanity than having
    any real value, and will probably be removed later.

    Parameters
    ----------
    in_channels : (int) Number of channels in the input tensor
    out_channels : (int) Number of channels produced by the convolution
    kernel_size : (int) Size of the convolving kernel
    stride : (int) Stride of the convolution
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            activation='leaky'):

        super().__init__()
        # this handles the padding for the case where the input is halved and
        # maintained
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False)

        # batch norm
        self.bn = nn.BatchNorm2d(out_channels)

        # activation
        if activation == 'leaky':
            self.activation = nn.LeakyReLU()

        if activation == 'relu':
            self.activation = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)

        return out


class ResBlock(nn.Module):
    """
    Block to represent a Residual Block from yolov3.

    This block consists of two convolutional layers, followed by a residual
    connection.

    Parameters
    ----------
    in_channels : (int) Number of channels in the input tensor
    """
    def __init__(self, in_channels):
        super().__init__()

        # The below combination of out_channels, stride and padding forces the
        # output tensor to have the same dimensions as the input tensor
        out_channels = in_channels // 2
        self.conv1 = ConvLayer(in_channels, out_channels, 1, 1)
        self.conv2 = ConvLayer(out_channels, in_channels, 3, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        # Residual connection here
        out += x

        return out


class StackedResBlock(nn.Module):
    """
    Block to represent a Stacked Residual Block from yolov3.

    Parameters
    ----------
    in_channels : (int) Number of channels in the input tensor
    num_repeat : (int) Number of blocks to stack
    """
    def __init__(self, in_channels, num_repeat):
        super().__init__()

        # initialize sequential model
        self.repeated_block = nn.Sequential()

        for i in range(num_repeat):
            self.repeated_block.add_module(
                'res_block_{}'.format(i), ResBlock(in_channels))

    def forward(self, x):
        out = self.repeated_block(x)
        return out


class FeatureExtractor(nn.Module):
    """
    The FeatureExtractor network that does all of the feature extraction.

    Outputs at three different locations:
    
        1. After the third stacked residual block
        2. After the fourth stacked residual block
        3. After the fifth stacked residual block

    These will then be fed into a smaller network which handles the
    classification tasks. Please see PredictionNetwork and YoloLayer for details
    on that network.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = ConvLayer(3, 32, 3, 1)
        self.conv2 = ConvLayer(32, 64, 3, 2)
        self.res_block_1 = StackedResBlock(64, 1)

        self.conv3 = ConvLayer(64, 128, 3, 2)
        self.res_block_2 = StackedResBlock(128, 2)

        self.conv4 = ConvLayer(128, 256, 3, 2)
        self.res_block_3 = StackedResBlock(256, 8)

        self.conv5 = ConvLayer(256, 512, 3, 2)
        self.res_block_4 = StackedResBlock(512, 8)

        self.conv6 = ConvLayer(512, 1024, 3, 2)
        self.res_block_5 = StackedResBlock(1024, 4)

    def forward(self, x):
        tmp = self.conv1(x)
        tmp = self.conv2(tmp)
        tmp = self.res_block_1(tmp)
        tmp = self.conv3(tmp)
        tmp = self.res_block_2(tmp)

        # first output
        tmp = self.conv4(tmp)
        out1 = self.res_block_3(tmp)

        # second output
        out2 = self.conv5(out1)
        out2 = self.res_block_4(out2)

        # third output
        out3 = self.conv6(out2)
        out3 = self.res_block_5(out3)

        return out3, out2, out1


class YoloLayer(nn.Module):
    """
    The layer responsible for outputting the predictions.

    Code here is heavily inspired by the following repo:
        
        https://github.com/eriklindernoren/PyTorch-YOLOv3

    This layer is complex, and is best viewed alongside utils.build_targets 
    in order to interpret the output of the forward method.

    Note on dimension sizes:
        - nB: Batch size
        - nA: Anchors per scale (3 throughout this implementation)
        - nG: Grid size (usually 13, 26 and 52 for the 3 yolo layers)
        - nC: Number of classes in the dataset

    Parameters
    ----------
    anchors: (list) List of tuples representing anchors. Each tuple will
        contain 2 ints, representing the width and height of the anchor box
        respectively. Example: [(10, 13), (16, 30), (33, 23)]
    nC: (int) Number of the classes in the dataset
    img_size: (int) Width and height (in pixels) of the input image. This
        network assumes square images.
    """
    def __init__(self, anchors, nC, img_size):
        super().__init__()
        self.anchors = anchors
        self.nA = len(anchors)
        self.nC = nC
        self.nG = 0  # grid size
        self.img_size = img_size
        self.ignore_thres = 0.5  # IoU threshold 

        # weights for object and no object
        self.obj_scale = 1
        self.noobj_scale = 100

        # construct loss functions
        self.mse_loss = MSELoss()
        self.bce_loss = BCELoss()

    def compute_grid_offsets(self, nG, cuda=True):
        self.nG = nG
        g = self.nG
        self.stride = self.img_size / self.nG
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.nA, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.nA, 1, 1))

    def forward(self, x, targets=None, img_size=None):
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        self.img_size = img_size
        nB = x.size(0)  # batch size
        nG = x.size(2)  # size of scaled grid

        # reshape raw input to match that of the expected output
        prediction = (x.view(nB, self.nA, self.nC + 5, nG, nG)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Unpack predictables from reshaped input and apply sigmoid to needed outputs
        x = torch.sigmoid(prediction[..., 0])          # (nB, nA, nG, nG)
        y = torch.sigmoid(prediction[..., 1])          # (nB, nA, nG, nG)
        w = prediction[..., 2]                         # (nB, nA, nG, nG)
        h = prediction[..., 3]                         # (nB, nA, nG, nG)
        pred_conf = torch.sigmoid(prediction[..., 4])  # (nB, nA, nG, nG)
        pred_cls = torch.sigmoid(prediction[..., 5:])  # (nB, nA, nG, nG, nC)

        # If grid size does not match current we compute new offsets
        if nG != self.nG:
            self.compute_grid_offsets(nG, cuda=x.is_cuda)

        # Add offset to x and y, and scale w and h
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x                 # x
        pred_boxes[..., 1] = y.data + self.grid_y                 # y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w    # w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h    # h

        output = torch.cat(
            (
                pred_boxes.view(nB, -1, 4) * self.stride,
                pred_conf.view(nB, -1, 1),
                pred_cls.view(nB, -1, self.nC),
            ),
            -1,
        )

        if targets is None:
            return output, 0, 0
        
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "train/loss": to_cpu(total_loss).item(),
                "train/x": to_cpu(loss_x).item(),
                "train/y": to_cpu(loss_y).item(),
                "train/w": to_cpu(loss_w).item(),
                "train/h": to_cpu(loss_h).item(),
                "train/conf": to_cpu(loss_conf).item(),
                "train/cls": to_cpu(loss_cls).item(),
                "train/cls_acc": to_cpu(cls_acc).item(),
                "train/recall50": to_cpu(recall50).item(),
                "train/recall75": to_cpu(recall75).item(),
                "train/precision": to_cpu(precision).item(),
                "train/conf_obj": to_cpu(conf_obj).item(),
                "train/conf_noobj": to_cpu(conf_noobj).item()
            }

            return output, total_loss, self.metrics


class PredictionNetwork(nn.Module):
    """
    This network is responsible for making the final predictions.

    It contains various convolutional layers and a final yolo layer. This
    layer outputs the following:
        branch: A tensor which will be fed into the next PredictionNetwork
        yolo_output: A tensor containing the predictions from this scale
        processed_targets: A tensor containing the processed tensors. This and
            the yolo outputs will be fed into the loss function.

    The passage below is taken from the original paper (https://arxiv.org/pdf/1804.02767.pdf):

        From our base feature extractor we add several convolutional layers.
        The last of these predicts a 3-d tensor encoding bounding box,
        objectness,  and  class  predictions.

    Other implementations seem to use 7 convolutional layers here, so we do
    the same.

    Parameters
    ----------
    in_channels : (int) Number of channels in the input tensor
    out_channels : (int) Number of channels produced by the convolution
    last_layer_dim: (int) Number of dimensions in the last layer. This will
        be equal to the product of:
            - nA (number of anchors per scale) (usually this is 3)
            - nC+5. 5 comes from w, y, w, h and obj_conf
        For example, for 80 classes, last_layer_dim will be 255
    anchors: (list) List of tuples representing anchors. Each tuple will
        contain 2 ints, representing the width and height of the anchor box
        respectively. Example: [(10, 13), (16, 30), (33, 23)]
    nC: (int) Number of the classes in the dataset
    img_size: (int) Width and height (in pixels) of the input image. This
        network assumes square images.
    """
    def __init__(self, in_channels, out_channels, last_layer_dim, anchors, nC, img_size):
        super().__init__()
        self.img_size = img_size
        half_out_channels = out_channels // 2

        # bunch of convolutional layers to do prediction
        self.conv1 = ConvLayer(in_channels, half_out_channels, 1, 1)
        self.conv2 = ConvLayer(half_out_channels, out_channels, 3, 1)
        self.conv3 = ConvLayer(out_channels, half_out_channels, 1, 1)
        self.conv4 = ConvLayer(half_out_channels, out_channels, 3, 1)
        self.conv5 = ConvLayer(out_channels, half_out_channels, 1, 1)
        self.conv6 = ConvLayer(half_out_channels, out_channels, 3, 1)
        self.conv7 = nn.Conv2d(out_channels, last_layer_dim, 1, bias=True)
        self.yolo = YoloLayer(anchors, nC, img_size)

    def forward(self, x, targets=None):
        tmp = self.conv1(x)
        tmp = self.conv2(tmp)
        tmp = self.conv3(tmp)
        tmp = self.conv4(tmp)

        # branch which will be used in the next prediction network
        branch = self.conv5(tmp)

        out = self.conv6(branch)
        out = self.conv7(out)

        if targets is None:
            yolo_out, _, _ = self.yolo(out, targets, img_size=self.img_size)
            return branch, yolo_out, _, _
        
        else:
            yolo_out, loss, metrics = self.yolo(out, targets, img_size=self.img_size)
            return branch, yolo_out, loss, metrics


class YoloNeck(nn.Module):
    """
    Neck of the Yolov3 Network. This connects the feature extractor to the
    prediction network.

    Given an input tensor of (1, 3, 512, 512), the dimensions of the outputs
    of the feature extractor will be:
    
        Tensor 1. (1, 1024, 8, 8)
        Tensor 2. (1, 512, 16, 16)
        Tensor 3. (1, 256, 32, 32)
    
    The above tensors are the inputs to this Neck. Below details the flow
    of these tensors.

    Tensor 1: Passes through the first prediction network, where the dimensions
    are converted from (1, 1024, 8, 8) -> (1, 192, 85)

    Tensor 2: The tensor outputted from the 2nd to last layer from the
    first prediction network is passed through an additional convolutional
    layer, converting the dimensions from (1, 512, 8, 8) -> (1, 256, 8, 8).
    This tensor is then interpolated to (1, 256, 16, 16). This is then
    concatanted with Tensor 2, to give a tensor with dimension (1, 768, 16, 16).
    Finally, this tensor is then passed through the second prediction network
    to output a tensor of dimension (1, 768, 85).

    Tensor 3: The tensor outputted from the 2nd to last layer from the
    second prediction network is passed through an additional convolutional
    layer, converting the dimensions from (1, 256, 16, 16) -> (1, 128, 16, 16).
    This tensor is then interpolated to (1, 128, 32, 32). This is then
    concatanted with Tensor 3, to give a tensor with dimension (1, 384, 32, 32).
    Finally, this tensor is then passed through the third prediction network
    to output a tensor of dimension (1, 3072, 85).

    Note: Interpolation only affects the feature map dimensions, not the number
    of feature maps. i.e. the last 2 dimensions of a tensor.

    Note: The dimesions given into the layers below may seem arbitrary, however
    they are independent from the input image dimensions. They are dependant on
    which layers are used to concatenate with the output of the feature
    extractor network.

    Parameters
    ----------
    nC: (int) Number of the classes in the dataset
    img_size: (int) Width and height (in pixels) of the input image. This
        network assumes square images.
    
    """
    def __init__(self, nC, img_size):
        super().__init__()

        # anchors for all scales
        self.s_anchors = torch.tensor([(10, 13), (16, 30), (33, 23)]) 
        self.m_anchors = torch.tensor([(30, 61), (62, 45), (59, 119)])
        self.l_anchors = torch.tensor([(116, 90), (156, 198), (373, 326)])

        # calculate last layer dim
        # (x, y, width, height) + (obj) + (class probabilities)
        num_predictables = 5 + nC

        # 3 box predictions per image
        self.last_layer_dim = 3 * num_predictables

        # this network acts on the earliest output from the feature extractor network, which outputs a tensor with 1024 filters
        self.detect1  = PredictionNetwork(1024, 1024, self.last_layer_dim, self.l_anchors, nC, img_size)
        
        self.conv1 = ConvLayer(512, 256, 1, 1) 
        # this network acts on the middle output from the feature extractor network, which outputs a tensor with 512 filters
        self.detect2 = PredictionNetwork(768, 512, self.last_layer_dim, self.m_anchors, nC, img_size)
        
        self.conv2 = ConvLayer(256, 128, 1, 1)
        # this network acts on the earliest output from the feature extractor network, which outputs a tensor with 256 filters
        self.detect3 = PredictionNetwork(384, 256, self.last_layer_dim, self.s_anchors, nC, img_size)  

    def forward(self, x1, x2, x3, targets=None):

        # prediction 1
        branch1, out1, l1, metrics1 = self.detect1(x1, targets)

        # prediction 2
        tmp = self.conv1(branch1)
        tmp = F.interpolate(tmp, scale_factor=2)  # upscales the branch by 2
        tmp = torch.cat((tmp, x2), 1)
        branch2, out2, l2, metrics2  = self.detect2(tmp, targets)

        # prediction 3
        tmp = self.conv2(branch2)
        tmp = F.interpolate(tmp, scale_factor=2)  # upscales the branch by 2
        tmp = torch.cat((tmp, x3), 1)
        _, out3, l3, metrics3 = self.detect3(tmp, targets) # branch is not needed here

        total_loss = l1 + l2 + l3
        yolo_outputs = to_cpu(torch.cat([out1, out2, out3], 1))
        if targets is None:
            return yolo_outputs
        
        else:
            metrics = {
                'Yolo Layer 1': metrics1,
                'Yolo Layer 2': metrics2,
                'Yolo Layer 3': metrics3
            }
            return (yolo_outputs, total_loss, metrics)


class YoloNetV3(nn.Module):

    def __init__(self, nC, img_size):
        super().__init__()
        self.darknet = FeatureExtractor()
        self.yolo_tail = YoloNeck(nC, img_size)

    def forward(self, x, targets=None):
        tmp1, tmp2, tmp3 = self.darknet(x)
        out = self.yolo_tail(tmp1, tmp2, tmp3, targets)
        return out

if __name__ == "__main__":

    img_size = 416
    num_classes = 80
    test = torch.rand(2, 3, img_size, img_size)
    label = torch.tensor([[0.000, 6.000, 0.546, 0.654, 0.123, 0.111], [0.000, 6.000, 0.546, 0.654, 0.123, 0.111]])
    yolo = YoloNetV3(num_classes, img_size)
    out = yolo(test)

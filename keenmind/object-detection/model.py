import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import build_targets


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
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_size):
        super().__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_size = img_size
        self.grid_size = 0  # grid size

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_size / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_size=None):

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        self.img_size = img_size
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None:
            return output, 0
        
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(pred_boxes, pred_cls, targets, self.scaled_anchors, self.ignore_thres)
            return output, (iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf)


class PredictionNetwork(nn.Module):
    """
    This network is responsible for making the final predictions.

    Whilst it does only explicitly return a single output, note the
    self.branch attribute stemming from the 5th convolutional layer. This
    is later utilized in the YoloNeck module.

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
    scale: (str) Can be 's', 'm' or 'l'. These represent the scale in which
        the prediction is being made. Ultimately, it decides which anchors to
        use in the Yolo Layer. Please see YoloLayer for more details.
    stride: (int) 
    """
    def __init__(self, in_channels, out_channels, last_layer_dim, anchors, num_classes, img_size):
        super().__init__()
        self.img_size = img_size
        half_out_channels = out_channels // 2
        self.conv1 = ConvLayer(in_channels, half_out_channels, 1, 1)
        self.conv2 = ConvLayer(half_out_channels, out_channels, 3, 1)
        self.conv3 = ConvLayer(out_channels, half_out_channels, 1, 1)
        self.conv4 = ConvLayer(half_out_channels, out_channels, 3, 1)
        self.conv5 = ConvLayer(out_channels, half_out_channels, 1, 1)
        self.conv6 = ConvLayer(half_out_channels, out_channels, 3, 1)
        self.conv7 = nn.Conv2d(out_channels, last_layer_dim, 1, bias=True)
        self.yolo = YoloLayer(anchors, num_classes, img_size)

    def forward(self, x, targets=None):
        tmp = self.conv1(x)
        tmp = self.conv2(tmp)
        tmp = self.conv3(tmp)
        tmp = self.conv4(tmp)

        # branch which will be used in the YoloNeck module
        branch = self.conv5(tmp)

        out = self.conv6(branch)
        out = self.conv7(out)

        if targets is None:
            out, _ = self.yolo(out, targets, img_size=self.img_size)
            return branch, out, _
        
        else:
            out, loss = self.yolo(out, targets, img_size=self.img_size)
            return branch, out, loss


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
    """
    def __init__(self, num_classes, img_size):
        super().__init__()

        # anchors for all scales
        self.s_anchors = torch.tensor([(10, 13), (16, 30), (33, 23)]) 
        self.m_anchors = torch.tensor([(30, 61), (62, 45), (59, 119)])
        self.l_anchors = torch.tensor([(116, 90), (156, 198), (373, 326)])

        # calculate last layer dim
        # (x, y, width, height) + (obj) + (class probabilities)
        num_predictables = 5 + num_classes

        # 3 box predictions per image
        self.last_layer_dim = 3 * num_predictables

        # this network acts on the earliest output from the feature extractor network, which outputs a tensor with 1024 filters
        self.detect1  = PredictionNetwork(1024, 1024, self.last_layer_dim, self.l_anchors, num_classes, img_size)
        
        self.conv1 = ConvLayer(512, 256, 1, 1) 
        # this network acts on the middle output from the feature extractor network, which outputs a tensor with 512 filters
        self.detect2 = PredictionNetwork(768, 512, self.last_layer_dim, self.m_anchors, num_classes, img_size)
        
        self.conv2 = ConvLayer(256, 128, 1, 1)
        # this network acts on the earliest output from the feature extractor network, which outputs a tensor with 256 filters
        self.detect3 = PredictionNetwork(384, 256, self.last_layer_dim, self.s_anchors, num_classes, img_size)  

    def forward(self, x1, x2, x3, targets=None):

        # prediction 1
        branch1, out1, l1 = self.detect1(x1, targets)

        # prediction 2
        tmp = self.conv1(branch1)
        tmp = F.interpolate(tmp, scale_factor=2)  # upscales the branch by 2
        tmp = torch.cat((tmp, x2), 1)
        branch2, out2, l2 = self.detect2(tmp, targets)

        # prediction 3
        tmp = self.conv2(branch2)
        tmp = F.interpolate(tmp, scale_factor=2)  # upscales the branch by 2
        tmp = torch.cat((tmp, x3), 1)
        _, out3, l3 = self.detect3(tmp, targets) # branch is not needed here

        if targets is None:
            return out1, out2, out3
        
        else:
            return (out1, l1), (out2, l2), (out3, l3)


class YoloNetV3(nn.Module):

    def __init__(self, num_classes, img_size, nms=False, post=True):
        super().__init__()
        self.darknet = FeatureExtractor()
        self.yolo_tail = YoloNeck(num_classes, img_size)
        self.nms = nms
        self._post_process = post

    def forward(self, x, targets=None):
        tmp1, tmp2, tmp3 = self.darknet(x)
        out1, out2, out3 = self.yolo_tail(tmp1, tmp2, tmp3, targets)
        # out = torch.cat((out1, out2, out3), 1)
        return out1, out2, out3

if __name__ == "__main__":

    img_size = 416
    num_classes = 80
    test = torch.rand(2, 3, img_size, img_size)
    label = torch.tensor([[0.000, 6.000, 0.546, 0.654, 0.123, 0.111], [0.000, 6.000, 0.546, 0.654, 0.123, 0.111]])
    yolo = YoloNetV3(num_classes, img_size)
    out1, out2, out3 = yolo(test, label)

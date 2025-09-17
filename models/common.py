# common.py + 4QAT

import math

import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, xyxy2xywh
from utils.plots import color_list
import warnings

import torch.nn.functional as F

from .quant_common import CommonIntActQuant, CommonUintActQuant, CommonWeightQuant, CommonActQuant
from .quant_common import CommonIntWeightPerChannelQuant, CommonIntWeightPerTensorQuant

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        #------------------------------------------------------
        self.bn = nn.BatchNorm2d(c2)
        #self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        #------------------------------------------------------
        #self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        #self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.act = nn.LeakyReLU(0.125, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        #self.act = self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class StemBlock(nn.Module):
    def __init__(self, c1, c2, k=3, s=2, p=None, g=1, act=True):
        super(StemBlock, self).__init__()
        self.stem_1 = Conv(c1, c2, k, s, p, g, act)
        self.stem_2a = Conv(c2, c2 // 2, 1, 1, 0)
        self.stem_2b = Conv(c2 // 2, c2, 3, 2, 1)
        self.stem_2p = nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)
        self.stem_3 = Conv(c2 * 2, c2, 1, 1, 0)

    def forward(self, x):
        stem_1_out  = self.stem_1(x)
        stem_2a_out = self.stem_2a(stem_1_out)
        stem_2b_out = self.stem_2b(stem_2a_out)
        stem_2p_out = self.stem_2p(stem_1_out)
        out = self.stem_3(torch.cat((stem_2b_out,stem_2p_out),1))
        return out

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        #self.act = nn.LeakyReLU(0.1, inplace=True)
        self.act = nn.LeakyReLU(0.125, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class ShuffleV2Block(nn.Module):
    """
    def __init__(self, inp, oup, stride):
        super(ShuffleV2Block, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.SiLU(),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.SiLU(),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.SiLU(),
        )
    """
    def __init__(self, inp, oup, stride):
        super(ShuffleV2Block, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp, eps=0.001, momentum=0.03),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features, eps=0.001, momentum=0.03),
                #nn.LeakyReLU(0.1, inplace=True),
                nn.LeakyReLU(0.125, inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features, eps=0.001, momentum=0.03),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.LeakyReLU(0.125, inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features, eps=0.001, momentum=0.03),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features, eps=0.001, momentum=0.03),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.LeakyReLU(0.125, inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)
        return out
    
class BlazeBlock(nn.Module):
    def __init__(self, in_channels,out_channels,mid_channels=None,stride=1):
        super(BlazeBlock, self).__init__()
        mid_channels = mid_channels or in_channels
        assert stride in [1, 2]
        if stride>1:
            self.use_pool = True
        else:
            self.use_pool = False

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=mid_channels,kernel_size=5,stride=stride,padding=2,groups=in_channels),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(in_channels=mid_channels,out_channels=out_channels,kernel_size=1,stride=1),
            nn.BatchNorm2d(out_channels),
        )

        if self.use_pool:
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(kernel_size=stride, stride=stride),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.SiLU(inplace=True)

    def forward(self, x):
        branch1 = self.branch1(x)
        out = (branch1+self.shortcut(x)) if self.use_pool else (branch1+x)
        return self.relu(out)    
  
class DoubleBlazeBlock(nn.Module):
    def __init__(self,in_channels,out_channels,mid_channels=None,stride=1):
        super(DoubleBlazeBlock, self).__init__()
        mid_channels = mid_channels or in_channels
        assert stride in [1, 2]
        if stride > 1:
            self.use_pool = True
        else:
            self.use_pool = False

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, stride=stride,padding=2,groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=5, stride=1,padding=2),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
        )

        if self.use_pool:
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(kernel_size=stride, stride=stride),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.SiLU(inplace=True)

    def forward(self, x):
        branch1 = self.branch1(x)
        out = (branch1 + self.shortcut(x)) if self.use_pool else (branch1 + x)
        return self.relu(out)
    
    
class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Focus(nn.Module):
    # Focus wh information into c-spacecd
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)

class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    img_size = 640  # inference size (pixels)
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('autoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=720, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/samples/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(720,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(720,1280,3)
        #   numpy:           = np.zeros((720,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,720,1280)  # BCHW
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1 = [], []  # image and inference shapes
        for i, im in enumerate(imgs):
            if isinstance(im, str):  # filename or uri
                im = Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)  # open
            im = np.array(im)  # to numpy
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32

        # Inference
        with torch.no_grad():
            y = self.model(x, augment, profile)[0]  # forward
        y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS

        # Post-process
        for i in range(n):
            scale_coords(shape1, y[i][:, :4], shape0[i])

        return Detections(imgs, y, self.names)


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, names=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)

    def display(self, pprint=False, show=False, save=False, render=False):
        colors = color_list()
        for i, (img, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'Image {i + 1}/{len(self.pred)}: {img.shape[0]}x{img.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f'{n} {self.names[int(c)]}s, '  # add to string
                if show or save or render:
                    img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        # str += '%s %.2f, ' % (names[int(cls)], conf)  # label
                        ImageDraw.Draw(img).rectangle(box, width=4, outline=colors[int(cls) % 10])  # plot
            if pprint:
                print(str)
            if show:
                img.show(f'Image {i}')  # show
            if save:
                f = f'results{i}.jpg'
                str += f"saved to '{f}'"
                img.save(f)  # save
            if render:
                self.imgs[i] = np.asarray(img)

    def print(self):
        self.display(pprint=True)  # print results

    def show(self):
        self.display(show=True)  # show results

    def save(self):
        self.display(save=True)  # save results

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def __len__(self):
        return self.n

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)


# ========= 4bit quantization ==========
class SCQActivation(nn.Module):
    """
    Signed Clipping Quantization for Activation
    """
    def __init__(self, bit=4):
        super(SCQActivation, self).__init__()
        self.bit = bit
        self.qmin = -(2 ** (bit - 1))
        self.qmax = (2 ** (bit - 1)) - 1

    def forward(self, x):
        scale = x.abs().max() / self.qmax + 1e-8  # avoid zero-div
        x_q = torch.clamp(torch.round(x / scale), self.qmin, self.qmax)
        return x_q * scale
    
class M4bQInputConv(nn.Module):
    """
    M4bQInputConv: RGB input 8bit --> 4bit odd / even 분해 --> conv 합산.
    """

    def __init__(self, conv: nn.Conv2d, bit=4):
        super(M4bQInputConv, self).__init__()
        self.bit = bit
        self.qmax = (1 << (bit - 1)) - 1  # 7
        self.qmin = -(1 << (bit - 1))     # -8

        # 4bit용으로 동일한 weight, bias를 복사한 두 개의 conv 정의
        self.conv_odd = nn.Conv2d(conv.in_channels, conv.out_channels,
                                  kernel_size=conv.kernel_size,
                                  stride=conv.stride,
                                  padding=conv.padding,
                                  dilation=conv.dilation,
                                  groups=conv.groups,
                                  bias=(conv.bias is not None))

        self.conv_even = nn.Conv2d(conv.in_channels, conv.out_channels,
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride,
                                   padding=conv.padding,
                                   dilation=conv.dilation,
                                   groups=conv.groups,
                                   bias=(conv.bias is not None))

        # weight, bias 복사
        self.conv_odd.weight.data.copy_(conv.weight.data)
        self.conv_even.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            self.conv_odd.bias.data.copy_(conv.bias.data)
            self.conv_even.bias.data.copy_(conv.bias.data)

    @staticmethod
    def from_conv(conv: nn.Conv2d):
        return M4bQInputConv(conv)

    def quantize_input_8bit(self, x):
        # SCQ 8bit activation quantization --> 양자화 후 fp32로 유지
        x_q = torch.clamp(x, min=-128, max=127)  # int8 범위
        return x_q

    def decompose_4bit(self, x_q8):
        # 논문 (5), (6): 8bit 값을 odd/even 4bit로 분해
        f_odd = torch.floor(x_q8 / 16)
        #f_even = (x_q8 % 16) - 8
        f_even = torch.remainder(x_q8, 16) - 8

        f_odd = torch.clamp(f_odd, self.qmin, self.qmax)
        f_even = torch.clamp(f_even, self.qmin, self.qmax)
        return f_odd, f_even

    def forward(self, x):
        # Step 1: 8bit quantization (SCQ simulation)
        x_q8 = self.quantize_input_8bit(x)

        # Step 2: decompose to two 4bit tensors
        x_odd, x_even = self.decompose_4bit(x_q8)

        # Step 3: apply conv separately
        #out_odd = self.conv_odd(x_odd.float()) * 16
        #out_even = self.conv_even(x_even.float() + 8)
        out_odd = self.conv_odd(x_odd * 1.0) * 16.0
        out_even = self.conv_even(x_even + 8.0)

        return out_odd + out_even
    
class QuantConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 bias=True, w_bits=4, groups=1):
        super().__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.w_bits = w_bits
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, bias=bias, groups=groups)

        # step size는 학습 가능한 변수로 정의
        self.alpha = nn.Parameter(torch.tensor(1.0))  # scale factor for quantization

    def quantize_weight(self, w):
        # LSQ - 4bit quantization
        qn = -2 ** (self.w_bits - 1)
        qp = 2 ** (self.w_bits - 1) - 1

        g = 1.0 / ((w.numel() * qp) ** 0.5)  # gradient scale
        alpha = self.alpha.clamp(min=1e-5)

        w_q = (w / alpha).clamp(qn, qp).round() * alpha
        return w_q

    def forward(self, x):
        w_q = self.quantize_weight(self.conv.weight)
        w_q = w_q.to(dtype=x.dtype, device=x.device)

        b = self.conv.bias
        if b is not None:
            b = b.to(dtype=x.dtype, device=x.device)

        return F.conv2d(x, w_q, b, self.conv.stride,
                        self.conv.padding, self.conv.dilation, self.conv.groups)


    @classmethod
    def from_conv(cls, conv: nn.Conv2d):
        quant = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=(conv.bias is not None),
            w_bits=4,
            groups=conv.groups
        )
        #quant.conv.groups = conv.groups  # group conv 호환

        # weight 복사
        with torch.no_grad():
            quant.conv.weight.copy_(conv.weight)
            if conv.bias is not None:
                quant.conv.bias.copy_(conv.bias)
        return quant
    
        # === Conv2d 속성 proxy 추가 ===
    @property
    def in_channels(self): return self.conv.in_channels
    @property
    def out_channels(self): return self.conv.out_channels
    @property
    def kernel_size(self): return self.conv.kernel_size
    @property
    def stride(self): return self.conv.stride
    @property
    def padding(self): return self.conv.padding
    @property
    def dilation(self): return self.conv.dilation
    @property
    def groups(self): return self.conv.groups
    @property
    def bias(self): return self.conv.bias
    @property
    def weight(self): return self.conv.weight

class QuantConv2d_LSQ(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 bias=True, w_bits=4, groups=1):
        super().__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.w_bits = w_bits
        self.qn = -2 ** (w_bits - 1)
        self.qp = 2 ** (w_bits - 1) - 1

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, bias=bias, groups=groups)

        self.alpha = nn.Parameter(torch.tensor(1.0))  # step size (learnable)
        self.init_done = False

            # === Conv2d 속성 proxy ===
    @property
    def weight(self): return self.conv.weight

    @property
    def bias(self): return self.conv.bias

    @property
    def in_channels(self): return self.conv.in_channels

    @property
    def out_channels(self): return self.conv.out_channels

    @property
    def kernel_size(self): return self.conv.kernel_size

    @property
    def stride(self): return self.conv.stride

    @property
    def padding(self): return self.conv.padding

    @property
    def dilation(self): return self.conv.dilation

    @property
    def groups(self): return self.conv.groups



    def forward(self, x):
        w = self.conv.weight.to(dtype=x.dtype, device=x.device)
        #w = self.conv.weight

        if not self.init_done:
            # alpha 초기값 설정: 평균적인 weight magnitude에 맞춤
            #self.alpha.data.copy_(2 * w.abs().mean() / (self.qp ** 0.5))
            # int8
            self.alpha.data.copy_(2 * w.abs().float().mean() / (self.qp ** 0.5))
            self.init_done = True

        # 실제 weight 양자화 (custom function with backward)
        w_q = LSQQuantizer.apply(w, self.alpha, self.qn, self.qp)

        b = self.conv.bias
        return F.conv2d(x, w_q, b, self.conv.stride,
                        self.conv.padding, self.conv.dilation, self.conv.groups)
    
    def __getattribute__(self, name):
        # 먼저 기본 속성을 가져오려 시도
        try:
            return super().__getattribute__(name)
        except AttributeError:
            # 기본 속성에 없으면 self.conv에서 가져오기 시도
            conv = super().__getattribute__('__dict__').get('conv', None)
            if conv is not None and hasattr(conv, name):
                return getattr(conv, name)
            raise

    @classmethod
    def from_conv(cls, conv: nn.Conv2d):
        q = cls(conv.in_channels, conv.out_channels,
                conv.kernel_size, conv.stride, conv.padding,
                bias=(conv.bias is not None), w_bits=4, groups=conv.groups)
        q.conv.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            q.conv.bias.data.copy_(conv.bias.data)
        return q


class LSQQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, alpha, qn, qp):
        ctx.save_for_backward(w, alpha)
        ctx.other = (qn, qp)

        # 정방향: 양자화 수행
        w_div = w / alpha
        w_clamped = torch.clamp(w_div, qn, qp)
        w_rounded = w_clamped.round()
        w_quant = w_rounded * alpha

        return w_quant

    @staticmethod
    def backward(ctx, grad_output):
        w, alpha = ctx.saved_tensors
        qn, qp = ctx.other

        # g = LSQ scaling factor
        g = 1.0 / ((w.numel() * qp) ** 0.5)

        # ∂L/∂w ≈ ∂L/∂w_quant
        grad_w = grad_output.clone()

        # ∂L/∂alpha: LSQ 방식
        w_div = w / alpha
        indicator = ((w_div >= qn) & (w_div <= qp)).float()
        grad_alpha = ((w_div - w_div.round()) * indicator * grad_output).sum().unsqueeze(0)
        grad_alpha = grad_alpha * g

        return grad_w, grad_alpha, None, None  # 나머지는 정수 qn/qp, gradient 없음
    
    #Stemblock --> M4bQ
class M4bQ(nn.Module):
    def __init__(self, c1, c2, k=3, s=2, p=None, g=1, act=True):
        super(M4bQ, self).__init__()
        base_conv = Conv(c1, c2, k, s, p, g, act)
        self.stem_1 = M4bQInputConv.from_conv(base_conv.conv)
        self.stem_2a = Conv(c2, c2 // 2, 1, 1, 0)
        self.stem_2b = Conv(c2 // 2, c2, 3, 2, 1)
        self.stem_2p = nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)
        self.stem_3 = Conv(c2 * 2, c2, 1, 1, 0)

    def forward(self, x):
        stem_1_out  = self.stem_1(x)
        stem_2a_out = self.stem_2a(stem_1_out)
        stem_2b_out = self.stem_2b(stem_2a_out)
        stem_2p_out = self.stem_2p(stem_1_out)
        out = self.stem_3(torch.cat((stem_2b_out,stem_2p_out),1))
        return out
    
# ===================


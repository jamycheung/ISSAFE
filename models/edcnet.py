from models.util import _BNReluConv, upsample
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from itertools import chain
import torch.utils.checkpoint as cp
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from models.util import _Upsample
import torch.nn.functional as F

__all__ = ['EDCNet']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)

def _bn_function_factory(conv, norm, relu=None):
    """return a conv-bn-relu function"""
    def bn_function(x):
        x = conv(x)
        if norm is not None:
            x = norm(x)
        if relu is not None:
            x = relu(x)
        return x

    return bn_function


def do_efficient_fwd(block, x, efficient):
    if efficient and x.requires_grad:
        return cp.checkpoint(block, x)
    else:
        return block(x)


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)

    return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, efficient=False, use_bn=True, dilation=1):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.downsample = downsample
        self.stride = stride
        self.efficient = efficient

    def forward(self, x):
        residual = x
        bn_1 = _bn_function_factory(self.conv1, self.bn1, self.relu)
        bn_2 = _bn_function_factory(self.conv2, self.bn2)
        out = do_efficient_fwd(bn_1, x, self.efficient)
        out = do_efficient_fwd(bn_2, out, self.efficient)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        relu = self.relu(out)
        return relu, out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, efficient=True, use_bn=True, dilation=1):
        super(Bottleneck, self).__init__()
        self.use_bn = use_bn
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.conv2 = conv3x3(planes, planes, stride=stride, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion) if self.use_bn else None
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.efficient = efficient

    def forward(self, x):
        residual = x

        bn_1 = _bn_function_factory(self.conv1, self.bn1, self.relu)
        bn_2 = _bn_function_factory(self.conv2, self.bn2, self.relu)
        bn_3 = _bn_function_factory(self.conv3, self.bn3, self.relu)

        out = do_efficient_fwd(bn_1, x, self.efficient)
        out = do_efficient_fwd(bn_2, out, self.efficient)
        out = do_efficient_fwd(bn_3, out, self.efficient)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        relu = self.relu(out)

        return relu, out


class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='bilinear'):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        if self.scale_factor:
            x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)
        else:
            x = self.interp(x, size=self.size, mode=self.mode, align_corners=True)
        return x


class GatedSpatialConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(GatedSpatialConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, 'zeros')

        self._gate_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels + 1),
            nn.Conv2d(in_channels + 1, in_channels + 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels + 1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, input_features, gating_features):
        """
        :param input_features:  [NxCxHxW]  featuers comming from the event branch.
        :param gating_features: [Nx1xHxW] features comming from the texture branch (resnet). Only one channel feature map.
        :return:
        """
        alphas = self._gate_conv(torch.cat([input_features, gating_features], dim=1))

        input_features = (input_features * (alphas + 1))
        return F.conv2d(input_features, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

class SPP_Event(nn.Module):
    def __init__(self, num_maps_in, num_maps_out, bn_momentum=0.1, event_dim=1):
        super(SPP_Event, self).__init__()
        self.event_conv=nn.Sequential(
            nn.Conv2d(event_dim, num_maps_out, kernel_size=1, bias=False),
            nn.ReLU(inplace=True))
    def forward(self, feat, event):
        feat_size = feat.size()
        event = F.interpolate(event, feat_size[2:], mode='bilinear',align_corners=True)
        event = self.event_conv(event)
        return event

class SpatialPyramidPooling_Event(nn.Module):
    def __init__(self, num_maps_in, num_levels, bt_size=128, level_size=42, out_size=128,
                 grids=(8, 4, 2, 1), square_grid=False, bn_momentum=0.1, use_bn=True, event_dim=1):
        super(SpatialPyramidPooling_Event, self).__init__()
        self.grids = grids
        self.square_grid = square_grid
        self.spp = nn.Sequential()
        self.spp.add_module('spp_bn',
                            _BNReluConv(num_maps_in, bt_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn))
        self.spp.add_module('spp_event', SPP_Event(bt_size, level_size, bn_momentum=bn_momentum, event_dim=event_dim))
        num_features = bt_size  
        final_size = num_features + level_size  
        for i in range(num_levels): 
            final_size += level_size
            self.spp.add_module('spp' + str(i),
                                _BNReluConv(num_features, level_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn))
        self.spp.add_module('spp_fuse',
                            _BNReluConv(final_size, out_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn))

    def forward(self, x, event):
        """
        :param x: feature shape (B, 512, H/32, W/32)
        :param event: event shape (B, 1, H, W)
        :return:
        """
        levels = []
        target_size = x.size()[2:4]  # Bx512x(H/32)x(W/32)

        ar = target_size[1] / target_size[0]

        x = self.spp[0].forward(x)
        levels.append(x)

        x_event = self.spp[1].forward(x, event)
        levels.append(x_event)
        num = len(self.spp) - 1

        for i in range(2, num):
            if not self.square_grid:
                if torch.is_tensor(ar): ar = ar.data.tolist()
                grid_size = (self.grids[i - 1], max(1, round(ar * self.grids[i - 1])))
                x_pooled = F.adaptive_avg_pool2d(x, grid_size)
            else:
                x_pooled = F.adaptive_avg_pool2d(x, self.grids[i - 1])
            level = self.spp[i].forward(x_pooled)

            level = upsample(level, target_size)
            levels.append(level)
        x = torch.cat(levels, 1)
        x = self.spp[-1].forward(x)
        return x

class ResNet(nn.Module):
    def __init__(self, rgb_dim, event_dim, block, layers, *, num_features=128, k_up=3, efficient=True, use_bn=True,
                 spp_grids=(16, 8, 4, 2, 1), spp_square_grid=False, **kwargs):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.efficient = efficient
        self.use_bn = use_bn


        self.conv1 = nn.Conv2d(rgb_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64) if self.use_bn else lambda x: x
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        upsamples = []
        self.layer1 = self._make_layer_rgb(block, 64, 64, layers[0], stride=1, dilation=1)
        self.gating_1 = self.gating(64, 4)
        self.layer1_e = self._make_layer_d(block, 64, 64)
        self.gate1 = GatedSpatialConv2d(64, 64)
        upsamples += [_Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up)] #  num_maps_in, skip_maps_in, num_maps_out, k: kernel size of blend conv

        self.layer2 = self._make_layer_rgb(block, 64, 128, layers[1], stride=2, dilation=1)
        self.gating_2 = self.gating(128, 8)
        self.layer2_e = self._make_layer_d(block, 64, 32)
        self.gate2 = GatedSpatialConv2d(32, 32)
        upsamples += [_Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up)]

        self.layer3 = self._make_layer_rgb(block, 128, 256, layers[2], stride=2, dilation=2)
        self.gating_3 = self.gating(256, 16)
        self.layer3_e = self._make_layer_d(block, 32, 16)
        self.gate3 = GatedSpatialConv2d(16, 16)
        upsamples += [_Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up)]

        self.layer4 = self._make_layer_rgb(block, 256, 512, layers[3], stride=2, dilation=3)
        self.gating_4 = self.gating(512, 32)
        self.layer4_e = self._make_layer_d(block, 16, 8)
        self.gate4 = GatedSpatialConv2d(8, 8)

        self.time_bins = event_dim
        self.fuse = nn.Conv2d(8, self.time_bins, 1, bias=False)

        self.fine_tune = list()
        self.fine_tune += [self.conv1, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4]
        self.fine_tune += [self.layer1_e, self.layer2_e, self.layer3_e, self.layer4_e]
        self.fine_tune += [self.gating_1, self.gate1, self.gating_2, self.gate2, self.gating_3, self.gate3, self.gating_4, self.gate4, self.fuse]
        if self.use_bn:
            self.fine_tune += [self.bn1]

        num_levels = 3
        self.spp_size = num_features
        bt_size = self.spp_size
        level_size = self.spp_size // num_levels
        self.spp = SpatialPyramidPooling_Event(self.inplanes, num_levels, bt_size=bt_size, level_size=level_size,
                                         out_size=self.spp_size, grids=spp_grids, square_grid=spp_square_grid,
                                         bn_momentum=0.01 / 2, use_bn=self.use_bn, event_dim=self.time_bins)

        self.upsample = nn.ModuleList(list(reversed(upsamples)))
        self.random_init = [self.spp, self.upsample, self.fuse]
        self.num_features = num_features

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer_rgb(self, block, inplanes, planes, blocks, stride=1, dilation=1):   #block, 64, 64, 2
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            layers = [nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)]
            if self.use_bn:
                layers += [nn.BatchNorm2d(planes * block.expansion)]
            downsample = nn.Sequential(*layers)
        layers = [block(inplanes, planes, stride, downsample, efficient=self.efficient, use_bn=self.use_bn)]
        inplanes = planes * block.expansion
        self.inplanes = inplanes
        for i in range(1, blocks):
            layers += [block(inplanes, planes, efficient=self.efficient, use_bn=self.use_bn)]

        return nn.Sequential(*layers)

    def _make_layer_d(self, block, inplanes, planes, stride=1):
        layers = [BasicBlock(inplanes, inplanes, stride=stride, efficient=self.efficient, use_bn=self.use_bn)]
        layers += [conv1x1(inplanes, planes)]
        return nn.Sequential(*layers)

    def gating(self, num_channels, scale_factor):
        """create gate/attention from RGB branch"""
        return nn.Sequential(
            nn.Conv2d(num_channels, 1, 1, bias=False),
            Interpolate(scale_factor=scale_factor)
        )

    def random_init_params(self):
        return chain(*[f.parameters() for f in self.random_init])

    def fine_tune_params(self):
        return chain(*[f.parameters() for f in self.fine_tune])

    def forward_resblock(self, x, layers):
        skip = None
        for l in layers:
            x = l(x)
            if isinstance(x, tuple):
                x, skip = x
        return x, skip


    def forward_down_fusion(self, rgb):
        x = self.conv1(rgb)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        y = F.interpolate(x, scale_factor=4)

        features = []
        # block 1
        x, skip_rgb = self.forward_resblock(x, self.layer1)
        y, skip_event = self.forward_resblock(y, self.layer1_e)
        x_gate = self.gating_1(x)
        y = self.gate1(y, x_gate)
        features += [skip_rgb]
        # block 2
        x, skip_rgb = self.forward_resblock(x, self.layer2)
        y, skip_event = self.forward_resblock(y, self.layer2_e)
        x_gate = self.gating_2(x)
        y = self.gate2(y, x_gate)
        features += [skip_rgb]
        # block 3
        x, skip_rgb = self.forward_resblock(x, self.layer3)
        y, skip_event = self.forward_resblock(y, self.layer3_e)
        x_gate = self.gating_3(x)
        y = self.gate3(y, x_gate)
        features += [skip_rgb]
        # block 4
        x, skip_rgb = self.forward_resblock(x, self.layer4)
        y, skip_event = self.forward_resblock(y, self.layer4_e)
        x_gate = self.gating_4(x)
        y = self.gate4(y, x_gate)
        y = self.fuse(y)

        features += [self.spp.forward(x, y)]
        return features, y


    def forward_up(self, features, y):
        features = features[::-1]
        x = features[0] 
        upsamples = []
        for skip, up in zip(features[1:], self.upsample):
            x = up(x, skip)
            upsamples += [x]
        return x, y

    def forward(self, rgb):
        features, y = self.forward_down_fusion(rgb)
        return self.forward_up(features, y)

    def _load_resnet_pretrained(self, url):
        pretrain_dict = model_zoo.load_url(model_urls[url])
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                if k.startswith('conv1'):
                    model_dict[k] = v
                    model_dict[k.replace('conv1', 'conv1_d')] = torch.mean(v, 1).data. \
                        view_as(state_dict[k.replace('conv1', 'conv1_d')])

                elif k.startswith('bn1'):
                    model_dict[k] = v
                    model_dict[k.replace('bn1', 'bn1_d')] = v
                elif k.startswith('layer'):
                    model_dict[k] = v
                    model_dict[k[:6]+'_d'+k[6:]] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


def resnet18(rgb_dim=3, event_dim=0, pretrained=True, **kwargs):
    model = ResNet(rgb_dim, event_dim, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained and rgb_dim:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
        print('pretrained dict loaded sucessfully')
    return model


class EDCNet(nn.Module):
    def __init__(self, rgb_dim=3, event_dim=0, num_classes=19, use_bn=True):
        super(EDCNet, self).__init__()
        self.rgb_dim = rgb_dim
        self.event_dim = event_dim
        self.backbone = resnet18(rgb_dim, event_dim, pretrained=True, efficient=False, use_bn= True)
        self.num_classes = num_classes
        self.logits = _BNReluConv(self.backbone.num_features, self.num_classes, batch_norm=use_bn)

    def forward(self, rgb_inputs=None, event_inputs = None):
        x, event = self.backbone(rgb_inputs)
        logits = self.logits.forward(x)
        up_shape = rgb_inputs.shape[-2:] if rgb_inputs is not None else event_inputs.shape[-2:]
        logits_upsample = upsample(logits, up_shape)
        return logits_upsample, event


    def random_init_params(self):
        return chain(*([self.logits.parameters(), self.backbone.random_init_params()]))

    def fine_tune_params(self):
        return self.backbone.fine_tune_params()


if __name__=='__main__':
    inp = torch.rand((1, 3, 256, 512))
    net = EDCNet(3, 18).cuda()
    net = torch.nn.DataParallel(net, device_ids=[0])
    import os
    from utils.saver import summary
    modelfile = os.path.join('model_functionsummary.txt')
    with open(modelfile, 'w') as f:
        f.write(str(net))
    modelfile = os.path.join('model_torchsummary.txt')
    with open(modelfile, 'w') as f:
        summary(net, [(3, 256, 512), (18, 256, 512)], f)
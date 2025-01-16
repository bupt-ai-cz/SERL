from torchvision import models
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


class RevGrad(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


grad_reverse = RevGrad.apply


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)

    _output = torch.div(input, norm.view(-1, 1).expand_as(input))

    output = _output.view(input_size)

    return output


class AlexNetBase(nn.Module):
    def __init__(self, pret=True, bootleneck_dim=256):
        super(AlexNetBase, self).__init__()
        model_alexnet = models.alexnet(pretrained=pret)
        self.features = nn.Sequential(*list(model_alexnet.
                                            features._modules.values())[:])
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i),
                                       model_alexnet.classifier[i])
        self.__in_features = model_alexnet.classifier[6].in_features
        self.bottle_neck = feat_bootleneck(feature_dim=4096, bottleneck_dim=bootleneck_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        x=self.bottle_neck(x)
        return x

    def output_num(self):
        return self.__in_feature


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or \
       classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)


class MLP(nn.Module):
    def __init__(self, pret=True,input_dim=3000,bootleneck_dim=256):
        super(MLP, self).__init__()
        self.features = nn.Sequential(nn.Linear(input_dim,out_features=500))
        self.features.apply(init_weights)
        self.__in_features = 500
        self.bottle_neck = feat_bootleneck(feature_dim=500, bottleneck_dim=bootleneck_dim)
        self.bottle_neck.apply(init_weights)

    def forward(self, x):
        x = self.features(x)
        x=self.bottle_neck(x)
        return x

    def output_num(self):
        return self.__in_feature


class feat_bootleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="bn"):
        super(feat_bootleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.3)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        x = self.dropout(x)
        return x


class VGGBase(nn.Module):
    def __init__(self, pret=True, no_pool=False,bootleneck_dim=256):
        super(VGGBase, self).__init__()
        vgg16 = models.vgg16(pretrained=pret)
        self.classifier = nn.Sequential(*list(vgg16.classifier.
                                              _modules.values())[:-1])
        self.features = nn.Sequential(*list(vgg16.features.
                                            _modules.values())[:])
        self.s = nn.Parameter(torch.FloatTensor([10]))
        self.bottle_neck=feat_bootleneck(feature_dim=4096,bottleneck_dim=bootleneck_dim)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 7 * 7 * 512)
        x = self.classifier(x)
        x = self.bottle_neck(x)
        return x


class VGGBase_no_neck(nn.Module):
    def __init__(self, pret=True, no_pool=False,bootleneck_dim=256):
        super(VGGBase_no_neck, self).__init__()
        vgg16 = models.vgg16(pretrained=pret)
        self.classifier = nn.Sequential(*list(vgg16.classifier.
                                              _modules.values())[:-1])
        self.features = nn.Sequential(*list(vgg16.features.
                                            _modules.values())[:])
        self.s = nn.Parameter(torch.FloatTensor([10]))
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 7 * 7 * 512)
        x = self.classifier(x)
        return x


momentum = 0.001


def mish(x):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function (https://arxiv.org/abs/1908.08681)"""
    return x * torch.tanh(F.softplus(x))


class PSBatchNorm2d(nn.BatchNorm2d):
    """How Does BN Increase Collapsed Neural Network Filters? (https://arxiv.org/abs/2001.11216)"""
    def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, x):
        return super().forward(x) + self.alpha


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001, eps=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001, eps=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=True) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            print(x.shape)
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNetVar(nn.Module):
    def __init__(self, first_stride, num_classes, depth=28, widen_factor=2, drop_rate=0.0, is_remix=False):
        super(WideResNetVar, self).__init__()
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor, 128 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=True)
        # 1st block
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], block, first_stride, drop_rate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(
            n, channels[2], channels[3], block, 2, drop_rate)
        # 4th block
        self.block4 = NetworkBlock(
            n, channels[3], channels[4], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[4], momentum=0.001, eps=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

        self.features = nn.Sequential(self.conv1,self.block2,self.block3,self.block4,self.bn1,self.relu)

        self.bottle_neck=feat_bootleneck(feature_dim=channels[4], bottleneck_dim=256)
        # self.fc = nn.Linear(, num_classes)
        self.channels = channels[4]

        # rot_classifier for Remix Match
        self.is_remix = is_remix
        if is_remix:
            self.rot_classifier = nn.Linear(self.channels, 4)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x, ood_test=False):
        # out = self.conv1(x)
        # out = self.block1(out)
        # out = self.block2(out)
        # out = self.block3(out)
        # out = self.block4(out)
        # out = self.relu(self.bn1(out))

        out=self.features(x)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)


        output = self.bottle_neck(out)

        if ood_test:
            return output, out
        else:
            if self.is_remix:
                rot_output = self.rot_classifier(out)
                return output, rot_output
            else:
                return output


class build_WideResNetVar:
    def __init__(self, first_stride=1, depth=28, widen_factor=2, bn_momentum=0.01, leaky_slope=0.1, dropRate=0.0,
                 use_embed=False, is_remix=False):
        self.first_stride = first_stride
        self.depth = depth
        self.widen_factor = widen_factor
        self.bn_momentum = bn_momentum
        self.dropRate = dropRate
        self.leaky_slope = leaky_slope
        self.use_embed = use_embed
        self.is_remix = is_remix

    def build(self, num_classes):
        return WideResNetVar(
            first_stride=self.first_stride,
            depth=self.depth,
            num_classes=num_classes,
            widen_factor=self.widen_factor,
            drop_rate=self.dropRate,
            is_remix=self.is_remix,
        )



import torch.nn.utils.weight_norm as weightNorm


class Predictor(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05, norm_feature=1):
        super(Predictor, self).__init__()
        self.fc = nn.Linear(inc, num_class, bias=True)
        self.fc.apply(init_weights)
        self.num_class = num_class
        self.temp = temp
        self.norm_feature = norm_feature
    def forward(self, x, reverse=False, eta=0.1):
        if reverse:
            x = grad_reverse(x, torch.tensor(eta, requires_grad=False))
        if self.norm_feature:
            x = F.normalize(x)
            x_out = self.fc(x) / self.temp
        else:
            x_out = self.fc(x)
        return x_out


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class Predictor_deep(nn.Module):
    def __init__(self, num_class=64, inc=4096, norm_feature=1,temp=0.05):
        super(Predictor_deep, self).__init__()
        self.fc1 = nn.Linear(inc, inc//2)
        # self.fc3 = nn.Linear(inc//2, inc//2)
        self.fc1.apply(init_weights)
        # self.fc3.apply(init_weights)
        self.fc2 = nn.Linear(inc//2, num_class,bias=False)
        nn.init.xavier_normal_(self.fc2.weight)
        self.bn = nn.BatchNorm1d(inc//2, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

        # self.bn3 = nn.BatchNorm1d(inc // 2, affine=True)
        # self.relu3 = nn.ReLU(inplace=True)
        # self.dropout3 = nn.Dropout(p=0.3)

        self.num_class = num_class
        self.temp = temp
        self.norm_feature=norm_feature

    def forward(self, x, reverse=False, eta=0.1):
        x = self.dropout(self.relu(self.bn(self.fc1(x))))
        # x = self.dropout3(self.relu3(self.fc3(x)))
        # if reverse:
        #     x = grad_reverse(x, eta)
        # x = F.normalize(x)
        # x_out = self.fc2(x) / self.temp
        if self.norm_feature:
            x = F.normalize(x)
            x_out = self.fc2(x) / self.temp
        else:
            x_out = self.fc2(x)
        return x_out


momentum = 0.001


def mish(x):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function (https://arxiv.org/abs/1908.08681)"""
    return x * torch.tanh(F.softplus(x))


class PSBatchNorm2d(nn.BatchNorm2d):
    """How Does BN Increase Collapsed Neural Network Filters? (https://arxiv.org/abs/2001.11216)"""

    def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, x):
        return super().forward(x) + self.alpha


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001, eps=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001, eps=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=True) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, first_stride, num_classes, depth=28, widen_factor=2, drop_rate=0.0, is_remix=False):
        super(WideResNet, self).__init__()
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        # print("channels",channels)
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=True)
        # 1st block
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], block, first_stride, drop_rate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(
            n, channels[2], channels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001, eps=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

        self.features = nn.Sequential(self.conv1,self.block1, self.block2,self.block3,self.bn1,self.relu)

        # self.bottle_neck=feat_bootleneck(feature_dim=channels[3], bottleneck_dim=256)

        # self.fc = nn.Linear(channels[3], num_classes)
        self.channels = channels[3]

        # rot_classifier for Remix Match
        self.is_remix = is_remix
        if is_remix:
            self.rot_classifier = nn.Linear(self.channels, 4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x, ood_test=False):
        # out = self.conv1(x)
        # out = self.block1(out)
        # out = self.block2(out)
        # out = self.block3(out)
        # out = self.relu(self.bn1(out))
        # print("kllllll")
        out=self.features(x)
        out = F.adaptive_avg_pool2d(out, 1)
        output = out.view(-1, self.channels)
        # print(self.channels)
        
        # output = self.fc(out)

        if ood_test:
            return output, out
        else:
            if self.is_remix:
                rot_output = self.rot_classifier(out)
                return output, rot_output
            else:
                return output


class build_WideResNet:
    def __init__(self, first_stride=1, depth=28, widen_factor=2, bn_momentum=0.01, leaky_slope=0.1, dropRate=0.0,
                 use_embed=False, is_remix=False):
        self.first_stride = first_stride
        self.depth = depth
        self.widen_factor = widen_factor
        self.bn_momentum = bn_momentum
        self.dropRate = dropRate
        self.leaky_slope = leaky_slope
        self.use_embed = use_embed
        self.is_remix = is_remix

    def build(self, num_classes):
        return WideResNet(
            first_stride=self.first_stride,
            depth=self.depth,
            num_classes=num_classes,
            widen_factor=self.widen_factor,
            drop_rate=self.dropRate,
            is_remix=self.is_remix,
        )
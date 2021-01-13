import torch
import torch.nn as nn
from torch.nn import functional as F


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation=True):
        super(Conv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation
        self.layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if activation:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, h, w = x.size()
        oh = ((h - 1) // self.stride) * self.stride + self.kernel_size
        ow = ((w - 1) // self.stride) * self.stride + self.kernel_size
        oh = max(0, oh - h)
        ow = max(0, ow - w)
        padding = (ow // 2, (ow + 1) // 2, oh // 2, (oh + 1) // 2)
        x = self.bn(self.layer(F.pad(x, padding)))
        if self.activation:
            x = self.relu(x)

        return x


class Bottleneck(nn.Module):
    def __init__(self, in_places, places, stride=1, expansion=2):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.stride = stride
        self.b0 = Conv(in_channels=in_places, out_channels=places, kernel_size=1, stride=1)
        self.b1 = Conv(in_channels=places, out_channels=places, kernel_size=5, stride=stride)
        self.b2 = Conv(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1,
                           activation=False)


        if self.stride > 1:
            self.downsample = Conv(in_channels=in_places, out_channels=places * self.expansion, kernel_size=3, \
                        stride=stride, activation=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.b0(x)
        out = self.b1(out)
        out = self.b2(out)

        if self.stride > 1:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


# ResNet50: [3, 4, 6, 3]
# ResNet101: [3, 4, 23, 3]
# ResNet152: [3, 8, 36, 3]
class ResNet(nn.Module):
    def __init__(self, blocks=[3, 4, 6, 3], class_name=[], expansion=2):
        super(ResNet, self).__init__()
        self.expansion = expansion
        assert len(class_name) > 0
        self.num_classes = len(class_name)
        self.class_name = class_name
        self.conv1 = Conv(in_channels=3, out_channels=32, kernel_size=7, stride=2)

        self.layer1 = self.make_layer(in_places=32, places=32, block=blocks[0], stride=2)
        self.layer2 = self.make_layer(in_places=64, places=64, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=128, places=128, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=256, places=256, block=blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, self.num_classes)

        self._centerloss = CenterLoss(self.num_classes, 512)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, stride))
        for i in range(1, block):
            layers.append(Bottleneck(places * self.expansion, places))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        f = x
        x = self.fc(x)
        return f, x

    def loss(self, x, y, eps=1e-16):
        x = torch.sigmoid(x)
        batch_size = y.size(0)
        one_hot = torch.zeros(batch_size, self.num_classes, device=x.device).scatter_(1, y, 1)
        cross_entropy = -one_hot * torch.log(x + eps) - (1 - one_hot) * torch.log(1 - x + eps)
        cross_entropy = cross_entropy.sum(1)

        prod, ind = x.max(1)
        acc = (ind.view(-1) == y.view(-1)).float()

        return cross_entropy.mean(), acc.mean()

    def regularization_loss(self, weight_decay=4e-5):
        reg_loss = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                reg_loss += torch.norm(param, p=2)

        return reg_loss * weight_decay

    def center_loss(self, x, y, loss_weight=1e-2):
        return self._centerloss(x, y, loss_weight)

    def save_weights(self, save_file):
        state = {'net': self.state_dict(), 'class_name': self.class_name}
        torch.save(state, save_file)
    
    def load_pretrained_weights(self, pretrianed_weights, by_name=True):
        # pretrianed_dict = torch.load(pretrianed_weights)['net']
        pretrianed_dict = torch.load(pretrianed_weights)
        model_dict = self.state_dict()
        if by_name:
            pretrianed_dict_update = {}
            for k, v in pretrianed_dict.items():
                if k in model_dict:
                    vv = model_dict[k]
                    if v.size() == vv.size():
                        pretrianed_dict_update[k] = v
            model_dict.update(pretrianed_dict_update)
        else:
            model_dict.update(pretrianed_dict)
        self.load_state_dict(model_dict)


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = torch.nn.Parameter(torch.randn(num_classes, feat_dim))
        self.classes = torch.arange(self.num_classes).long()
        self.use_cuda = False

    def forward(self, y, labels, loss_weight):
        # print(self.use_cuda)
        batch_size = y.size(0)
        # distmat = y**2 + centers**2
        distmat = torch.pow(y, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        # distmat = distmat - 2* dot(y, centers.t()) = (y-centers)**2
        distmat.addmm_(1, -2, y, self.centers.t())

        labels = labels.expand(batch_size, self.num_classes)
        mask = labels.eq(self.classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss * loss_weight

    def cuda(self, device_id=None):
        self.use_cuda = True
        self.classes.cuda(device_id)
        return self._apply(lambda t: t.cuda(device_id))

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        self.use_cuda = True
        self.classes = fn(self.classes)
        return self


if __name__ == '__main__':
    model = ResNet(num_classes=15)
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)
    torch.save(model.state_dict(), 'model.pth')

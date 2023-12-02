import torch
import torch.nn as nn

cfg = {
    "vgg11": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
}

#####################################################################################################
class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=100):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
                nn.init.kaiming_normal_(conv2d.weight, mode="fan_out", nonlinearity="relu")
                # batchnorm = nn.BatchNorm2d(x)
                # nn.init.constant_(batchnorm.weight, 1)
                # nn.init.constant_(batchnorm.bias, 0)

                layers += [
                    conv2d,
                    # batchnorm,
                    nn.ReLU(inplace=True),
                ]

                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
def vgg11(num_classes=10):
    return VGG("vgg11", num_classes=num_classes)

#####################################################################################################




#####################################################################################################
class VGG_conv0(nn.Module):
    def __init__(self, vgg_name, num_classes=100):
        super(VGG_conv0, self).__init__()
        self.conv0 = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=True)
        self.features = self._make_layers(cfg[vgg_name])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x):
        x = self.conv0(x)
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 4
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
                nn.init.kaiming_normal_(conv2d.weight, mode="fan_out", nonlinearity="relu")
                # batchnorm = nn.BatchNorm2d(x)
                # nn.init.constant_(batchnorm.weight, 1)
                # nn.init.constant_(batchnorm.bias, 0)

                layers += [
                    conv2d,
                    # batchnorm,
                    nn.ReLU(inplace=True),
                ]

                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
def vgg11conv0(num_classes=10):
    return VGG_conv0("vgg11", num_classes=num_classes)

#####################################################################################################
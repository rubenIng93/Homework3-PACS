import torch
import torch.nn as nn
from torch.autograd import Function
#from .utils import load_state_dict_from_url

__all__ = ['AlexNet', 'alexnet']

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
}

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class DANNAlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(DANNAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLu(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLu(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLu(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLu(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLu(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLu(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLu(inplace=True),
            nn.Linear(4096, num_classes)
        )

        self.dann_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLu(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLu(inplace=True),
            nn.Linear(4096, 2)
            # it must distinguish between target and source
            # then only 2 classes
        )

    def forward(self, x, alpha=None):
        features = self.features
        features = features.avgpool(features)
        # Flatten the features:
        features = features.view(features.size(0), -1)
        # If we pass alpha, we can assume we are training the discriminator
        if alpha is not None:
            # gradient reversal layer (backward gradients will be reversed)
            reverse_feature = ReverseLayerF.apply(features, alpha)
            discriminator_output = self.dann_classifier(reverse_feature)
            return discriminator_output
        # If we don't pass alpha, we assume we are training with supervision
        else:
            # go in the classifier
            class_outputs = self.classifier(features)
            return class_outputs


def DANNalexnet(pretrained=False, progress=True, **kwargs):

    model = DANNAlexNet(**kwargs)

    if pretrained:
        state_dict= torch.hub.load_state_dict_from_url(model_urls['alexnet'], progress=progress)
        model.load_state_dict(state_dict)

        # Applying at the discriminator's layers the pretrained weights
        dann_fc_idx = [1, 4, 6]
        for idx in dann_fc_idx:
            model.dann_classifier[idx].weight.data = model.classifier[idx].weight.data.clone()

        return model

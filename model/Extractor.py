import os
import torch
from torch import Tensor
import torch.nn as nn
from torchvision import models

def get_resnet(name: str='resnet34', pretrain: str=''):
    name = name.lower()
    pretrain = pretrain.lower()
    if name == 'resnet34':
        net = models.resnet34(pretrained=False)
        ft_size = 512
        net.load_state_dict(torch.load('./model weights/ResNet34.pth'))
    elif name == 'resnet101':
        net = models.resnet101(pretrained=False)
        ft_size = 2048
        net.load_state_dict(torch.load('./model weights/ResNet101.pth'))
    else:
        raise NotImplementedError("Unkown extractor name")
    extractor = nn.Sequential(*list(net.children())[:-2])
    if pretrain != '' and pretrain != 'none':
        if os.path.exists(pretrain):
            extractor.load_state_dict(torch.load(pretrain))
            print(f"Loaded pretrained file {pretrain} .")
        else:
            print(f"The pretrained file {pretrain} does not exist. Use the default one.")
    return extractor, ft_size   # delete the last avgpool layer and fc layer.

class FeaturesExtractor(nn.Module):
    def __init__(self, num_features: int=512, output_size: 'tuple[int, int]'=(16, 32), 
                extractor_name: str='resnet34', pretrain: str='', tr_extractor: bool=False):
        super(FeaturesExtractor, self).__init__()
        extractor_name = extractor_name.lower()
        pretrain = pretrain.lower()
        self.extractor, ft_size = get_resnet(name=extractor_name, pretrain=pretrain)
        if not tr_extractor:
            for param in self.extractor.parameters():
                param.requires_grad = False
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=output_size) if output_size else None
        self.fc = nn.Linear(ft_size, num_features)

    def forward(self, img: Tensor):
        '''
        :param img: (batch_size, n_channel, H, W)
        :return: features. Shape: (output_w, output_h, batch_size, n_feature)
        '''
        if self.avgpool:
            ft = self.avgpool(self.extractor(img))      # (batch_size, n_feature, *output_size)
        else:
            ft = self.extractor(img)                    # (batch_size, n_feature, *default_size)
        ft = ft.permute(0, 2, 3, 1).contiguous()        # (batch_size, output_w, output_h, n_feature)
        return self.fc(ft)

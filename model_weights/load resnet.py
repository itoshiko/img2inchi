# %%
from torchvision import models
import torch

resnet34 = models.resnet34(pretrained=True)
print(resnet34)
torch.save(resnet34.state_dict(), "ResNet34.pth")
resnet101 = models.resnet34(pretrained=True)
print(resnet101)
torch.save(resnet101.state_dict(), "ResNet101.pth")
# %%

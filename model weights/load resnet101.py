# %%
from torchvision import models
import torch

resnet101 = models.resnet101(pretrained=True)
torch.save(resnet101.state_dict(), "ResNet101.pth")
# %%

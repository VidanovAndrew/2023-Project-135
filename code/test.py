import torch
from torch import nn
import torchvision

from lightly.data import LightlyDataset
from lightly.data.collate import VICRegCollateFunction
from lightly.models.modules import BarlowTwinsProjectionHead
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from lightly.loss import VICRegLoss
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


class VICReg(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead(2048, 2048, 2048)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        if self.training:
          out = self.projection_head(x)
        else:
          out = x
        return out


def Test():
    resnet = torch.hub.load('facebookresearch/vicreg:main', 'resnet50')
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    model_pre = VICReg(backbone)
    model_pre.load_state_dict(torch.load('model_parameters'))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_pre.to(device)

    cifar10 = torchvision.datasets.CIFAR10("datasets/cifar10", train=True, download=True)
    dataset = LightlyDataset.from_torch_dataset(cifar10)
    # or create a dataset from a folder containing images or videos:
    # dataset = LightlyDataset("path/to/folder")

    collate_fn = VICRegCollateFunction(input_size=32)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )

    model_pre.eval()
    clf = LogisticRegression(random_state=0)
    for (x0, x1), _1, _2 in dataloader:
        clf.fit(model_pre(x0.to(device)).cpu().detach().numpy(), _1)
        break

    cifar10_test = torchvision.datasets.CIFAR10("datasets/cifar10", train=False, download=True)
    testset = LightlyDataset.from_torch_dataset(cifar10_test)
    # or create a dataset from a folder containing images or videos:
    # dataset = LightlyDataset("path/to/folder")

    collate_fn = VICRegCollateFunction(input_size=32)

    test = torch.utils.data.DataLoader(
        testset,
        batch_size=256,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )

    for (x0, x1), _1, _2 in test:
        print(model_pre(x0.to(device)).shape)
        print("f1_score: ", f1_score(_1, clf.predict(model_pre(x0.to(device)).cpu().detach().numpy()), average='macro'))
        print("Accuracy: ", accuracy_score(_1, clf.predict(model_pre(x0.to(device)).cpu().detach().numpy())))
        break
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
        # не использовать при обучении
        self.projection_head = BarlowTwinsProjectionHead(512, 2048, 2048)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        if self.training:
          out = self.projection_head(x)
        else:
          out = x
        return out

def Train():
    resnet = torch.hub.load('facebookresearch/vicreg:main', 'resnet50')
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    model_pre = VICReg(backbone)

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

    criterion = VICRegLoss()
    optimizer = torch.optim.SGD(model_pre.parameters(), lr=0.06)

    print("Starting Training")
    model_pre.train()
    for epoch in range(10):
        total_loss = 0
        for (x0, x1), _, _ in dataloader:
            x0 = x0.to(device)
            x1 = x1.to(device)
            z0 = model_pre(x0)
            z1 = model_pre(x1)
            loss = criterion(z0, z1)
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(dataloader)
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}") 
    torch.save(model_pre.state_dict(), 'model_parameters')   


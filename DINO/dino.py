from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models import resnet50
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

"""
gt and gs -> same architecture
different set of weights
θt ← λθt + (1 − λ)θs ---> λ follows cosine schedule from 0.996 to 1

gt and gs is a ResNet followed by a 3 Layer MLP (hidden dim = 2048, l2 norm, weight norm FCL 

links: 
EMA - https://www.zijianhu.com/post/pytorch/ema/
Transforms -  https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
PyTorch-MNIST -  https://gist.github.com/kdubovikov/eb2a4c3ecadd5295f68c126542e59f0a
"""


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.resnet = resnet50()
        self.size_proj = 3
        self.hidden_dim = 2048
        self.K = 256
        self.projection_head, self.last_layer = self.make_projection_head()

    def make_projection_head(self):
        layers = []
        for i in range(self.size_proj):
            if i == 0:
                l = nn.Linear(in_features=1000, out_features=self.hidden_dim)
            else:
                l = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
            layers.append(l)
            layers.append(nn.GELU())
        # add l2 norm layer
        layers.append(nn.Linear(in_features=self.hidden_dim, out_features=self.K))

        last_layer = weight_norm(nn.Linear(in_features=self.K, out_features=self.K, bias=False))
        return nn.Sequential(*layers), last_layer

    def forward(self, x):
        chunks = len(x)
        x = torch.cat(x)
        x = self.resnet(x)
        x = self.projection_head(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        x = x.chunk(chunks)
        return x


class TrainData:
    def __init__(self, root, transform=True, batch_size=10):
        self.root = root
        self.transform = transform
        self.batch_size = batch_size

    def load_data(self):
        if self.transform:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,)),
                                            transforms.ColorJitter(),
                                            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                                            transforms.RandomSolarize(threshold=192.0)])
            mnist = MNIST(self.root, train=True, transform=transform)
        else:
            mnist = MNIST(self.root, train=True)
        return torch.utils.data.DataLoader(mnist, batch_size=self.batch_size)


class DINO(nn.Module):
    def __init__(self):
        super(DINO, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.student = BaseModel().to(self.device)
        self.teacher = BaseModel().to(self.device)
        self.batch_size = 1
        self.base_lr = 0.0005 * self.batch_size / 256
        self.decay = 0.9995
        self.global_view_cropper = transforms.RandomResizedCrop(24, (0.3, 1))
        self.local_view_cropper = transforms.RandomResizedCrop(24, (0.05, 0.3))
        self.dataset = TrainData("./data", batch_size=self.batch_size).load_data()
        self.epochs = 10
        self.tps = 0.1
        self.tpt = 0.04  # linear warm-up from 0.04 to 0.07 during the first 30 epochs
        self.C = 0
        self.center_momentum_rate = 0.9
        # self.network_momentum_rate =
        self.optimizer = self.get_optimizer(optim="SGD")
        self.scheduler = self.get_scheduler()

    def get_optimizer(self, optim="SGD"):
        if optim == "SGD":
            return SGD(self.student.parameters(), lr=self.base_lr)

    def get_scheduler(self):
        return CosineAnnealingLR(self.optimizer, T_max=10, eta_min=0)

    def augment(self, x):
        """
        returns random view of x
        """

        global_view = [self.global_view_cropper(x.expand(1,3,28,28)) for _ in range(2)]
        local_view = [self.local_view_cropper(x.expand(1,3,28,28)) for _ in range(6)]

        return global_view + local_view

    def augment_single(self, x):
        return self.global_view_cropper(x)

    def H_single(self, t, s):
        t = t.detach()
        s = F.softmax(s / self.tps, dim=1)
        t = F.softmax((t - self.C) / self.tpt, dim=1)
        return -  (t * torch.log(s)).sum(dim=1).mean()

    def H_multi(self, t, s):
        student_temp = [F.softmax(i / self.tps) for i in s]
        teacher_temp = [F.softmax((i - self.C) / self.tpt).detach() for i in t]

        total_loss = 0
        n_loss_terms = 0

        for i, x in enumerate(student_temp):
            for j, y in enumerate(teacher_temp):
                if i == j:
                    continue
                loss = torch.sum(-y * torch.log(x), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms
        print(f"Total Loss: {total_loss}")
        return total_loss

    def train_wo_multicrop(self):
        for epoch in range(self.epochs):
            for x in self.dataset:
                self.optimizer.zero_grad()
                x1, x2 = self.augment_single(x), self.augment_single(x)
                s1, s2 = self.student(x1), self.student(x1)
                t1, t2 = self.teacher(x1), self.teacher(x1)

                loss = self.H(t1, s2) / 2 + self.H(t2, s1) / 2
                loss.backward()
                self.optimizer.step()
                if epoch + 1 > 10:
                    self.scheduler.step()

                if epoch + 1 < 30:
                    self.tpt += (0.03 / 30)  # 0.04 -> 0.07: diff = 0.03, step per epoch 0.03 / 30

                student_params = OrderedDict(self.student.named_parameters())
                teacher_params = OrderedDict(self.teacher.named_parameters())
                assert student_params.keys() == teacher_params.keys()

                for name, param in student_params.items():
                    teacher_params[name].sub((1. - self.decay) * (teacher_params[name] - param))

                self.C = self.center_momentum_rate * self.C + (1 - self.center_momentum_rate) * torch.cat(
                    [t1, t2]).mean(dim=0)

    def train_multicrop(self):
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.load_state_dict(self.student.state_dict())
        for epoch in range(self.epochs):
            for x in self.dataset:
                # x = [img.to(self.device) for img in x]
                x = x[0].to(self.device)
                self.optimizer.zero_grad()
                crops = self.augment(x)
                s = self.student(crops)
                t = self.teacher(crops[:2])

                loss = self.H_multi(t, s)
                loss.backward()
                self.optimizer.step()

                if epoch + 1 > 10:
                    self.scheduler.step()

                if epoch + 1 < 30:
                    self.tpt += (0.03 / 30)  # 0.04 -> 0.07: diff = 0.03, step per epoch 0.03 / 30

                student_params = OrderedDict(self.student.named_parameters())
                teacher_params = OrderedDict(self.teacher.named_parameters())
                assert student_params.keys() == teacher_params.keys()

                for name, param in student_params.items():
                    teacher_params[name].sub((1. - self.decay) * (teacher_params[name] - param))

                self.C = self.center_momentum_rate * self.C + (1 - self.center_momentum_rate) * torch.cat(t).mean(dim=0)


if __name__ == '__main__':
    model = DINO()
    model.train_multicrop()
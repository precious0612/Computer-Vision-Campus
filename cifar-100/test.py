'''
[CIFAR-100分类]
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
https://github.com/weiaicunzai/pytorch-cifar100

[环境要求]
- torch
- numpy
- torchvision
- matplotlib

[数据]
在 datasets.CIFAR100 中设置 download=True 即可自动下载
'''


from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


# CIFAR-100数据集图像的均值和方差
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

# 训练数据使用随机切割、水平翻转、随机旋转的数据增强，并进行归一化处理
# 验证数据仅使用归一化，不进行数据增强
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ]),
}

# 搭建训练和验证数据集
Mode = {'train':True, 'val':False}

image_datasets = {x: datasets.CIFAR100(root=os.path.join('cifar', x), train=Mode[x],
                                          transform=data_transforms[x],
                                          download=True)
                  for x in ['train', 'val']}
# 在linux系统中可以使用多个子进程加载数据，而在windows系统中不能。所以在windows中要将DataLoader中的num_workers设置为0或者采用默认为0的设置？
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=128,
#                                               shuffle=Mode[x], num_workers=4)
#                for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=128,
                                              shuffle=Mode[x], num_workers=0)
               for x in ['train', 'val']}

# 获取数据集大小
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# 获取数据集的类别
class_names = image_datasets['train'].classes



# 图像展示函数
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.1)

# 从数据集中取出一组样本
inputs, classes = next(iter(dataloaders['train']))

# 将一组样本拼成一幅图像
out = torchvision.utils.make_grid(inputs)

# 在屏幕中展示图像
plt.ion()
imshow(out, title=[class_names[x] for x in classes])


# ResNet18的基本模块
class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # 定义残差学习函数
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # 定义短路连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


# 定义通用的ResNet结构
class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=100):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        # 生成一个ResNet的基本单元
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # 向前传播
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output


# 定义ResNet18
def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])








# 搭建模型
model_ft = resnet18()

# 重新设置模型的最后一层全连接层的输出通道数
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

# 设置使用GPU或CPU
device = torch.device("mps" if torch.back.is_available() else "cpu")


# 设置模型为GPU或CPU
model_ft = model_ft.to(device)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器
optimizer_ft = optim.Adam(model_ft.parameters(), lr=2e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)













# 模型训练函数
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # 定义记录最优模型以及最高准确率的变量
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 每一个epoch包括训练和验证两个阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为验证模式

            running_loss = 0.0
            running_corrects = 0

            # 在训练数据上迭代
            for inputs, labels in dataloaders[phase]:
                # 设置输入图像和标签为GPU或CPU
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 优化器的参数归零
                optimizer.zero_grad()

                # 向前传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 若为训练模式，则反传梯度并更新模型参数
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 记录信息
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # 若为训练模式，则更新优化器的参数
            if phase == 'train':
                scheduler.step()

            # 记录并打印信息
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 保存在验证集上性能最优的模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    # 打印信息
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 加载性能最优的模型
    model.load_state_dict(best_model_wts)
    return model

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)





''' 

## 直接调用torchvision.models中的resnet18的方式 

# 加载ImageNet预训练的参数
model_conv = models.resnet18(pretrained=True)

# 重新设置模型的最后一层全连接层的输出通道数
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, len(class_names))

# 设置模型为GPU或CPU
model_conv = model_conv.to(device)

# 定义优化器
# 此处指定仅更新model_conv.fc的参数
# 以此实现特征提取卷积层的参数固定
optimizer_conv = optim.Adam(model_conv.fc.parameters(), lr=2e-4)

# 设置优化器的更新策略
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

# 模型训练
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)

'''

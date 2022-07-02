'''
Train CDS-Large on CIFAR 10
Code based on https://github.com/kuangliu/pytorch-cifar and https://github.com/davidcpage/cifar10-fast
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from skimage import color

import os
import argparse

from model import CDS_large
from utils import count_params
from collections import namedtuple
import numpy as np
import logging
import time
import tqdm
import coloredlogs


def labbify(x):
    x = torch.tensor(color.rgb2lab(x.numpy().transpose(1, 2, 0)).transpose(
        2, 0, 1)).type(torch.FloatTensor)/255.0
    y = torch.stack([torch.stack([x[0], 0*x[0]], dim=0),
                    torch.stack([x[1], x[2]], dim=0)], dim=0)
    return y


class PiecewiseLinear(namedtuple('PiecewiseLinear', ('knots', 'vals'))):
    def __call__(self, t):
        return np.interp([t], self.knots, self.vals)[0]


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--epochs', default=200,
                    type=int, help='number of epochs')
parser.add_argument('--resume', '-r', default=None, type=str,
                    help='resume from checkpoint')

args = parser.parse_args()


logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] %(message)s", handlers=[
                    logging.FileHandler(f'{int(time.time())}.log'), logging.StreamHandler()])
log = logging.getLogger()
coloredlogs.install(level='DEBUG', logger=log)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_val_acc = 0  # best test accuracy
best_test_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


lr_schedule = PiecewiseLinear([0, 10, 100, 120, 150, 200], [
                              0.01, 0.2, 0.01, 0.001, 0.0001, 0.0001])

# Data
print('==> Preparing data..')
transform_train = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    labbify

]

transform_test = [
    transforms.ToTensor(),
    labbify
]


transform_train = transforms.Compose(transform_train)
transform_test = transforms.Compose(transform_test)

outsize = 10
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)


total_size = len(trainset)
train_size = int(total_size*0.9)
val_size = total_size - train_size

trainset, valset = torch.utils.data.random_split(
    trainset, [train_size, val_size])


trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=256, shuffle=True, num_workers=2)

valloader = torch.utils.data.DataLoader(
    valset, batch_size=64, shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
net = CDS_large(outsize=outsize)


net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


print(f"#Parameters: {str(count_params(net)/1e6)[:3]}M")

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['net'])
    best_val_acc = checkpoint['val_acc']
    best_test_acc = checkpoint['test_acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)


def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader), desc=f'Epoch {epoch}'):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()


def test(epoch):
    global best_val_acc
    global best_test_acc
    net.eval()
    val_loss = 0  # First we do val
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        log.info(f"val acc {100*correct/total}")

    val_acc = 100.*correct/total

    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        log.info(f"test acc {100*correct/total}")

    # Save checkpoint.
    test_acc = 100.*correct/total
    if (val_acc > best_val_acc):
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'val_acc': val_acc,
            'test_acc': test_acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(
            state, f'./checkpoint/ckpt_{epoch}_{val_acc}_{test_acc}.pth')
        best_val_acc = val_acc
        best_test_acc = test_acc


for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)
    test(epoch)
    for g in optimizer.param_groups:
        g['lr'] = lr_schedule(epoch)

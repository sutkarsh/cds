import torch
import numpy as np
from skimage import color
import torchvision
import logging
import os
import random

LOG = logging.getLogger('base')

os.environ['PYTHONHASHSEED'] = str(0)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def convert_real_imag(inputs):
    # Converts a [B, C, H, W] input into a [B, 2, C, H, W] complex representation
    real = np.real(inputs)
    imag = np.imag(inputs)
    return np.stack([real, imag], axis=1)


def generate_MSTAR_dataloader(split_percent=10, data_path='../MSTAR/', train_batch=256, test_batch=256, val_batch=256, seed=0, normalize_mag=True, random_seed=False, mag_only=False, *args, **kwargs):

    # Generates MSTAR dataloader
    # where percent_train is just random split with **percent_train** % for training and rest for test.
    if random_seed:
        seed = np.random.randint(10000)
        LOG.info("Random seed "+str(seed))

    # Fix the np seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load the input
    X_ = np.load(os.path.join(data_path, 'X_data.npy'))
    y_ = np.load(os.path.join(data_path, 'Y_data.npy'))

    train_idx = np.load(os.path.join(data_path, 'splits/train.npy'))
    test_idx = np.load(os.path.join(data_path, 'splits/test.npy'))

    percentage = int(split_percent)
    assert 0 < percentage <= 100, "Split percentage invalid"

    train_idx = np.random.permutation(train_idx)
    split_idx = int(len(train_idx)*percentage/100)
    train_idx = train_idx[:split_idx]
    split_idx = int(len(train_idx)*90/100)
    train_idx, val_idx, test_idx = train_idx[:
                                             split_idx], train_idx[split_idx:], test_idx

    LOG.info("Using {} Train, {} Validation, and {} Test examples".format(
        train_idx.size, val_idx.size, test_idx.size))

    x_train = X_[train_idx]
    x_test = X_[test_idx]
    if len(val_idx) > 0:
        x_val = X_[val_idx]

    if normalize_mag:
        mag = np.abs(x_train)
        mag_max = mag.max()

        x_train /= mag_max
        x_test /= mag_max
        if len(val_idx) > 0:
            x_val /= mag_max

    x_train = x_train[:, None, ...]  # BHW -> BCHW
    x_test = x_test[:, None, ...]  # BHW -> BCHW
    x_val = x_val[:, None, ...]  # BHW -> BCHW

    if mag_only:
        x_train = np.linalg.norm(x_train, axis=1)
        x_test = np.linalg.norm(x_test, axis=1)
        x_val = np.linalg.norm(x_val, axis=1)

    data_train = torch.utils.data.TensorDataset(torch.from_numpy(x_train).type(
        torch.complex64), torch.from_numpy(y_[train_idx]).type(torch.LongTensor))
    data_test = torch.utils.data.TensorDataset(torch.from_numpy(x_test).type(
        torch.complex64), torch.from_numpy(y_[test_idx]).type(torch.LongTensor))
    data_val = torch.utils.data.TensorDataset(torch.from_numpy(x_val).type(
        torch.complex64), torch.from_numpy(y_[val_idx]).type(torch.LongTensor))

    params_train = {'batch_size': train_batch,
                    'shuffle': True,
                    'worker_init_fn': worker_init_fn}
    params_test = {'batch_size': test_batch,
                   'shuffle': False,
                   'worker_init_fn': worker_init_fn}
    params_val = {'batch_size': val_batch,
                  'shuffle': False,
                  'worker_init_fn': worker_init_fn}

    train_generator = torch.utils.data.DataLoader(
        dataset=data_train, **params_train)
    test_generator = torch.utils.data.DataLoader(
        dataset=data_test, **params_test)
    val_generator = torch.utils.data.DataLoader(dataset=data_val, **params_val)

    return train_generator, val_generator, test_generator


def generate_CIFAR_dataloader(split_percent=10, data_path='../CIFAR/', train_batch=256,
                              test_batch=256, val_batch=256, seed=0, normalize_mag=True,
                              lab=False, remove_train=0, dset_type='cifar10',
                              transform=None, sliding=False, *args, **kwargs):

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    assert dset_type in ['cifar10', 'cifar100', 'svhn']
    if dset_type == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(
            root=data_path, train=True, download=True)
        test = torchvision.datasets.CIFAR10(
            root=data_path, train=False, download=True)
        y_train = np.array(train_set.targets)
        y_test = np.array(test.targets)
        x_train = train_set.data
        x_test = test.data
    elif dset_type == 'cifar100':
        train_set = torchvision.datasets.CIFAR100(
            root=data_path, train=True, download=True)
        test = torchvision.datasets.CIFAR100(
            root=data_path, train=False, download=True)
        y_train = np.array(train_set.targets)
        y_test = np.array(test.targets)
        x_train = train_set.data
        x_test = test.data
    else:
        train_set = torchvision.datasets.SVHN(
            root=data_path, split='train', download=True)
        test = torchvision.datasets.SVHN(
            root=data_path, split='test', download=True)
        y_train = np.array(train_set.labels)
        y_test = np.array(test.labels)
        x_train = train_set.data.transpose(0, 2, 3, 1)
        x_test = test.data.transpose(0, 2, 3, 1)

    if transform:
        x_train = np.array([transform(im) for im in x_train])
        x_test = np.array([transform(im) for im in x_test])

    if lab:
        x_train = np.array([color.rgb2lab(np.nan_to_num(x)) for x in x_train])
        x_test = np.array([color.rgb2lab(x) for x in x_test])

    x_train = x_train/255.0
    x_test = x_test/255.0

    train_idx = np.random.permutation(np.arange(x_train.shape[0]))
    x_train = x_train[train_idx].transpose(0, 3, 1, 2)
    y_train = y_train[train_idx]
    x_test = x_test.transpose(0, 3, 1, 2)

    if remove_train != 0:
        total_size = len(x_train)
        print("Original train set size: " + str(total_size))
        print("Using this much data: "+str(int(total_size*remove_train)))
        x_train = x_train[:int(total_size*remove_train)]
        y_train = y_train[:int(total_size*remove_train)]

    total_size = len(x_train)
    trainval_subset = int(split_percent*total_size/100)
    train_size = int(trainval_subset*0.9)
    val_size = int(trainval_subset*0.1)

    x_train, x_val = x_train[:train_size], x_train[train_size:train_size+val_size]
    y_train, y_val = y_train[:train_size], y_train[train_size:train_size+val_size]
    LOG.info("Using {} Train, {} Validation, and {} Test examples".format(
        len(x_train), len(x_val), len(x_test)))

    if lab:
        # 2 Channels containing (L, a+ib)
        x_train = np.stack(
            [x_train[:, 0], x_train[:, 1]+1j*x_train[:, 2]], axis=1)
        x_val = np.stack([x_val[:, 0], x_val[:, 1]+1j*x_val[:, 2]], axis=1)
        x_test = np.stack([x_test[:, 0], x_test[:, 1]+1j*x_test[:, 2]], axis=1)

    if normalize_mag:
        mag = np.abs(x_train)
        mag_max = mag.max()

        x_train = x_train/mag_max
        x_test = x_test/mag_max
        x_val = x_val/mag_max

    if sliding:
        x_train = np.stack(
            [x_train[:, 0]+1j*x_train[:, 1], x_train[:, 1]+1j*x_train[:, 2]], axis=1)
        x_val = np.stack([x_val[:, 0]+1j*x_val[:, 1],
                          x_val[:, 1]+1j*x_val[:, 2]], axis=1)
        x_test = np.stack(
            [x_test[:, 0]+1j*x_test[:, 1], x_test[:, 1]+1j*x_test[:, 2]], axis=1)

    data_train = torch.utils.data.TensorDataset(torch.from_numpy(x_train).type(
        torch.complex64), torch.from_numpy(y_train).type(torch.LongTensor))
    data_val = torch.utils.data.TensorDataset(torch.from_numpy(x_val).type(
        torch.complex64), torch.from_numpy(y_val).type(torch.LongTensor))
    data_test = torch.utils.data.TensorDataset(torch.from_numpy(x_test).type(
        torch.complex64), torch.from_numpy(y_test).type(torch.LongTensor))

    params_train = {'batch_size': train_batch,
                    'shuffle': True,
                    'worker_init_fn': worker_init_fn}
    params_val = {'batch_size': val_batch,
                  'shuffle': False,
                  'worker_init_fn': worker_init_fn}
    params_test = {'batch_size': test_batch,
                   'shuffle': False,
                   'worker_init_fn': worker_init_fn}

    train_generator = torch.utils.data.DataLoader(
        dataset=data_train, pin_memory=True, **params_train)
    val_generator = torch.utils.data.DataLoader(
        dataset=data_val, pin_memory=True, **params_val)
    test_generator = torch.utils.data.DataLoader(
        dataset=data_test, pin_memory=True, **params_test)

    return train_generator, val_generator, test_generator

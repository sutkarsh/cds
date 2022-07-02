import dataloader
import config
import argparse
import os
import sys
import tqdm
import torch
import numpy as np
import model
from utils import *
import random
import coloredlogs
import logging
from torch.utils.tensorboard import SummaryWriter

LOG = logging.getLogger('base')
coloredlogs.install(level='DEBUG', logger=LOG)


def test_model(test_loader, net, current_iter, tracker, writer, logger, save_this_iter):
    net.eval()
    logger.info("Testing....")
    test_losses = []
    test_acc = []
    with torch.no_grad():
        for batch in tqdm.tqdm(iter(test_loader), dynamic_ncols=True):
            x, y = batch
            loss, acc = loss_fn(net(x.cuda()), y.cuda())
            test_losses.append(loss.item())
            test_acc.append(acc.item())
    mean_test_acc = np.sum(test_acc)/len(test_loader.dataset)
    mean_test_loss = np.sum(test_losses)/len(test_loader.dataset)

    writer.add_scalar('Test Loss', mean_test_loss, current_iter)
    writer.add_scalar('Test Acc', mean_test_acc, current_iter)
    logger.info("Finished Testing! Mean Loss: {}, Acc: {}.".format(
        mean_test_loss, mean_test_acc))
    logger.info("Back to Training.")

    if save_this_iter:
        tracker.create_new_save('val_'+str(round(tracker.best_acc, 4))+'_test_'+str(
            round(mean_test_acc, 4))+'_'+str(current_iter+1).zfill(8)+'.pth', net)


def val_model(val_loader, net, current_iter, tracker, writer, logger):
    net.eval()
    logger.info("Validating....")
    val_losses = []
    val_acc = []

    with torch.no_grad():
        for idx, batch in tqdm.tqdm(enumerate(val_loader), dynamic_ncols=True):
            x, y = batch
            loss, acc = loss_fn(net(x.cuda()), y.cuda())
            val_losses.append(loss.item())
            val_acc.append(acc.item())

    mean_val_acc = np.sum(val_acc)/len(val_loader.dataset)
    mean_val_loss = np.sum(val_losses)/len(val_loader.dataset)
    writer.add_scalar('Validation Loss', mean_val_loss, current_iter)
    writer.add_scalar('Validation Acc', mean_val_acc, current_iter)
    logger.info("Finished Validation! Mean Loss: {}, Acc: {}".format(
        mean_val_loss, mean_val_acc))
    return tracker.update(mean_val_acc)


def main(args):
    cfg = config.parse_config(args.config)

    ckpt_dir = args.checkpoint_dir
    setup_ckpt(ckpt_dir)
    setup_logger(LOG, ckpt_dir)
    LOG.info(f"\ncfg: {str(cfg)}")
    LOG.info(f"args: {str(args)}")
    LOG.info(f"system command: {''.join(sys.argv)}")

    if args.save_on is not None:
        save_on = eval(f'[{args.save_on}]')
    else:
        save_on = None

    # Tensorboard
    writer = SummaryWriter(ckpt_dir+'/runs')

    # Random seed
    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)

    net = getattr(model, cfg['model']['name'])(
        cifarnet_config=args.cifarnet_config, **cfg['model']['args'])

    net.cuda()
    net.zero_grad()
    net.train()

    LOG.info(f"# Params: {count_params(net)}")

    optimizer = torch.optim.AdamW(
        params=net.parameters(), lr=args.lr, weight_decay=args.wd)

    if 'args' in cfg['dataset']:
        train_loader, val_loader, test_loader = getattr(dataloader, cfg['dataset']['function'])(
            **cfg['dataset']['args'], seed=args.seed, train_batch=args.bs, val_batch=args.bs, test_batch=args.bs)
    else:
        train_loader, val_loader, test_loader = getattr(dataloader, cfg['dataset']['function'])(
            seed=args.seed, train_batch=args.bs, val_batch=args.bs, test_batch=args.bs)

    # Tracking best validation model
    tracker = Model_Tracker(ckpt_dir, LOG)
    save_this_iter = False

    train_iter = iter(train_loader)
    LOG.info("train_iter_size " + str(len(train_iter)))
    LOG.info("testloader " + str(test_loader))

    for current_iter in tqdm.trange(args.num_iters+1, dynamic_ncols=True):
        if ((current_iter % args.val_every) == 0) and current_iter > 0:
            LOG.info(f"Last Training Batch Acc: {acc.item()}")
            if val_loader:
                save_this_iter = val_model(
                    val_loader, net, current_iter, tracker, writer, LOG)
            if save_this_iter:
                test_model(test_loader, net, current_iter,
                           tracker, writer, LOG, save_this_iter)
                save_this_iter = False

            net.zero_grad()
            net.train()

        optimizer.zero_grad()

        ret = next(train_iter, None)  # Reset trainloader if empty
        if ret is None:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        else:
            x, y = ret

        pred = net(x.cuda())
        loss, acc = loss_fn(pred, y.cuda())
        loss.backward()
        optimizer.step()

        if (current_iter % args.log_every) == 0:
            writer.add_scalar('Train Loss', loss.item(), current_iter)
            writer.add_scalar('Train Acc', acc.item(), current_iter)

        if (save_on is not None) and (current_iter in save_on):
            os.makedirs(f'{ckpt_dir}/saves/', exist_ok=True)
            torch.save(
                net, f"{ckpt_dir}/saves/{str(current_iter).zfill(6)}.pth")

    current_iter += 1
    LOG.info("Done Training!")

    if val_loader:
        save_this_iter = val_model(
            val_loader, net, current_iter, tracker, writer, LOG)

    LOG.info("Done. Exiting")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0,
                        help="global random seed.")
    parser.add_argument("--config", default='config/default.yml',
                        help="path to a config file (optional). can use colon to separate multiple config files: in this case they are applied sequentially to the default config.")

    parser.add_argument("--checkpoint_dir", default='./ckpt/',
                        help="Output directory where checkpoints are saved")

    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate for the optimizer")

    parser.add_argument("--bs", type=int, default=256, help="Batch size")

    parser.add_argument("--wd", type=float, default=0, help="Weight Decay")

    parser.add_argument("--num_iters", type=int, default=500000,
                        help="Number of batches to train for")

    parser.add_argument("--log_every", type=int, default=50,
                        help="Log losses every N iterations")

    parser.add_argument("--cifarnet_config", default=None,
                        help="Which type of cifarnet config to use")

    parser.add_argument("--val_every", type=int, default=1000,
                        help="validate every N iterations")

    parser.add_argument("--deterministic", action='store_true', default=False,
                        help="Use torch's deterministic algorithms. Breaks some models")

    parser.add_argument("--save_on", type=str, default=None,
                        help="List of iters on which to save")

    parser.add_argument("--note", type=str, default=None,
                        help="any additional notes or remarks")

    parser.set_defaults()

    args = parser.parse_args()

    main(args)

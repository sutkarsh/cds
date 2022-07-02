import logging
import coloredlogs
import os
import torch


def count_params(model): return sum(p.numel()
                                    for p in model.parameters() if p.requires_grad)


def setup_ckpt(ckpt_dir):
    if os.path.exists(ckpt_dir):
        print("FOLDER EXISTS")
    os.system('rm -r '+ckpt_dir)
    os.makedirs(ckpt_dir+'/runs', exist_ok=True)
    os.makedirs(ckpt_dir+'/code', exist_ok=True)
    # Makes a copy of all local code files to the log directory, just in case there are any problems
    os.system('cp *.py '+ckpt_dir+'/code/')
    # Makes a copy of all config files
    os.system('cp -r config '+ckpt_dir+'/code/')


LOG = logging.getLogger('base')
coloredlogs.install(level='DEBUG', logger=LOG)


def setup_logger(LOG, checkpoint_dir, debug=True):
    '''set up logger'''

    formatter = logging.Formatter('%(asctime)s [%(process)d] %(levelname)s %(name)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    LOG.setLevel(logging.DEBUG)

    os.makedirs(checkpoint_dir, exist_ok=True)
    log_file = os.path.join(checkpoint_dir, "run.log")
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    LOG.addHandler(fh)


class Model_Tracker:
    def __init__(self, path, LOG):
        self.path = path
        self.name = None
        self.best_acc = 0
        self.LOG = LOG

        if self.path[-1] != '/':
            self.path = self.path+'/'

    def remove_old_save(self):
        if self.name:
            self.LOG.info("Removing Old Model Checkpoint at " +
                          self.path+self.name)
            os.remove(os.path.join(self.path, self.name))

    def create_new_save(self, name, net):
        self.LOG.info("New Best Validation Accuracy: "+str(self.best_acc))
        self.remove_old_save()
        self.name = name
        self.LOG.info("Creating New Model Checkpoint at " +
                      os.path.join(self.path, self.name))
        torch.save(net, os.path.join(self.path, self.name))

    def update(self, new_acc):
        better = new_acc > self.best_acc
        if better:
            self.best_acc = new_acc
        return better


xent = torch.nn.CrossEntropyLoss()


def loss_fn(x, y):
    return xent(x, y), (torch.argmax(x, dim=-1) == y).type(torch.cuda.FloatTensor).sum()

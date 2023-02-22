import shutil
import numpy as np
import torch
from torch import nn
from torch.utils.data import SubsetRandomSampler
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score


def train_val_test_split(dataset, batch_size=64,
                         train_ratio=0.6, valid_ratio=0.2,
                         test_ratio=0.2, num_workers=1,
                         **kwargs):

    total_size = len(dataset)
    index_list = list(range(total_size))
    if kwargs['train_size']:
        train_size = kwargs['train_size']
    else:
        train_size = int(train_ratio * total_size)
    if kwargs['valid_size']:
        valid_size = kwargs['valid_size']
    else:
        valid_size = int(valid_ratio * total_size)
    if kwargs['test_size']:
        test_size = kwargs['test_size']
    else:
        test_size = int(test_ratio * total_size)
    train_sampler = SubsetRandomSampler(index_list[:train_size])
    valid_sampler = SubsetRandomSampler(index_list[train_size:train_size + valid_size])
    test_sampler = SubsetRandomSampler(index_list[-test_size:])
    train_loader = DataLoader(dataset=dataset, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers)
    valid_loader = DataLoader(dataset=dataset, sampler=valid_sampler, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(dataset=dataset, sampler=test_sampler, batch_size=batch_size, num_workers=num_workers)
    return train_loader, valid_loader, test_loader


def mae_metric(prediction, target):

    mae = torch.mean(torch.abs(prediction - target))
    return mae

def class_metric(prediction, target):

    probability = nn.functional.softmax(prediction, dim=1)
    probability = probability.cpu().detach().numpy()
    target = target.detach().numpy()
    y_pred = np.argmax(probability, axis=1)
    accuracy = accuracy_score(target, y_pred)
    return accuracy


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, transfer=False, filename='checkpoint.pth.tar'):

    file = 'weights/' + filename
    torch.save(state, file)
    if is_best or transfer:
        shutil.copyfile(file, 'weights/model_best.pth.tar')

class Normalizer(object):

    def __init__(self, tensor, atom_ref):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)
        self.atom_ref = atom_ref

    def norm(self, tensor):
        if self.atom_ref is not None:
            return tensor
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        if self.atom_ref is not None:
            return normed_tensor
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='weights/checkpoint.pth.tar', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def get_full_dims():
    return np.array([100, 18, 9, 4, 10, 10, 10, 10, 10])

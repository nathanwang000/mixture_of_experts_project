from torch import distributions, nn
from functools import partial
import string, random, os
import collections
import copy
import torch, copy
from torch.utils import data
import tqdm
import pandas as pd
from scipy.linalg import block_diag
import math
import numpy as np
import matplotlib.pyplot as plt
from models import MoO, AdaptiveGate, MLP
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

def get_subset_batch(dataset, indices):
    '''return a batch of data given indices'''
    return next(iter(DataLoader(Subset(dataset, indices), batch_size=len(indices))))

def random_string(N=5):
    return ''.join(random.choice(string.ascii_uppercase + string.digits)
                   for _ in range(N))

def random_split_dataset(dataset, proportions, seed=None):
    n = len(dataset)
    ns = [int(math.floor(p*n)) for p in proportions]
    ns[-1] += n - sum(ns)

    def random_split(dataset, lengths):
        if sum(lengths) != len(dataset):
            raise ValueError("Sum of input lengths does not equal\
            the length of the input dataset!")

        if seed is not None:
            np.random.seed(seed)
        indices = np.random.permutation(sum(lengths))
        return [torch.utils.data.Subset(dataset, indices[offset - length:offset])\
                for offset, length in zip(torch._utils._accumulate(lengths), lengths)]
    #return torch.utils.data.random_split(dataset, ns)
    return random_split(dataset, ns)

def getXY(loader):
    X, Y = [], []
    for x, y in loader:
        x, y = x.numpy(), y.numpy().ravel()
        X.append(x)
        Y.append(y)
    X, Y = np.vstack(X), np.hstack(Y)
    return X, Y

def get_output(net, loader, device='cuda'):
    net.eval()
    o = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        o.append(net(x).detach().cpu().numpy())
    net.train()        
    return np.vstack(o)

def get_y(loader):
    Y = []
    for x, y in loader:
        Y.append(y.cpu().numpy())
    return np.hstack(Y)

def get_x(loader):
    X = []
    for x, y in loader:
        X.append(x.cpu().numpy())
    return np.vstack(X)

def train_sklearn(net, loader):
    net.fit(*getXY(loader))
    return net

def sklearn_acc(net, loader):
    X, Y = getXY(loader)
    return (net.predict(X) == Y).mean()

def get_criterion(net, loader, criterion):
    net.eval()
    c = criterion
    n = 0
    ll = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            o = net(x)
            l = c(o, y).mean()
            bs = x.shape[0]
            ll += l.detach().cpu().numpy() * bs
            n += bs

    net.train()            
    return ll / n        

get_ll = partial(get_criterion, criterion=lambda o,y: -nn.CrossEntropyLoss()(o, y))

def get_ll_first_entry(idx):

    def f(net, loader):
        c = nn.CrossEntropyLoss()
        n = 0
        ll = 0
        with torch.no_grad():
            for x, y in loader:
                criteria = (x[:,0]==idx)
                if criteria.sum() == 0: continue
                o = net(x[criteria])
                l = -c(o, y[criteria]).mean()
                bs = x.shape[0]
                ll += l.detach().cpu().numpy() * bs
                n += bs
            
        return ll / n        
        
    return f

def get_acc(net, loader):
    o = []
    t = []
    with torch.no_grad():
        for x, y in loader:
            o.append(net(x).detach().cpu().numpy())
            t.append(y)
    o, t = np.vstack(o), np.hstack(t)
    return (o.argmax(1) == t).mean()


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha

def genCovX(C, n): # helper function to create N(0, C)
    # C is the covariance matrice (assume to be psd)
    # n is number of examples
    C = np.array(C)
    A = np.linalg.cholesky(C)
    d, _ = C.shape
    Z = np.random.randn(n, d)
    X = Z.dot(A.T) 
    return X.astype(np.float32)
        
def bdot(a, b): # batch wise dot product
    return (a * b).sum(-1).unsqueeze(-1)

def forward(theta, x):
    return bdot(theta, x)

def getCov(group_corr, n_per_groups, d):
    assert sum(n_per_groups) == d, 'must sum to d'
    blocks = []
    for n_per_group in n_per_groups:
        base = np.diag(np.ones(n_per_group))
        base[base == 0] = group_corr
        blocks.append(base)
    covariance = block_diag(*blocks)
    return covariance

def sample_simplex(n, d, uniform=False):
    if uniform:
        # was using torch.nn.functional.softmax(torch.randn(n,d), dim=1)
        u = torch.zeros(n, d+1)
        u[:, -1] = 1
        u[:,1:-1] = torch.rand(n, d-1)
        u, _ = u.sort(dim=1)
        u = u[:, 1:] - u[:, :-1]
        return u
    else: # use dirchlet distribution
        u = np.random.dirichlet(1 * np.ones(d), size=n)
        u = torch.from_numpy(u).float()
        return u

def early_stop(net, val_loader, named_criterion, patience):
    g = {'count': 0} # python 2.x work around
    values = []
    name, criterion, lower_is_better = named_criterion
    
    def f():
        v = criterion(net, val_loader)
        val_name_value = (name, v)        
        v = -v if lower_is_better else v

        if values and v >= max(values):
            g['count'] = 0
            update_best_model = True
        else:
            g['count'] += 1
            update_best_model = False

        values.append(v)                
        if g['count'] > patience:
            return update_best_model, True, val_name_value

        return update_best_model, False, val_name_value
    
    return f

def train(net, loader, criterion, opt, n_epochs, verbose=False,
          report_criteria=[], report_every=1,
          # early stopping parameters
          val_loader=None, patience=10, es_named_criterion=(None, None, False),
          # save model parameters
          savename=None, test_loader=None):
    # report_criteria: List[(name, f(net, loader))]
    train_log = []
    net.train()
    losses = []
    n = len(loader.dataset)
    es = early_stop(net, val_loader, es_named_criterion, patience)
    net_best = net
    for i in range(n_epochs):

        if savename:
            os.system('mkdir -p {}'.format(savename))
            torch.save(net, savename + '/epoch{}.m'.format(i))
        
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            opt.zero_grad()
            o = net(x)
            l = criterion(o, y).mean()
            l.backward()
            opt.step()            
            losses.append(l.detach().item())

        # early stopping
        stop = False
        do_es = (val_loader and es_named_criterion[1] is not None)
        if do_es:
            update_best_model, stop, val_name_value = es()
            if update_best_model:
                net_best = copy.deepcopy(net)
            
        # for reporting
        if verbose and (stop or i % report_every == 0):
            train_report = dict((name, f(net, loader)) for name, f in report_criteria)
            train_report['loss'] = np.mean(losses[-len(loader):])
            
            # print out report
            print('epoch {:>3}: '.format(i) +
                  ' '.join('{} {:.3f}'.format(
                      name, val
                  ) for name, val in train_report.items()) +
                  (" val_{} {:.3f}".format(*val_name_value) if do_es else "") +
                  (" early stopping..." if stop else ""))

            # update log
            if do_es:
                name, value = val_name_value
                train_report.update(dict([("val_" + name, value)])) 
            if val_loader:
                val_report = dict(('val_' + name, f(net, val_loader))\
                                  for name, f in report_criteria)
                train_report.update(val_report)
            if test_loader:
                test_report = dict(('test_' + name, f(net, test_loader))\
                                  for name, f in report_criteria)
                train_report.update(test_report)
            
            train_report.update({'epoch': i})
            train_log.append(train_report)

        if stop:
            break

    return net_best, train_log

def test(net, loader, criterion):
    net.eval()
    losses = []
    total = 0
    for x, y in loader:
        o = net(x)
        l = criterion(o, y).mean()
        bs = o.shape[0]
        total += bs        
        losses.append(l.detach().item() * bs)
    net.train()
    return sum(losses) / total

def test_ensemble(num_experts, f, loader, criterion, n_samples=10):
    losses = []
    total = 0
    for x, y in loader:
        o_s = []
        for i in range(n_samples):
            # instead of doing theta[0], we reduce variance of l estimation
            # see local reparametrization trick            
            theta = sample_simplex(x.shape[0], num_experts)
            o_s.append(f(theta, x))
        o = sum(o_s) / len(o_s)
        l = criterion(o, y).mean()
        bs = o.shape[0]
        total += bs        
        losses.append(l.detach().item() * bs)
    return sum(losses) / total
    
def train_random_gate(num_experts, f, loader, criterion, opt, n_epochs):

    losses = []
    n = len(loader.dataset)
    for i in tqdm.tqdm(range(n_epochs)):
        for x, y in loader:
            theta = sample_simplex(x.shape[0], num_experts) 
            o = f(theta, x)
            opt.zero_grad() 
            l = criterion(o, y).mean()
            l.backward()
            opt.step()
            
            losses.append(l.detach().item())
    return losses

def train_pseudo_data(net, f, alpha, loader, criterion, opt, n_epochs):

    losses = []
    n = len(loader.dataset)
    for i in tqdm.tqdm(range(n_epochs)):
        for x, y in loader:
            # don't need to do reparametrization trick b/c we are not 
            # doing ensemble
            theta = net(x)[0] 
            o = f(theta, x)
            opt.zero_grad() 
            l = criterion(o, y)
            bs = l.shape[0]
            total = alpha + n
            l = l.reshape(-1) * torch.tensor([(alpha+1)/total] + [(n-1)/total/bs] * (bs-1)).float()
            l = l.sum()
            l.backward()
            opt.step()
            
            losses.append(l.detach().item())
    return losses

def test_pseudo_data(net, loader, forward):
    losses = []
    total = 0
    for x, y in loader:
        theta = net(x)
        o = forward(theta, x)
        l = criterion(o, y).mean()
        bs = o.shape[0]
        total += bs        
        losses.append(l.detach().item() * bs)
    return sum(losses) / total

def gen_corr_data(f, covariance, n, shuffle=True, bs=10):
    X = genCovX(covariance, n)
    X = torch.from_numpy(X)
    y = f(X).detach()
    dataset = data.TensorDataset(X, y)
    loader = data.DataLoader(dataset, batch_size=bs, shuffle=shuffle)
    return loader
            

'''
jw: code for comparing different methods of clustering and combining hetergeneous groups
in MIMIC iii data for the mortality task
'''
from functools import partial
import copy, glob

import torch
from torch import nn
import torch.utils.data as data
torch.set_num_threads(1)

import os, tqdm
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib

from run_mortality_prediction import stratified_split, bootstrap_predict
from models import Global_MIMIC_Model, MoE_MIMIC_Model, MTL_MIMIC_Model, Separate_MIMIC_Model
from utils import train, get_criterion, get_output, load_data

def get_args(): # adapted from run_mortality_prediction.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str, default='mimic',
                        choices=['mimic', 'eicu'],
                        help="indicating which data to run. Type: String.")
    parser.add_argument("--result_dir", type=str, default='result',
                        help="where result are saved. Type: String.")    
    parser.add_argument("--eicu_cohort", type=str, default='ARF4',
                        choices=['ARF4', 'ARF12', 'Shock4', 'Shock12', 'mortality'],
                        help="the cohort for eicu")    
    parser.add_argument("--runname", type=str, default=None, help="setting name (default None)")    
    parser.add_argument("--global_model_fn",
                        type=str, default=None, help="name of the global model to load")        
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate for Adam")   
    parser.add_argument("--wd", type=float, default=0, help="weight decay Adam")
    parser.add_argument("--num_clusters", type=int, default=3, \
        help='Number of clusters for MoE. Type: int. Default: 3.')    
    parser.add_argument("--random_run", action="store_true", default=False,
                        help="run stochstically, including weight initialization")
    parser.add_argument("--result_suffix", type=str, default='',
                        help="this will add to the end of results|models|checkpoints")
    parser.add_argument("--data_hours", type=int, default=24,
                        help="The number of hours of data to use in making the prediction. \
        Type: int. Default: 24.")
    parser.add_argument("--gap_time", type=int, default=12, \
                        help="The gap between data and when we are making predictions. Type: int. Default: 12.")
    parser.add_argument("--model_type", type=str, default='MOE',
                        choices=['GLOBAL', 'MULTITASK', 'SEPARATE', 'MOE', "SNAPSHOT", "MTL_PT"],
                        help="indicating \
        which type of model to run. Type: String.")
    parser.add_argument("--pmt", action="store_true", default=False, help="This is an indicator \
        flag to remove noise in input by using a global model weighted by permutation importance")
    parser.add_argument("--num_lstm_layers", type=int, default=1,
                        help="Number of beginning LSTM layers, applies to all model types. \
        Type: int. Default: 1.")
    parser.add_argument("--lstm_layer_size", type=int, default=16,
                        help="Number of units in beginning LSTM layers, applies to all model types. \
        Type: int. Default: 16.")
    parser.add_argument("--num_dense_shared_layers", type=int, default=0,
                        help="Number of shared dense layers following LSTM layer(s), applies to \
        all model types. Type: int. Default: 0.")
    parser.add_argument("--dense_shared_layer_size", type=int, default=0,
                        help="Number of units in shared dense layers, applies to all model types. \
        Type: int. Default: 0.")
    parser.add_argument("--num_multi_layers", type=int, default=0,
                        help="Number of separate-task dense layers, only applies to multitask models. Currently \
        only 0 or 1 separate-task dense layers are supported for the original code. Type: int. Default: 0.")
    parser.add_argument("--multi_layer_size", type=int, default=0,
                        help="Number of units in separate-task dense layers, only applies to multitask \
        models. Type: int. Default: 0.")
    parser.add_argument("--cohorts", type=str, default='careunit',
                        help="One of {'careunit', 'saps', 'custom'}. Indicates whether to use pre-defined cohorts \
        (careunits or saps quartile) or use a custom cohort membership (i.e. result of clustering). \
        Type: String. Default: 'careunit'. ")
    parser.add_argument("--cohort_filepath", type=str, help="This is the filename containing a numpy \
        array of length len(X), containing the cohort membership for each example in X. This file should be \
        saved in the folder 'cluster_membership'. Only applies to cohorts == 'custom'. Type: str.")
    parser.add_argument("--sample_weights", action="store_true", default=False, help="This is an indicator \
        flag to weight samples during training by their cohort's inverse frequency (i.e. smaller cohorts will be \
        more highly weighted during training).")
    parser.add_argument("--include_cohort_as_feature", action="store_true", default=False,
                        help="This is an indicator flag to include cohort membership as an additional feature in the matrix.")
    parser.add_argument("--epochs", type=int, default=100, # jw: was 30
                        help="Number of epochs to train for. Type: int. Default: 30.")
    parser.add_argument("--train_val_random_seed", type=int, default=0,
                        help="Random seed to use during train / val / split process. Type: int. Default: 0.")
    parser.add_argument("--repeats_allowed", action="store_true", default=False,
                        help="Indicator flag allowing training and evaluating of existing models. Without this flag, \
        if you run a configuration for which you've already saved models & results, it will be skipped.")
    parser.add_argument("--viz_time", action="store_true", default=False,
                        help="Indicator flag of whether we are in visualization time. With this flag, we will load in the already trained model \
        of the specified configuration, and evaluate it on the validation set. ")
    parser.add_argument("--test_time", action="store_true", default=False,
                        help="Indicator flag of whether we are in testing time. With this flag, we will load in the already trained model \
        of the specified configuration, and evaluate it on the test and validation set. ")
    parser.add_argument("--bootstrap", action="store_true", default=False,
                        help="Indicator flag of whether to evaluate on bootstrapped samples of the test set, or just the single \
        test set. Adding the flag will result in saving minimum, maximum and average AUCs on bo6otstrapped samples of the test dataset. ")
    parser.add_argument("--num_bootstrap_samples", type=int, default=100,
                        help="Number of bootstrapping samples to evaluate on for the test set. Type: int. Default: 100. ")
    # parser.add_argument("--gpu_num", type=str, default='5',
    #                     help="Limit GPU usage to specific GPUs. Specify multiple GPUs with the format '0,1,2'. Type: String. Default: '0'.") # this is handled elsewhere

    args = parser.parse_args()
    print(args)
    return args

def make_deterministic():
    '''make the running of the models deterministic'''
    from numpy.random import seed
    seed(1)
    from tensorflow import set_random_seed
    set_random_seed(2)
    torch.manual_seed(3)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cohorts2clusters(cohorts, all_tasks):
    '''turn an array of str to array of int'''
    res = np.zeros(cohorts.shape)
    for i, task in enumerate(all_tasks):
        res[cohorts==task] = i
    return res.astype(int)

def get_output_mtl(net, X, y, clusters, device='cuda'):
    '''clusters: assignment (cohorts) of shape (n,)'''
    loader = create_mtl_loader(X, y, clusters, shuffle=False)
    net.eval()
    o = []
    for i, (x, y_z) in enumerate(loader):
        x = x.to(device)
        assert y_z.shape[1] in [2, 3], "only y, z, optional sample weight"
        y, z = y_z[:, 0], y_z[:, 1]

        # from net(x) of shape (n_tasks, bs, 1) to (bs, 1), where x is (bs, T, d)
        o_ = torch.stack(net(x))[z.long(), torch.arange(len(x))]
        o.append(o_.detach().cpu().numpy())
        
    net.train()
    return np.vstack(o)

def pmt_importance(net, X_orig, y_orig, n_pmt=10, bs=None, device='cuda'):
    '''
    X: numpy array of size (n, T, d)
    y: numpy array of size (n,)
    net: pytorch model
    return feature importance array
    '''
    n, T, d = X_orig.shape
    if bs == None or bs >= n:
        bs = n
        X, y = X_orig, y_orig
        y_pred_orig = get_output(net, create_loader(X, y))

    fps = 0
    for _ in range(n_pmt):

        if bs != n: # small sample to boost speed
            indices = np.random.choice(len(X_orig), bs)
            X, y = X_orig[indices], y_orig[indices]
            y_pred_orig = get_output(net, create_loader(X, y))

        fp = [] # todo: currently doesn't consider correlation
        indices = np.random.choice(len(X), bs)
        X_ = X[indices]
        for i in tqdm.tqdm(range(d)):
            X_p = copy.deepcopy(X)
            X_p[:, :, i] = X_[:, :, i] # todo: temporal dimension assumes to be correlated
            y_pred_pmt = get_output(net, create_loader(X_p, y))
            fimp = y_pred_orig - y_pred_pmt # (bs, 1); no softmax b/c already after sigmoid
            fp.append(fimp.ravel()) # fp: length d list of (bs,)

        fp = np.vstack(fp).T # (bs, d)
        fps += np.abs(fp) # todo: play with this # fps += fp

    return np.abs(fps / n_pmt).mean(0) # (d,)

def save_output(model, model_name, X, y, cohorts, all_tasks, fname, FLAGS):
    if 'mtl' in model_name:
        y_pred = get_output_mtl(model, X, y,
                                cohorts2clusters(cohorts, all_tasks)).ravel()
    else:
        loader = create_loader(X, y)        
        y_pred = get_output(model, loader).ravel()

    # secondary mark for future change
    if FLAGS.runname is not None:
        fname += "_"  + FLAGS.runname
    joblib.dump((y_pred, y, cohorts),
                '{}/logs'.format(FLAGS.result_dir) + '/results/' +
                fname + FLAGS.result_suffix + '.pkl')
        
###### losses
def sample_weighted_bce_loss(yhat, y):
    '''
    yhat: a list of k element with (n, 1) shape in each element
    y: size of (n,) or (n, 2)
       in the first case it is regular bce loss
       second case the 0th dim is y, 1st dim is sample weights
    '''
    w = None # sample_weights
    if len(y.shape) > 1:
        y, w = y[:, 0], y[:, 1]

    if w is not None:
        criterion = nn.BCELoss(reduction='none')
        l = (criterion(yhat.view(-1), y) * w).mean()
    else:
        criterion = nn.BCELoss()
        l = criterion(yhat.view(-1), y)
    return l

def mtl_loss(yhat, y_z):
    '''
    yhat: a list of k element with (n, 1) shape in each element
    y_z: size (n, 2) with 0th dim being the output, 1st dim being the cluster
         or size (n, 3) with the 2nd dim being the sample weights
    '''
    w = None # sample_weights
    if y_z.shape[1] == 3:
        w = y_z[:, 2]

    y, z = y_z[:, 0], y_z[:, 1]
    # from (k, n, 1) to (n, 1)
    yhat = torch.stack(yhat)[z.long(), torch.arange(y.shape[0])].view(-1)

    if w is not None:
        criterion = nn.BCELoss(reduction='none')
        l = (criterion(yhat.view(-1), y) * w).mean()
    else:
        criterion = nn.BCELoss()
        l = criterion(yhat.view(-1), y)
    return l

###### loaders
def create_mtl_loader(X, y, clusters, samp_weights=None, batch_size=100, shuffle=False):
    '''
    clusters: cluster assignment of shape (n,)
    samp_weights: shape (n,) or None
    pmt: scales input by permutation importance or not

    dataset:
        from tensor dataset (X, y) to (X, (y, clusters))
    '''
    y_z = np.vstack((y, clusters)).T
    if samp_weights is not None:
        y_z = np.hstack((y_z, samp_weights.reshape(-1, 1)))
    loader = data.DataLoader(data.TensorDataset(
        torch.from_numpy(X).float(), torch.from_numpy(y_z).float()),
                             batch_size=batch_size, shuffle=shuffle)
    return loader

def create_loader(X, y, samp_weights=None, batch_size=100, shuffle=False):
    '''
    y: shape (n, )
    samp_weights: shape (n,) or None
    '''
    if samp_weights is not None:
        y = np.vstack((y, samp_weights)).T
    loader = data.DataLoader(data.TensorDataset(
        torch.from_numpy(X).float(), torch.from_numpy(y).float()),
                             batch_size=batch_size, shuffle=shuffle)
    return loader

###### models
def create_snapshot_model(model_args):
    """
    Create snapshot models with LSTM layer(s), shared dense layer(s), and sigmoided output.
    model_args: a dictionary with the following keys
        n_layers (int): Number of initial LSTM layers.
        units (int): Number of units in each LSTM layer.
        num_dense_shared_layers (int): Number of dense layers following LSTM layer(s).
        dense_shared_layer_size (int): Number of units in each dense layer.
        input_dim (int): Number of features in the input.
        output_dim (int): Number of outputs (1 for binary tasks).
        tasks (list): list of the tasks.
        global_model_dir (str): directory in which each epoch's performance is saved
        X_val: validation X (n, T, d)
        y_val: validation y (n,)
        cohorts_val: cluster assignment val (n,)
        FLAGS: arguments in this file
    Returns:
        PyTorch model
    """
    # similar to create_separate_model but with experts pretrained
    # 1. get model directory path with models at each epoch for a global model
    # 2. choose the model at epochs that gives best validation performance for each cohort
    # as starting point
    # 3. finetune the resulting model
    tasks = model_args['tasks']
    X_val, y_val, cohorts_val = model_args['X_val'], model_args['y_val'], model_args['cohorts_val']
    val_loader = create_loader(X_val, y_val, batch_size=100, shuffle=False)

    experts_auc = [(None, 0) for _ in range(len(tasks))] # init to (n model, 0 auc)
    for fn in glob.glob(model_args['global_model_dir'] + "/epoch*.m"):
        net = torch.load(fn)
        y_pred = get_output(net, val_loader).ravel()
        for i, task in enumerate(tasks):
            x_val_in_task = X_val[cohorts_val == task]
            y_val_in_task = y_val[cohorts_val == task]
            y_pred_in_task = y_pred[cohorts_val == task]
            try:
                auc = roc_auc_score(y_val_in_task, y_pred_in_task)
            except:
                auc = 0.1 # slightly larger than 0 but shouldn't be selected
            if auc > experts_auc[i][1]:
                experts_auc[i] = (net, auc)

    experts = nn.ModuleList([expert for expert, auc in experts_auc])
    # currently is inefficient by running all models for all tasks
    # I should be able to just run the required expert
    model = Separate_MIMIC_Model(experts)
    return model

def create_separate_model(model_args):
    """
    Create independent models with LSTM layer(s), shared dense layer(s), and sigmoided output.
    model_args: a dictionary with the following keys
        n_layers (int): Number of initial LSTM layers.
        units (int): Number of units in each LSTM layer.
        num_dense_shared_layers (int): Number of dense layers following LSTM layer(s).
        dense_shared_layer_size (int): Number of units in each dense layer.
        input_dim (int): Number of features in the input.
        output_dim (int): Number of outputs (1 for binary tasks).
        tasks (list): list of the tasks.
    Returns:
        PyTorch model
    """
    experts = nn.ModuleList()
    for i in range(len(model_args['tasks'])):
        experts.append(Global_MIMIC_Model(model_args['n_layers'],
                                          model_args['units'],
                                          model_args['num_dense_shared_layers'],
                                          model_args['dense_shared_layer_size'],
                                          model_args['input_dim'],
                                          model_args['output_dim']))

    # currently is inefficient by running all models for all tasks
    # I should be able to just run the required expert
    model = Separate_MIMIC_Model(experts)
    return model

def create_global_pytorch_model(model_args):
    """
    Create a global pytorch model with LSTM layer(s), shared dense layer(s), and sigmoided output.
    model_args: a dictionary with the following keys
        n_layers (int): Number of initial LSTM layers.
        units (int): Number of units in each LSTM layer.
        num_dense_shared_layers (int): Number of dense layers following LSTM layer(s).
        dense_shared_layer_size (int): Number of units in each dense layer.
        input_dim (int): Number of features in the input.
        output_dim (int): Number of outputs (1 for binary tasks).
    Returns:
        PyTorch model
    """
    model = Global_MIMIC_Model(model_args['n_layers'],
                               model_args['units'],
                               model_args['num_dense_shared_layers'],
                               model_args['dense_shared_layer_size'],
                               model_args['input_dim'],
                               model_args['output_dim'])

    return model

def create_moe_model(model_args):
    """
    Create a moe model with LSTM layer(s), shared dense layer(s), separate dense layer(s)
    and separate sigmoided outputs.
    model_args: a dictionary with the following keys
        input_dim (int): Number of features in the input.
        n_layers (int): Number of initial LSTM layers.
        units (int): Number of units in each LSTM layer.
        num_dense_shared_layers (int): Number of dense layers following LSTM layer(s).
        dense_shared_layer_size (int): Number of units in each dense layer.
        n_multi_layers (int): Number of task-specific dense layers.
        multi_layer_size (int): Number of units in each task-specific dense layer.
        output_dim (int): Number of outputs (1 for binary tasks).
        tasks (list): list of the tasks.
    Returns:
        final_model (Keras model): A compiled model with the provided architecture.
    """
    model = MoE_MIMIC_Model(model_args["input_dim"],
                            model_args["n_layers"],
                            model_args["units"],
                            model_args["num_dense_shared_layers"],
                            model_args["dense_shared_layer_size"],
                            model_args["n_multi_layers"],
                            model_args["multi_units"],
                            model_args["output_dim"],
                            model_args["FLAGS"].num_clusters)
    return model

def create_mtl_model(model_args):
    """
    Create a mtl model with LSTM layer(s), shared dense layer(s), separate dense layer(s)
    and separate sigmoided outputs.
    model_args: a dictionary with the following keys
        input_dim (int): Number of features in the input.
        n_layers (int): Number of initial LSTM layers.
        units (int): Number of units in each LSTM layer.
        num_dense_shared_layers (int): Number of dense layers following LSTM layer(s).
        dense_shared_layer_size (int): Number of units in each dense layer.
        n_multi_layers (int): Number of task-specific dense layers.
        multi_layer_size (int): Number of units in each task-specific dense layer.
        output_dim (int): Number of outputs (1 for binary tasks).
        tasks (list): list of the tasks.
    Returns:
        final PyTorch model
    """
    model = MTL_MIMIC_Model(model_args["input_dim"],
                            model_args["n_layers"],
                            model_args["units"],
                            model_args["num_dense_shared_layers"],
                            model_args["dense_shared_layer_size"],
                            model_args["n_multi_layers"],
                            model_args["multi_units"],
                            model_args["output_dim"],
                            model_args["tasks"])
    return model

def create_mtl_pt_model(model_args):
    """
    Create a mtl model with LSTM layer(s), shared dense layer(s), separate dense layer(s)
    and separate sigmoided outputs.
    model_args: a dictionary with the following keys
        input_dim (int): Number of features in the input.
        n_layers (int): Number of initial LSTM layers.
        units (int): Number of units in each LSTM layer.
        num_dense_shared_layers (int): Number of dense layers following LSTM layer(s).
        dense_shared_layer_size (int): Number of units in each dense layer.
        n_multi_layers (int): Number of task-specific dense layers.
        multi_layer_size (int): Number of units in each task-specific dense layer.
        output_dim (int): Number of outputs (1 for binary tasks).
        tasks (list): list of the tasks.
        global_model_fn (str): global best model fn
    Returns:
        final PyTorch model
    """
    global_model = torch.load(model_args['global_model_fn'])
    model = MTL_MIMIC_Model(model_args["input_dim"],
                            model_args["n_layers"],
                            model_args["units"],
                            model_args["num_dense_shared_layers"],
                            model_args["dense_shared_layer_size"],
                            model_args["n_multi_layers"],
                            model_args["multi_units"],
                            model_args["output_dim"],
                            model_args["tasks"],
                            lstm = global_model.lstm,
                            shared = global_model.rest[:-2],
    )
    return model

###### run
def get_model_fname_parts(model_name, FLAGS):
    '''
    affect how the data are saved in ['results', 'checkpoints', 'models']
    '''
    # secondary mark change later
    if FLAGS.runname is not None:
        return [FLAGS.runname]
    
    model_fname_parts = [model_name, FLAGS.num_lstm_layers,
                         str(FLAGS.lstm_layer_size), 'units',
                         str(FLAGS.num_dense_shared_layers), 'dense_shared',
                         str(FLAGS.dense_shared_layer_size), 'dense_units', 'mortality']
    model_fname_parts = [str(p) for p in model_fname_parts]
    if FLAGS.sample_weights:
        model_fname_parts.append("sw")
    if FLAGS.pmt:
        model_fname_parts.append("pmt")
    return model_fname_parts

def evaluation(model, model_name, X, y, cohorts, all_tasks, FLAGS):
    '''
    model: pytorch model
    X: (n, d) numpy array
    y: (n,) numpy array
    cohorts: (n,) numpy array
    all_tasks (Numpy array/list): List of tasks
    '''
    cohort_aucs = []

    loader = create_loader(X, y)
    if 'mtl' in model_name:
        y_pred = get_output_mtl(model, X, y,
                                cohorts2clusters(cohorts, all_tasks)).ravel()
    else:
        y_pred = get_output(model, loader).ravel()

    # all bootstrapped AUCs
    for task in all_tasks:
        if FLAGS.bootstrap:
            all_aucs = bootstrap_predict(X, y, cohorts, task, y_pred,
                                         return_everything=True,
                                         test=True,
                                         num_bootstrap_samples=FLAGS.num_bootstrap_samples)
            cohort_aucs.append(np.array(all_aucs))
            min_auc, max_auc, avg_auc = np.min(all_aucs), np.max(all_aucs), np.mean(all_aucs)
            print('{} Model AUC on {}: [min: {}, max: {}, avg: {}]'.format(model_name,
                                                                           task, min_auc,
                                                                           max_auc, avg_auc))
        else:
            y_pred_in_cohort = y_pred[cohorts == task]
            y_true_in_cohort = y[cohorts == task]
            auc = roc_auc_score(y_true_in_cohort, y_pred_in_cohort)
            cohort_aucs.append(auc)
            print('{} Model AUC on {}: {}'.format(model_name, task, cohort_aucs[-1]))

    if FLAGS.bootstrap:
        # Macro AUC
        cohort_aucs = np.array(cohort_aucs)
        cohort_aucs = np.concatenate(
            (cohort_aucs, np.expand_dims(np.mean(cohort_aucs, axis=0), 0)))

        all_aucs = cohort_aucs[-1]
        min_auc, max_auc, avg_auc = np.min(all_aucs), np.max(all_aucs), np.mean(all_aucs)
        print('{} Model AUC Macro: [min: {}, max: {}, avg: {}]'.format(model_name,
                                                                       min_auc, max_auc, avg_auc))

        # Micro AUC
        all_micro_aucs = bootstrap_predict(X, y, cohorts, 'all', y_pred,
                                           return_everything=True, test=True,
                                           num_bootstrap_samples=FLAGS.num_bootstrap_samples)
        cohort_aucs = np.concatenate(
            (cohort_aucs, np.array([all_micro_aucs])))

        all_aucs = cohort_aucs[-1]
        min_auc, max_auc, avg_auc = np.min(all_aucs), np.max(all_aucs), np.mean(all_aucs)
        print('{} Model AUC Micro: [min: {}, max: {}, avg: {}]'.format(model_name,
                                                                       min_auc, max_auc, avg_auc))
    else:
        # Macro AUC
        macro_auc = np.mean(cohort_aucs)
        cohort_aucs.append(macro_auc)
        print('{} Model AUC Macro: {}'.format(model_name, cohort_aucs[-1]))

        # Micro AUC
        micro_auc = roc_auc_score(y, y_pred)
        cohort_aucs.append(micro_auc)
        print('{} Model AUC Micro: {}'.format(model_name, cohort_aucs[-1]))

    return cohort_aucs

def save_cohort_aucs(cohort_aucs, model_name, fname, FLAGS):
    # secondary mark for future change
    if FLAGS.runname is not None:
        suffix = FLAGS.runname + "_"
    else:
        suffix = model_name + "_"
        if FLAGS.sample_weights:
            suffix += "sw_"
        if FLAGS.pmt:
            suffix += "pmt_"
    suffix += 'single' if not FLAGS.bootstrap else 'all'
    auc_fname = '{}_{}'.format(fname, suffix)
    np.save('{}/logs'.format(FLAGS.result_dir) + '/results/' +
            auc_fname + FLAGS.result_suffix, cohort_aucs)

def run_pytorch_model(model_name, create_model, X_train, y_train, cohorts_train,
                      X_val, y_val, cohorts_val,
                      X_test, y_test, cohorts_test,
                      all_tasks, FLAGS, samp_weights):
    """
    Train and evaluate pytorch models.
    Results are saved in result_dir/logs/results:
        - The numpy file ending in '_keys' contains the parameters for the model,
          and the numpy file ending in '_results' contains the validation AUCs for that
          configuration.
        - If you run multiple configurations for the same experiment name,
          those parameters and results will append to the same files.
        - At test time, results are saved into the file beginning 'test_auc_on_{model_name}'.
          The format of results will depend on whether you use bootstrapping or not. With bootstrapping,
          minimum, maximum and average AUCs are saved. Without, just the single AUC on the actual
          val / test dataset is saved.
    Args:
        model_name: model name when saved
        create_model: a function for creating the specific model
        X_train (Numpy array): The X matrix w training examples.
        y_train (Numpy array): The y matrix w training examples.
        cohorts_train (Numpy array): List of cohort membership for each validation example.
        X_val (Numpy array): The X matrix w validation examples.
        y_val (Numpy array): The y matrix w validation examples.
        cohorts_val (Numpy array): List of cohort membership for each validation example.
        X_test (Numpy array): The X matrix w testing examples.
        y_test (Numpy array): The y matrix w testing examples.
        cohorts_test (Numpy array): List of cohort membership for each testing example.
        all_tasks (Numpy array/list): List of tasks.
        FLAGS (dictionary): all the arguments.
        samp_weights: sample weights
    """
    model_fname_parts = get_model_fname_parts(model_name, FLAGS)
    if FLAGS.viz_time:
        model_path = '{}/logs'.format(FLAGS.result_dir) + \
            '/models/' + "_".join(model_fname_parts) + \
            FLAGS.result_suffix + '.m' # secondary mark for future change
        model = torch.load(model_path)

        # save output
        save_output(model, model_name, X_val, y_val, cohorts_val, all_tasks, "output", FLAGS)
        return
        
    if FLAGS.test_time:
        model_path = '{}/logs'.format(FLAGS.result_dir) + \
            '/models/' + "_".join(model_fname_parts) + \
            FLAGS.result_suffix + '.m' # secondary mark for future change
        model = torch.load(model_path)

        # print('testing on validation set')
        # cohort_aucs = evaluation(model, model_name, X_val, y_val, cohorts_val, all_tasks, FLAGS)
        # save_cohort_aucs(cohort_aucs, model_name, 'val_auc_on', FLAGS)

        # test
        print('testing on test set')
        cohort_aucs = evaluation(model, model_name, X_test, y_test, cohorts_test, all_tasks, FLAGS)
        save_cohort_aucs(cohort_aucs, model_name, 'test_auc_on', FLAGS)
        return

    batch_size = 100
    if 'mtl' in model_name:
        criterion = mtl_loss
        train_loader = create_mtl_loader(X_train, y_train, cohorts2clusters(cohorts_train, all_tasks),
                                         samp_weights=samp_weights,
                                         batch_size=batch_size, shuffle=True)
        # no samp_weights for val; samp_weights is only for train
        val_loader = create_mtl_loader(X_val, y_val, cohorts2clusters(cohorts_val, all_tasks),
                                       batch_size=batch_size, shuffle=False)
    else:
        criterion = sample_weighted_bce_loss
        train_loader = create_loader(X_train, y_train, samp_weights=samp_weights,
                                     batch_size=batch_size, shuffle=True)
        # no samp_weights for val; samp_weights is only for train
        val_loader = create_loader(X_val, y_val, batch_size=batch_size, shuffle=False)

    # secondary mark
    if FLAGS.global_model_fn is None:
        global_model_dir = '{}/logs'.format(FLAGS.result_dir) + \
                           '/checkpoints/global_pytorch_' + \
                           "_".join(model_fname_parts[1:]) + \
                           FLAGS.result_suffix
        global_model_fn = '{}/logs'.format(FLAGS.result_dir) + \
                         '/models/global_pytorch_' +\
                         "_".join(model_fname_parts[1:]) + \
                         FLAGS.result_suffix + ".m",
    else:
        global_model_dir = '{}/logs'.format(FLAGS.result_dir) + \
                           '/checkpoints/' + \
                           FLAGS.global_model_fn[:-2] # drop ".m"
        global_model_fn = '{}/logs'.format(FLAGS.result_dir) + \
                         '/models/' +\
                         FLAGS.global_model_fn
        
    model_args = {
        'n_layers': FLAGS.num_lstm_layers,
        'units': FLAGS.lstm_layer_size,
        'num_dense_shared_layers': FLAGS.num_dense_shared_layers,
        'dense_shared_layer_size': FLAGS.dense_shared_layer_size,
        'input_dim': X_train.shape[2], # (bs, T, d)
        'output_dim': 1,
        'n_multi_layers': FLAGS.num_multi_layers, # mtl layers
        'multi_units': FLAGS.multi_layer_size,
        'tasks': all_tasks,
        'global_model_dir': global_model_dir,
        'global_model_fn': global_model_fn,
        'X_val': X_val,
        'y_val': y_val,
        'cohorts_val': cohorts_val,
        'FLAGS': FLAGS,
    }

    model = create_model(model_args)
    model = model.cuda()

    # secondary mark for change later
    model_dir = '{}/logs'.format(FLAGS.result_dir) + \
        '/checkpoints/' + "_".join(model_fname_parts) + FLAGS.result_suffix
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.wd)
    get_c = partial(get_criterion, criterion=criterion)
    model, train_log = train(model, train_loader, criterion, optimizer, FLAGS.epochs,
                             savename = model_dir,
                             val_loader = val_loader,
                             es_named_criterion = ('loss', get_c, True),
                             verbose=True)

    joblib.dump(train_log, '{}/log'.format(model_dir))
    # secondary mark for change
    torch.save(model, '{}/logs'.format(FLAGS.result_dir) + '/models/' +
               "_".join(model_fname_parts) + FLAGS.result_suffix + '.m')

    ############### evaluation ###########
    # validation
    print('testing on validation set')
    cohort_aucs = evaluation(model, model_name, X_val, y_val, cohorts_val, all_tasks, FLAGS)
    save_cohort_aucs(cohort_aucs, model_name, 'val_auc_on', FLAGS)

    # test
    print('testing on test set')
    cohort_aucs = evaluation(model, model_name, X_test, y_test, cohorts_test, all_tasks, FLAGS)
    save_cohort_aucs(cohort_aucs, model_name, 'test_auc_on', FLAGS)

    print('Saved {} results.'.format(model_name))

def main():
    '''main function'''
    FLAGS = get_args()
    if not FLAGS.random_run:
        make_deterministic()

   # Make folders for the results & models
    for folder in ['results', 'models', 'checkpoints']:
        if not os.path.exists(os.path.join('{}/logs'.format(FLAGS.result_dir), folder)):
            os.makedirs(os.path.join('{}/logs'.format(FLAGS.result_dir), folder))

    # Load Data
    X, Y, cohort_col = load_data(FLAGS.dataname, FLAGS)

    # Include cohort membership as an additional feature
    if FLAGS.include_cohort_as_feature:
        cohort_col_onehot = pd.get_dummies(cohort_col).as_matrix()
        cohort_col_onehot = np.expand_dims(cohort_col_onehot, axis=1)
        cohort_col_onehot = np.tile(cohort_col_onehot, (1, 24, 1))
        X = np.concatenate((X, cohort_col_onehot), axis=-1)

    # Train, val, test split
    if FLAGS.viz_time:
        X_train, X_val, X_test, \
            y_train, y_val, y_test, \
            cohorts_train, cohorts_val, cohorts_test = X, X, X, \
                Y, Y, Y, cohort_col, cohort_col, cohort_col
    else:
        X_train, X_val, X_test, \
            y_train, y_val, y_test, \
            cohorts_train, cohorts_val, cohorts_test = stratified_split(
                X, Y, cohort_col, train_val_random_seed=FLAGS.train_val_random_seed)

    # Sample Weights
    task_weights = dict()
    all_tasks = sorted(np.unique(cohorts_train))

    for cohort in all_tasks:
        num_in_cohort = len(np.where(cohorts_train == cohort)[0])
        print("Number of people in cohort " +
              str(cohort) + ": " + str(num_in_cohort))
        task_weights[cohort] = len(X_train)*1.0/num_in_cohort

    if FLAGS.sample_weights:
        samp_weights = np.array([task_weights[cohort]
                                 for cohort in cohorts_train])
    else:
        samp_weights = None

    # Permutation importance
    if FLAGS.pmt:
        # secondary mark for change
        if FLAGS.global_model_fn is not None:
            global_model_fn = '{}/logs'.format(FLAGS.result_dir) + \
                '/models/' + FLAGS.global_model_fn
        else:
            # the last is pmt thus dropping
            model_fname_parts = get_model_fname_parts("dummy", FLAGS)[1:-1] 
            global_model_fn = '{}/logs'.format(FLAGS.result_dir) + \
                '/models/global_pytorch_' + "_".join(model_fname_parts) +\
                FLAGS.result_suffix + ".m"
        net = torch.load(global_model_fn)

        # if True: # this is used to investigate how stable bs is for pmt_importance
        #     for bs in [100, X_train.shape[0], 500, 2000, 5000, 10000]:
        #         feature_importance_fn = 'feature_importance{}.pkl'.format(bs)
        #         feature_importance = pmt_importance(net, X_train, y_train, bs=bs)
        #         joblib.dump(feature_importance, feature_importance_fn)

        feature_importance_fn = 'feature_importance1000.pkl'
        if os.path.exists(feature_importance_fn):
            feature_importance = joblib.load(feature_importance_fn)
        else:
            feature_importance = pmt_importance(net, X_train, y_train, bs=1000)
            joblib.dump(feature_importance, feature_importance_fn)

        feature_importance = feature_importance / feature_importance.max()
        X_train = X_train * feature_importance
        X_val = X_val * feature_importance
        X_test = X_test * feature_importance
        X = X * feature_importance

    # Run model
    run_model_args = [X_train, y_train, cohorts_train,
                      X_val, y_val, cohorts_val,
                      X_test, y_test, cohorts_test,
                      all_tasks, FLAGS, samp_weights]

    if FLAGS.model_type == 'SEPARATE':
        run_pytorch_model('separate_mtl', create_separate_model, *run_model_args)
    elif FLAGS.model_type == 'GLOBAL':
        run_pytorch_model('global_pytorch', create_global_pytorch_model, *run_model_args)
    elif FLAGS.model_type == 'MULTITASK':
        run_pytorch_model('mtl_pytorch', create_mtl_model, *run_model_args)
    elif FLAGS.model_type == 'MOE':
        run_pytorch_model('moe', create_moe_model, *run_model_args)
    elif FLAGS.model_type == 'SNAPSHOT': # pretrained version of separate
        run_pytorch_model('snapshot_mtl', create_snapshot_model, *run_model_args)
    elif FLAGS.model_type == 'MTL_PT': # pretrained MTL from global - specific layers
        run_pytorch_model('mtl_pt', create_mtl_pt_model, *run_model_args)

if __name__ == "__main__":
    main()

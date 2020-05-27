import os, re, copy
# os.environ["CUDA_VISIBLE_DEVICES"]="5" # jw: handles externally
from functools import partial
import torch
from torch import nn
import torch.utils.data as data

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import torch
torch.manual_seed(3)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import numpy as np
from numpy.random import permutation
import argparse
import glob
from keras.models import Model
from keras.layers import Input, LSTM, RepeatVector
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import EarlyStopping
from run_mortality_prediction import stratified_split
from sklearn.mixture import GaussianMixture
from generate_clusters import create_seq_ae
from sklearn.externals import joblib
from utils import train, get_criterion, get_output, get_y, get_x, random_split_dataset, load_data
from moe import create_loader, pmt_importance
from models import Global_MIMIC_Cluster_Model, Seq_AE_Model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str, default='mimic',
                        choices=['mimic', 'eicu'],
                        help="indicating which data to run. Type: String.")
    parser.add_argument("--result_dir", type=str, default='result/',
                        help="where result are saved. Type: String.")
    parser.add_argument("--eicu_cohort", type=str, default='ARF4',
                        choices=['ARF4', 'ARF12', 'Shock4', 'Shock12', 'mortality'],
                        help="the cohort for eicu")    
    parser.add_argument("--runname", type=str, default=None, help="setting name (default None)")
    parser.add_argument("--result_suffix", type=str, default='',
                        help="this will add to the end of every saved files")    
    parser.add_argument("--global_model_fn",
                        type=str, default=None, help="name of the global model to load")        
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate for Adam")
    parser.add_argument("--wd", type=float, default=0, help="weight decay Adam")    
    parser.add_argument("--model_type", type=str, default='AE',
                        choices=['AE', 'INPUT', 'GLOBAL', 'VAL_CURVE'],
                        help="indicating \
        which type of model to run. Type: String.")
    parser.add_argument("--pmt", action="store_true", default=False)
    parser.add_argument("--not_pt", action="store_true", default=False,
                        help='not use pretrained global model when training validation curve')
    parser.add_argument("--latent_dim", type=int, default=100, #50, \
        help='The embedding size, or latent dimension of the autoencoder. Type: int. Default: 50.')
    parser.add_argument("--ae_epochs", type=int, default=100, \
        help='Number of epochs to train autoencoder. Type: int. Default: 100.')
    
    # parser.add_argument("--ae_learning_rate", type=float, default=0.001, #0.0001, \
    #     help='Learning rate for autoencoder. Type: float. Default: 0.0001.') # jw: use lr instead
    
    parser.add_argument("--num_clusters", type=int, default=3, \
        help='Number of clusters for GMM. Type: int. Default: 3.')
    parser.add_argument("--gmm_tol", type=float, default=0.0001,
        help='The convergence threshold for the GMM. Type: float. Default: 0.0001.')
    parser.add_argument("--data_hours", type=int, default=24, \
        help='The number of hours of data to use. \
        Type: int. Default: 24.')
    parser.add_argument("--gap_time", type=int, default=12, help="Gap between data and when predictions are made. Type: int. Default: 12.")
    parser.add_argument("--save_to_fname", type=str, default='test_clusters.npy', \
        help="Filename to save cluster memberships to. Type: String. Default: 'test_clusters.npy'")
    parser.add_argument("--train_val_random_seed", type=int, default=0, \
        help="Random seed to use during train / val / split process. Type: int. Default: 0.")
    args = parser.parse_args()
    print(args)
    return args

def val_curve_kmeans(curves, k=2, niters=10):
    '''
    curves: (n, epochs)
    output assignment, cluster_centers (best_epochs)
    '''
    n, epochs = curves.shape
    if k <= 0: # each assign a cluster
        assignment = np.arange(n)
        best_epochs = np.argmax(curves, 1)
        return assignment, best_epochs
    else:
        assignment = np.random.choice(k, n)
        # init at different percentile
        if k >= curves.shape[1]:
            best_epochs = np.arange(k)
        else:
            best_epochs = np.percentile([np.argmax(curves[i]) for i in range(n)],
                                        np.linspace(0, 100, k)).astype(np.int)

    print('init best epochs', best_epochs)
    for _ in range(niters):
        for i in range(n):
            # break tie in performance by random permutation
            best_cluster = permutation([(j, curves[i][best_epochs[j]])\
                                        for j in range(k)])
            best_cluster = max(best_cluster, key=lambda x: x[1])[0]
            assignment[i] = best_cluster
            
        best_epochs = [np.argmax(curves[(assignment==i).nonzero()].mean(0))\
                       for i in range(k)]

    return assignment, best_epochs

def get_suffix_fname_model(FLAGS):
    '''common model suffix'''
    # secondary mark change later
    if FLAGS.runname is not None:
        return FLAGS.runname # + FLAGS.result_suffix # don't need this b/c it is for model
    
    fname_parts = [FLAGS.latent_dim, FLAGS.data_hours]
    if FLAGS.pmt:
        fname_parts.append("pmt")
    return "_".join(map(str, fname_parts))

def get_suffix_fname_cluster(FLAGS):
    '''common cluster model suffix'''
    # secondary mark change later
    if FLAGS.runname is not None:
        return FLAGS.runname # + FLAGS.result_suffix # don't need this b/c it is for model
    
    fname_parts = [FLAGS.model_type, FLAGS.num_clusters, FLAGS.data_hours]
    if FLAGS.pmt:
        fname_parts.append("pmt")
    if not FLAGS.not_pt:
        fname_parts.append("pt")
    return "_".join(map(str, fname_parts))

def train_seq_ae(X_train, X_val, FLAGS):
    """
    Train a sequence to sequence autoencoder.
    Args: 
        X_train (Numpy array): training data. (shape = n_samples x n_timesteps x n_features)
        X_val (Numpy array): validation data.
        FLAGS (dictionary): all provided arguments.
    Returns: 
        encoder (Keras model): trained model to encode to latent space.
        sequence autoencoer (Keras model): trained autoencoder.
    """
    encoder, sequence_autoencoder = create_seq_ae(X_train, X_val, FLAGS.latent_dim, FLAGS.lr)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    fname_suffix = get_suffix_fname_model(FLAGS)
    encoder_fn = '{}/clustering_models/encoder_{}'.format(FLAGS.result_dir,
                                                          fname_suffix)
    seq_ae_fn = '{}/clustering_models/seq_ae_{}'.format(FLAGS.result_dir,
                                                        fname_suffix)

    print(encoder_fn, seq_ae_fn)
    if os.path.exists(encoder_fn) and os.path.exists(seq_ae_fn):
        return load_model(encoder_fn), load_model(seq_ae_fn)
    
    # fit the model
    print("Fitting Sequence Autoencoder ... ")
    sequence_autoencoder.fit(X_train, X_train,
                    epochs=FLAGS.ae_epochs,
                    batch_size=128,
                    shuffle=True,
                    callbacks=[early_stopping],
                    validation_data=(X_val, X_val))

    encoder.save(encoder_fn)
    sequence_autoencoder.save(seq_ae_fn)
    return encoder, sequence_autoencoder

def create_seq_ae_pytorch(input_dim, latent_dim):
    '''
    Build sequence autoencoder. 
    Args: 
        X_train (Numpy array): training data. (shape = n_samples x n_timesteps x n_features)
        X_val (Numpy array): validation data.
        latent_dim (int): hidden representation dimension.
    Returns: 
        sequence_autoencoder (pytorch model): autoencoder model.
    '''
    return Seq_AE_Model(input_dim, latent_dim)

def train_seq_ae_pytorch(X_train, X_val, FLAGS):
    """
    Train a sequence to sequence autoencoder.
    Args: 
        X_train (Numpy array): training data. (shape = n_samples x n_timesteps x n_features)
        X_val (Numpy array): validation data.
        FLAGS (dictionary): all provided arguments.
    Returns: 
        encoder (pytorch model): trained model to encode to latent space.
    """
    sequence_autoencoder = create_seq_ae_pytorch(X_train.shape[2], FLAGS.latent_dim)
    
    fname_suffix = get_suffix_fname_model(FLAGS)
    model_dir = '{}/clustering_models/seq_ae_pytorch_{}'.format(FLAGS.result_dir,
                                                                fname_suffix)
    seq_ae_fn = model_dir + '.m'

    print(seq_ae_fn)
    if os.path.exists(seq_ae_fn):
        return torch.load(seq_ae_fn)
    
    # fit the model
    print("Fitting Sequence Autoencoder ... ")    
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.wd)
    criterion = nn.MSELoss()
    get_c = partial(get_criterion, criterion=criterion)
    model, train_log = train(model,
                             create_loader(X_train, X_train),
                             criterion, optimizer,
                             FLAGS.ae_epochs,
                             savename = model_dir
                             val_loader = create_loader(X_val, X_val),
                             es_named_criterion = ('loss', get_c, True),
                             verbose=True)

    joblib.dump(train_log, '{}/log'.format(model_dir))
    torch.save(model, seq_ae_fn)
    return sequence_autoencoder.encoder_forward

def train_assignment(FLAGS, k, assignment, loader, savename_suffix,
                     n_epochs=50, net=None,
                     criterion=nn.CrossEntropyLoss()):
    ''' 
    assignment: (n,) cluster assignments array
    mapp from input to assignment
    '''
    fname_suffix = get_suffix_fname_cluster(FLAGS)
    gate_fn = '{}/clustering_models/gate_{}{}.m'.format(FLAGS.result_dir,
                                                        fname_suffix,
                                                        savename_suffix)

    print(gate_fn)
    if os.path.exists(gate_fn):
        return torch.load(gate_fn)
    
    X = get_x(loader) # (n, T, d)
    dataset = data.TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(assignment).long())

    # create dataset
    dataset_train, dataset_val = random_split_dataset(dataset, [0.8, 0.2])
    train_loader = data.DataLoader(dataset_train,
                                   batch_size=100, shuffle=True)
    val_loader = data.DataLoader(dataset_val, batch_size=100,
                                 shuffle=False)

    if net == None:
        net = Global_MIMIC_Cluster_Model(X.shape[2], k)
        net = net.cuda()

    opt = torch.optim.Adam(net.parameters(),
                           lr=FLAGS.lr,
                           weight_decay=FLAGS.wd)
    get_c = partial(get_criterion, criterion=criterion)
    net, train_log = train(net, train_loader, criterion, opt, n_epochs, verbose=True,
                           val_loader=val_loader,
                           es_named_criterion = ('loss', get_c, True))

    torch.save(net, gate_fn)
    joblib.dump(train_log, gate_fn[:-2] + '_log')
    return net

def gmm_fit_and_predict(embedded_train, embedded_all, FLAGS, savename_suffix):
    '''
    embedded_train: training data for gmm (n_tr, d) array
    embedded_all: all data for gmm (n, d) array
    return
        numpy array embeddings
    '''
    fname_suffix = get_suffix_fname_cluster(FLAGS)
    gmm_fn_name = '{}/clustering_models/gmm_{}{}'.format(FLAGS.result_dir,
                                                         fname_suffix, savename_suffix)
    if os.path.exists(gmm_fn_name):
        gm = joblib.load(gmm_fn_name)
    else:
        # Train GMM
        print("Fitting GMM ...")
        gm = GaussianMixture(n_components=FLAGS.num_clusters, tol=FLAGS.gmm_tol,
                             n_init=30, # jw: reported in paper
                             init_params='kmeans',
                             verbose=True)
        gm.fit(embedded_train)
        joblib.dump(gm, gmm_fn_name)

    # Get cluster membership
    cluster_preds = gm.predict(embedded_all)
    return cluster_preds
    
''' 'AE', 'GLOBAL', 'INPUT', 'VAL_CURVE' '''
####### training clusters #######
def train_ae(cluster_args):
    '''
    returns 
       cluster_predictions (type:int): shape (n,)
    '''
    X = cluster_args['X']
    X_train = cluster_args['X_train']
    X_val = cluster_args['X_val']
    FLAGS = cluster_args['FLAGS']
    
    # Train autoencoder
    encoder, sequence_autoencoder = train_seq_ae(X_train, X_val, FLAGS)

    # Get Embeddings
    embedded_train = encoder.predict(X_train)
    embedded_all = encoder.predict(X)
    return gmm_fit_and_predict(embedded_train, embedded_all, FLAGS,
                               savename_suffix="_ae")

def train_ae_pytorch(cluster_args):
    '''
    pytorch version of train ae
    returns 
       cluster_predictions (type:int): shape (n,)
    '''
    X = cluster_args['X']
    X_train = cluster_args['X_train']
    X_val = cluster_args['X_val']
    FLAGS = cluster_args['FLAGS']
    
    # Train autoencoder
    encoder = train_seq_ae_pytorch(X_train, X_val, FLAGS)

    # Get Embeddings
    embedded_train = get_output(encoder, create_loader(X_train, y_train))
    embedded_all = get_output(encoder, create_loader(X, y))
    return gmm_fit_and_predict(embedded_train, embedded_all, FLAGS,
                               savename_suffix="_ae")

def train_input(cluster_args):
    '''cluster on the input space'''
    X = cluster_args['X']
    X_train = cluster_args['X_train']
    X_val = cluster_args['X_val']
    FLAGS = cluster_args['FLAGS']

    # Get Embeddings: flatten T dimension
    embedded_train = X_train.reshape(X_train.shape[0], -1)
    embedded_all = X.reshape(X.shape[0], -1)
    return gmm_fit_and_predict(embedded_train, embedded_all, FLAGS, savename_suffix="_input")

def train_global(cluster_args):
    '''
    1. apply GMM or kmeans on the repr after lstm
    returns 
       cluster_predictions (type:int): shape (n,)
    '''
    X = cluster_args['X']
    y = cluster_args['y']
    X_train = cluster_args['X_train']
    y_train = cluster_args['y_train']
    global_model_fn = cluster_args['global_model_fn']
    FLAGS = cluster_args['FLAGS']
    
    global_model = torch.load(global_model_fn)
    global_model.cuda()

    def extract_embedding(x):
        '''x assumes to be pytorch tensor'''
        o, (h, c) = global_model.lstm(x)
        embeddings = h[-1]
        return embeddings

    # Get Embeddings:    
    embedded_train = get_output(global_model, create_loader(X_train, y_train))
    embedded_all = get_output(global_model, create_loader(X, y))
    return gmm_fit_and_predict(embedded_train, embedded_all, FLAGS,
                               savename_suffix="_global")
    
def train_val_curve(cluster_args):
    # 1. learn validation curve: save as snapshot; reuse the code
    # 2. train from x to validation curve cluster
    # 3. apply this on Train, val and test to save 
    X = cluster_args['X']
    y = cluster_args['y']
    X_train = cluster_args['X_train']
    y_train = cluster_args['y_train']
    X_val = cluster_args['X_val']
    y_val = cluster_args['y_val']
    global_model_dir = cluster_args['global_model_dir']
    global_model_fn = cluster_args['global_model_fn']
    val_loader = cluster_args['val_loader']
    FLAGS = cluster_args['FLAGS']

    def sorted_by_epoch(l):
        return sorted(l, key=lambda s: int(re.search("epoch(.*)\.m", s).group(1)))

    curves = []
    y_val = get_y(val_loader)
    criterion = nn.BCELoss(reduction='none')
    for fn in sorted_by_epoch(glob.glob(global_model_dir + "/epoch*.m")):
        net = torch.load(fn)
        y_pred_val = get_output(net, val_loader).ravel() # (n_val,)
        # collect the loss
        curves.append(-criterion(
            torch.from_numpy(y_pred_val),
            torch.from_numpy(y_val)
        ).detach().cpu().numpy()) # (epochs, n_val)
    curves = np.array(curves).T # (n_val, epochs)
    # joblib.dump(curves, 'val_curves.pkl')
    
    # train a mapping from input to val_curve
    k = FLAGS.num_clusters
    assignment, experts_epochs = val_curve_kmeans(curves, k=k, niters=10)

    if FLAGS.not_pt:    
        net = None
    else:
        net = torch.load(global_model_fn)
        net.rest[-2] = nn.Linear(net.rest[-2].in_features, k).cuda()
        net.rest = net.rest[:-1] # drop sigmoid layer
    gate = train_assignment(FLAGS, k, assignment, val_loader, net=net, savename_suffix="_val_curve")

    # # for debug
    # cluster_preds_val = get_output(gate, create_loader(X_val, y_val)).argmax(1)
    # joblib.dump(cluster_preds_val, "val_assignments.pkl")
    
    # Get cluster membership: from (n, k) -> (n,)
    cluster_preds = get_output(gate, create_loader(X, y)).argmax(1)
    return cluster_preds

####### running ##################
def main():
    FLAGS = get_args()

    # Load Data
    X, Y, cohort_col = load_data(FLAGS.dataname, FLAGS)

    # Train, val, test split
    X_train, X_val, X_test, \
    y_train, y_val, y_test, \
    cohorts_train, cohorts_val, cohorts_test = stratified_split(X, Y, cohort_col, train_val_random_seed=FLAGS.train_val_random_seed)

    # mark for change
    if FLAGS.global_model_fn is not None:
        # drop ".m"
        global_model_dir = "{}/logs/checkpoints/{}".format(FLAGS.result_dir,
                                                           FLAGS.global_model_fn[:-2])
        global_model_fn = "{}/logs/models/{}".format(FLAGS.result_dir,
                                                     FLAGS.global_model_fn)
    else:
        global_model_dir = "dummy"
        global_model_fn = "dummy.m"
        
    if FLAGS.pmt:
        net = torch.load(global_model_fn)

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

    cluster_args = {
        'X': X,
        'y': Y,
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,        
        'val_loader': create_loader(X_val, y_val),
        'FLAGS': FLAGS,
        'global_model_fn': global_model_fn,
        'global_model_dir': global_model_dir,
    }

    cluster_model_dir = '{}/clustering_models/'.format(FLAGS.result_dir)
    if not os.path.exists(cluster_model_dir):
        os.makedirs(cluster_model_dir)
    
    if FLAGS.model_type == 'AE':
        cluster_preds = train_ae_pytorch(cluster_args)        
    if FLAGS.model_type == 'INPUT': # cluster on input
        cluster_preds = train_input(cluster_args)
    elif FLAGS.model_type == 'GLOBAL':
        cluster_preds = train_global(cluster_args)
    elif FLAGS.model_type == 'VAL_CURVE':
        cluster_preds = train_val_curve(cluster_args)

    if not os.path.exists('{}/cluster_membership/'.format(FLAGS.result_dir)):
        os.makedirs('{}/cluster_membership/'.format(FLAGS.result_dir))
    # secondary mark maybe change later
    if FLAGS.runname is not None:
        savename = FLAGS.runname + FLAGS.result_suffix + ".npy"
    else:
        model_name = FLAGS.model_type + ("_pmt" if FLAGS.pmt else "")
        savename = model_name + "_" + FLAGS.save_to_fname
    np.save('{}/cluster_membership/'.format(FLAGS.result_dir) + savename, cluster_preds)

if __name__ == "__main__":
    main()

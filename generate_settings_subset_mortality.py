'''
a setting is a list of argument like the following
[('--lr', 0.001), ('--wd', 0), '--pmt', ('--runname', runname)]
'''
import numpy as np
import os
from sklearn.externals import joblib
import argparse
from tune import run

gpus = [5, 6]
# result_dir_prefix = './' #'/data7/jiaxuan/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global_model_fn", required=True, type=str,
                        help="best global model file name in result_dir/logs/models/,\
                        example=10_global_exp.m")
    parser.add_argument("--result_dir_prefix", type=str, default="./",
                        help="where data is saved: e.g. '/data7/jiaxuan/'")
    parser.add_argument("--train_data_subset_path", required=True, type=str,
                        help='subset indices for the data; e.g. eICU_data/mortality/pct_{pct}_train_indices/0.pkl')
    parser.add_argument("--pct_val", default=1, type=float,
                        help='pct of validation data to use: useful for snapshot val_curve')    
    parser.add_argument("--eicu_cohort", type=str, default='mortality',
                        choices=['ARF4', 'ARF12', 'Shock4', 'Shock12', 'mortality'],
                        help="the cohort for eicu")
    parser.add_argument("--nc", default=1, type=int,
                        help="number of concurrent jobs, default 1")
    args = parser.parse_args()
    print(args)
    return args

def setting2dict(setting):
    '''
    INPUT: setting = [('--lr', 0.001), ('--wd', 0), '--pmt', ('--runname', runname)]
    RETURNS:
       {'--lr': 0.001, '--wd': 0, '--pmt': 'default', '--runname': runname}
    '''
    args = []
    for arg in setting:
        if type(arg) in [list, tuple]:
            assert len(arg) == 2, "only accept (k,v) pairs for list or tuple argument"
            args.append(arg)
        elif type(arg) == str:
            args.append((arg, 'default'))
        else:
            raise Exception('unrecognized type, args can only be (k,v) or str')
    return dict(args)

def create_joint_settings(FLAGS, n_settings=30):
    '''
    this applies to global and moe because they don't require the
    clustering function to be given
    RETURN: a list of settings
    setting: [('--lr', 0.001), ('--wd', 0), '--pmt', ('--runname', runname)]
    '''
    setting_dir = '{}/settings/'.format(FLAGS.eicu_cohort)
    if not os.path.exists(setting_dir):
        os.makedirs(setting_dir)

    settings = []
    fname = 'model_joint_settings.pkl'
    fname = os.path.join(setting_dir, fname)
    if os.path.exists(fname):
        print('settings already exists in {}, loading...'.format(fname))
        settings = joblib.load(fname)
        if n_settings <= len(settings):
            return settings[:n_settings]
        else:
            n_settings -= len(settings) # continue generating

    for _ in range(n_settings):
        setting = [
            ('--runname', str(len(settings))), # name of the saved model
            ('--lr', 10**np.random.uniform(-2,-4)),
            ('--wd', 10**np.random.uniform(-3,-10)),
            ('--num_clusters', np.random.choice([2, 3, 4, 5])), # only for MoE
            ('--num_lstm_layers', np.random.choice([1, 2, 3])),
            ('--lstm_layer_size', np.random.choice([16, 100, 300])),
            ('--num_dense_shared_layers', np.random.choice([0, 1, 2, 3])),
            ('--dense_shared_layer_size', np.random.choice([16, 100, 300])),
            ('--num_multi_layers', np.random.choice([0, 1, 2, 3])),
            ('--multi_layer_size', np.random.choice([16, 100, 300]))]
        settings.append(setting)

    print('saving {}'.format(fname))
    joblib.dump(settings, fname)
    return settings

def create_cluster_model_settings(FLAGS, n_settings=30):
    '''
    uses create_joint settings as base, assumes global model is given
    return model_settings, cluster_settings

    caution: user need to provide '--cohort_filepath' and '--global_model_fn'
    '''

    cluster_settings, model_settings = [], []
    settings = create_joint_settings(FLAGS, n_settings)

    fname = '{}/settings/cluster_model_settings.pkl'.format(FLAGS.eicu_cohort)
    if os.path.exists(fname):
        print('settings already exists in {}, loading...'.format(fname))
        cluster_settings, model_settings = joblib.load(fname)
        if n_settings <= len(cluster_settings):
            return cluster_settings[:n_settings], model_settings[:n_settings]
        else:
            n_settings -= len(cluster_settings) # continue generating
            settings = settings[-n_settings:]

    for i in range(n_settings):
        # create remaining cluster settings
        runname = str(len(cluster_settings))
        cluster_setting = [
            ('--runname', runname), # name of the saved model
            ('--lr', 10**np.random.uniform(-2,-4)),
            ('--wd', 10**np.random.uniform(-3,-10)),
            ('--num_clusters', np.random.choice([2, 3, 4, 5])),
            ('--latent_dim', np.random.choice([16, 100, 300, 500]))]
        cluster_settings.append(cluster_setting)

        # create remaining model settings: continue from create_joint_settings
        # note model_setting already have runname, so no need to add again
        model_setting = settings[i] # based on create_joint_settings
        model_setting.append(('--cohorts', 'custom'))
        # if np.random.choice(2) == 1:
        #     model_setting.append('--sample_weights')
        model_settings.append(model_setting)

    # save the settings
    print('saving {}'.format(fname))
    joblib.dump((cluster_settings, model_settings), fname)
    return cluster_settings, model_settings

### debugging settings
def experiment_debug_joint(FLAGS, expname='debug', test_time=False, viz_time=False,
                     debug=None):
    '''
    debug settings: see read me for manually tuned performance
    '''
    settings = create_joint_settings(FLAGS)

    if debug is not None:
        if type(debug) is list:
            settings = [settings[idx] for idx in debug]
        else:
            idx = debug
            settings = settings[idx:idx+1]

    settings = [[
        ('--lr', 0.001), ('--wd', 1e-4)
    ]]

    tasks = [[('--model_type', 'MULTITASK'),
              ('--result_dir', FLAGS.result_dir_prefix + 'debug'),
              '--include_cohort_as_feature',
              # '--test_time',
              # '--bootstrap',
              ('--epochs', 100),
              ('--global_model_fn', FLAGS.global_model_fn),
              ('--result_suffix', '_' + expname),
              ('--cohorts', 'careunit')] +
             setting for setting in settings]

    if test_time:
        tasks = [['--test_time', '--bootstrap'] + setting for setting in tasks]
    if viz_time:
        tasks = [['--viz_time'] + setting for setting in tasks]
    run('moe.py', tasks, gpus=gpus, n_concurrent_process=FLAGS.nc)

def experiment_debug_separate(FLAGS, expname='debug2', test_time=False,
                              debug=None, dataname='eicu'):
    '''
    global clustering followed by mtl
    '''
    if FLAGS.global_model_fn is None: return
    cluster_settings, model_settings = create_cluster_model_settings(FLAGS)

    if debug is not None:
        if type(debug) is list:
            cluster_settings = [cluster_settings[idx] for idx in debug]
            model_settings = [model_settings[idx] for idx in debug]
        else:
            idx = debug
            cluster_settings = cluster_settings[idx:idx+1]
            model_settings = model_settings[idx:idx+1]

    cluster_settings = [[('--model_type', 'AE'),
                         ('--result_dir', FLAGS.result_dir_prefix + 'debug'),
                         ('--dataname', dataname),
                         ('--global_model_fn', FLAGS.global_model_fn),
                         ('--result_suffix', '_' + expname)] +
                        setting for setting in cluster_settings]
    model_settings = [[('--model_type', 'MULTITASK'),
                       ('--result_dir', FLAGS.result_dir_prefix + 'debug'),
                       ('--dataname', dataname),
                       ('--result_suffix', '_' + expname),
                       ('--global_model_fn', FLAGS.global_model_fn),
                       ('--cohort_filepath', str(i) + '_' + expname + '.npy')] +
                      setting for i, setting in enumerate(model_settings)]

    # acknowledge the temporal dependence between the runs
    # first run cluster_settings, followed by model_settings
    # also make sure model_settings uses cluster settings' model
    run('cluster_moe.py', cluster_settings, gpus=gpus, n_concurrent_process=FLAGS.nc)
    if test_time:
        model_settings = [['--test_time', '--bootstrap'] + setting for setting in model_settings]
    run('moe.py', model_settings, gpus=gpus, n_concurrent_process=FLAGS.nc)

### real experiments
def experiment1(FLAGS, expname='moe_exp', test_time=False, dataname='eicu',
                debug=None):
    '''
    MoE experiment
    '''
    pct = FLAGS.train_data_subset_path.split('pct_')[1].split('_')[0]
    pct_num = os.path.basename(FLAGS.train_data_subset_path)[:-4] # remove '.pkl'
    expname = "pct{}_{}_{}".format(pct, pct_num, expname)

    settings = create_joint_settings(FLAGS)

    if debug is not None:
        if type(debug) is list:
            settings = [settings[idx] for idx in debug]
        else:
            idx = debug
            settings = settings[idx:idx+1]

    tasks = [[('--model_type', 'MOE'),
              ('--train_data_subset_path', FLAGS.train_data_subset_path),
              ('--result_dir', FLAGS.result_dir_prefix + FLAGS.eicu_cohort),
              ('--eicu_cohort', FLAGS.eicu_cohort),
              ('--dataname', dataname),
              ('--result_suffix', '_' + expname)] +
             setting for setting in settings]
    if test_time:
        tasks = [['--test_time', '--bootstrap'] + setting for setting in tasks]
    run('moe.py', tasks, gpus=gpus, n_concurrent_process=FLAGS.nc)

def experiment2(FLAGS, expname='global_exp', test_time=False, dataname='eicu',
                debug=None):
    '''
    Global model only experiment
    '''
    pct = FLAGS.train_data_subset_path.split('pct_')[1].split('_')[0]
    pct_num = os.path.basename(FLAGS.train_data_subset_path)[:-4] # remove '.pkl'
    expname = "pct{}_{}_{}".format(pct, pct_num, expname)

    settings = create_joint_settings(FLAGS)

    if debug is not None:
        if type(debug) is list:
            settings = [settings[idx] for idx in debug]
        else:
            idx = debug
            settings = settings[idx:idx+1]

    tasks = [[('--model_type', 'GLOBAL'),
              ('--train_data_subset_path', FLAGS.train_data_subset_path),
              ('--result_dir', FLAGS.result_dir_prefix + FLAGS.eicu_cohort),
              ('--eicu_cohort', FLAGS.eicu_cohort),
              ('--dataname', dataname),
              ('--result_suffix', '_' + expname)] +
             setting for setting in settings]
    if test_time:
        tasks = [['--test_time', '--bootstrap'] + setting for setting in tasks]
    run('moe.py', tasks, gpus=gpus, n_concurrent_process=FLAGS.nc)

### experiments that requires clustering
# mtl no prior
def experiment3(FLAGS, expname='mtl_od', test_time=False,
                debug=None, dataname='eicu'):
    '''
    mtl outcome dependent (global)
    '''
    pct = FLAGS.train_data_subset_path.split('pct_')[1].split('_')[0]
    pct_num = os.path.basename(FLAGS.train_data_subset_path)[:-4] # remove '.pkl'
    expname = "pct{}_{}_{}".format(pct, pct_num, expname)

    if FLAGS.global_model_fn is None: return
    cluster_settings, model_settings = create_cluster_model_settings(FLAGS)

    if debug is not None:
        if type(debug) is list:
            cluster_settings = [cluster_settings[idx] for idx in debug]
            model_settings = [model_settings[idx] for idx in debug]
        else:
            idx = debug
            cluster_settings = cluster_settings[idx:idx+1]
            model_settings = model_settings[idx:idx+1]

    cluster_settings = [[('--model_type', 'GLOBAL'),
                         ('--train_data_subset_path', FLAGS.train_data_subset_path),
                         "--cluster_add_result_suffix",
                         ('--result_dir', FLAGS.result_dir_prefix + FLAGS.eicu_cohort),
                         ('--eicu_cohort', FLAGS.eicu_cohort),
                         ('--dataname', dataname),
                         ('--global_model_fn', FLAGS.global_model_fn),
                         ('--result_suffix', '_' + expname)] +
                        setting for setting in cluster_settings]
    model_settings = [[('--model_type', 'MULTITASK'),
                       ('--train_data_subset_path', FLAGS.train_data_subset_path),
                       ('--result_dir', FLAGS.result_dir_prefix + FLAGS.eicu_cohort),
                       ('--eicu_cohort', FLAGS.eicu_cohort),
                       ('--dataname', dataname),
                       ('--result_suffix', '_' + expname),
                       ('--global_model_fn', FLAGS.global_model_fn),
                       ('--cohort_filepath', str(i) + '_' + expname + '.npy')] +
                      setting for i, setting in enumerate(model_settings)]

    # acknowledge the temporal dependence between the runs
    # first run cluster_settings, followed by model_settings
    # also make sure model_settings uses cluster settings' model
    run('cluster_moe.py', cluster_settings, gpus=gpus, n_concurrent_process=FLAGS.nc)
    if test_time:
        model_settings = [['--test_time', '--bootstrap'] + setting for setting in model_settings]
    run('moe.py', model_settings, gpus=gpus, n_concurrent_process=FLAGS.nc)

def experiment4(FLAGS, expname='mtl_val_curve', test_time=False,
                debug=None, dataname='eicu'):
    '''
    mtl val curve
    '''
    pct = FLAGS.train_data_subset_path.split('pct_')[1].split('_')[0]
    pct_num = os.path.basename(FLAGS.train_data_subset_path)[:-4] # remove '.pkl'
    expname = "pct{}_{}_{}".format(pct, pct_num, expname)
    if FLAGS.pct_val < 1:
        expname = "{}_val{}".format(expname, int(FLAGS.pct_val * 100))

    if FLAGS.global_model_fn is None: return
    cluster_settings, model_settings = create_cluster_model_settings(FLAGS)

    if debug is not None:
        if type(debug) is list:
            cluster_settings = [cluster_settings[idx] for idx in debug]
            model_settings = [model_settings[idx] for idx in debug]
        else:
            idx = debug
            cluster_settings = cluster_settings[idx:idx+1]
            model_settings = model_settings[idx:idx+1]

    cluster_settings = [[('--model_type', 'VAL_CURVE'),
                         ('--train_data_subset_path', FLAGS.train_data_subset_path),
                         ('--pct_val', FLAGS.pct_val),
                         "--cluster_add_result_suffix",
                         ('--result_dir', FLAGS.result_dir_prefix + FLAGS.eicu_cohort),
                         ('--eicu_cohort', FLAGS.eicu_cohort),
                         ('--dataname', dataname),
                         ('--global_model_fn', FLAGS.global_model_fn),
                         ('--result_suffix', '_' + expname)] +
                        setting for setting in cluster_settings]
    model_settings = [[('--model_type', 'MULTITASK'),
                       ('--train_data_subset_path', FLAGS.train_data_subset_path),
                       ('--result_dir', FLAGS.result_dir_prefix + FLAGS.eicu_cohort),
                       ('--eicu_cohort', FLAGS.eicu_cohort),
                       ('--dataname', dataname),
                       ('--result_suffix', '_' + expname),
                       ('--global_model_fn', FLAGS.global_model_fn),
                       ('--cohort_filepath', str(i) + '_' + expname + '.npy')] +
                      setting for i, setting in enumerate(model_settings)]

    # acknowledge the temporal dependence between the runs
    # first run cluster_settings, followed by model_settings
    # also make sure model_settings uses cluster settings' model
    run('cluster_moe.py', cluster_settings, gpus=gpus, n_concurrent_process=FLAGS.nc)
    if test_time:
        model_settings = [['--test_time', '--bootstrap'] + setting for setting in model_settings]
    run('moe.py', model_settings, gpus=gpus, n_concurrent_process=FLAGS.nc)

def experiment5(FLAGS, expname='mtl_oi', test_time=False,
                debug=None, dataname='eicu'):
    '''
    mtl outcome independent (AE)
    '''
    pct = FLAGS.train_data_subset_path.split('pct_')[1].split('_')[0]
    pct_num = os.path.basename(FLAGS.train_data_subset_path)[:-4] # remove '.pkl'
    expname = "pct{}_{}_{}".format(pct, pct_num, expname)

    if FLAGS.global_model_fn is None: return
    cluster_settings, model_settings = create_cluster_model_settings(FLAGS)

    if debug is not None:
        if type(debug) is list:
            cluster_settings = [cluster_settings[idx] for idx in debug]
            model_settings = [model_settings[idx] for idx in debug]
        else:
            idx = debug
            cluster_settings = cluster_settings[idx:idx+1]
            model_settings = model_settings[idx:idx+1]

    cluster_settings = [[('--model_type', 'AE'),
                         ('--train_data_subset_path', FLAGS.train_data_subset_path),
                         "--cluster_add_result_suffix",
                         ('--result_dir', FLAGS.result_dir_prefix + FLAGS.eicu_cohort),
                         ('--eicu_cohort', FLAGS.eicu_cohort),
                         ('--dataname', dataname),
                         ('--global_model_fn', FLAGS.global_model_fn),
                         ('--result_suffix', '_' + expname)] +
                        setting for setting in cluster_settings]
    model_settings = [[('--model_type', 'MULTITASK'),
                       ('--train_data_subset_path', FLAGS.train_data_subset_path),
                       ('--result_dir', FLAGS.result_dir_prefix + FLAGS.eicu_cohort),
                       ('--eicu_cohort', FLAGS.eicu_cohort),
                       ('--dataname', dataname),
                       ('--result_suffix', '_' + expname),
                       ('--global_model_fn', FLAGS.global_model_fn),
                       ('--cohort_filepath', str(i) + '_' + expname + '.npy')] +
                      setting for i, setting in enumerate(model_settings)]

    # acknowledge the temporal dependence between the runs
    # first run cluster_settings, followed by model_settings
    # also make sure model_settings uses cluster settings' model
    run('cluster_moe.py', cluster_settings, gpus=gpus, n_concurrent_process=FLAGS.nc)
    if test_time:
        model_settings = [['--test_time', '--bootstrap'] + setting for setting in model_settings]
    run('moe.py', model_settings, gpus=gpus, n_concurrent_process=FLAGS.nc)

def experiment6(FLAGS, expname='snapshot_od', test_time=False,
                debug=None, dataname='eicu'):
    '''
    snapshot outcome dependent (global)
    '''
    pct = FLAGS.train_data_subset_path.split('pct_')[1].split('_')[0]
    pct_num = os.path.basename(FLAGS.train_data_subset_path)[:-4] # remove '.pkl'
    expname = "pct{}_{}_{}".format(pct, pct_num, expname)

    if FLAGS.global_model_fn is None: return
    cluster_settings, model_settings = create_cluster_model_settings(FLAGS)

    if debug is not None:
        if type(debug) is list:
            cluster_settings = [cluster_settings[idx] for idx in debug]
            model_settings = [model_settings[idx] for idx in debug]
        else:
            idx = debug
            cluster_settings = cluster_settings[idx:idx+1]
            model_settings = model_settings[idx:idx+1]

    cluster_settings = [[('--model_type', 'GLOBAL'),
                         ('--train_data_subset_path', FLAGS.train_data_subset_path),
                         "--cluster_add_result_suffix",
                         ('--result_dir', FLAGS.result_dir_prefix + FLAGS.eicu_cohort),
                         ('--eicu_cohort', FLAGS.eicu_cohort),
                         ('--dataname', dataname),
                         ('--global_model_fn', FLAGS.global_model_fn),
                         ('--result_suffix', '_' + expname)] +
                        setting for setting in cluster_settings]
    model_settings = [[('--model_type', 'SNAPSHOT'),
                       ('--train_data_subset_path', FLAGS.train_data_subset_path),
                       ('--result_dir', FLAGS.result_dir_prefix + FLAGS.eicu_cohort),
                       ('--eicu_cohort', FLAGS.eicu_cohort),
                       ('--dataname', dataname),
                       ('--result_suffix', '_' + expname),
                       ('--global_model_fn', FLAGS.global_model_fn),
                       ('--cohort_filepath', str(i) + '_' + expname + '.npy')] +
                      setting for i, setting in enumerate(model_settings)]

    # acknowledge the temporal dependence between the runs
    # first run cluster_settings, followed by model_settings
    # also make sure model_settings uses cluster settings' model
    run('cluster_moe.py', cluster_settings, gpus=gpus, n_concurrent_process=FLAGS.nc)
    if test_time:
        model_settings = [['--test_time', '--bootstrap'] + setting for setting in model_settings]
    run('moe.py', model_settings, gpus=gpus, n_concurrent_process=FLAGS.nc)

def experiment7(FLAGS, expname='snapshot_val_curve', test_time=False,
                debug=None, dataname='eicu'):
    '''
    snapshot val curve
    '''
    pct = FLAGS.train_data_subset_path.split('pct_')[1].split('_')[0]
    pct_num = os.path.basename(FLAGS.train_data_subset_path)[:-4] # remove '.pkl'
    expname = "pct{}_{}_{}".format(pct, pct_num, expname)
    if FLAGS.pct_val < 1:
        # old:
        # expname = "{}_val{}".format(expname, int(FLAGS.pct_val * 100))
        # new:
        expname = "{}_valpct{}".format(expname, int(FLAGS.pct_val * 100))

    if FLAGS.global_model_fn is None: return
    cluster_settings, model_settings = create_cluster_model_settings(FLAGS)

    if debug is not None:
        if type(debug) is list:
            cluster_settings = [cluster_settings[idx] for idx in debug]
            model_settings = [model_settings[idx] for idx in debug]
        else:
            idx = debug
            cluster_settings = cluster_settings[idx:idx+1]
            model_settings = model_settings[idx:idx+1]

    cluster_settings = [[('--model_type', 'VAL_CURVE'),
                         ('--train_data_subset_path', FLAGS.train_data_subset_path),
                         ('--pct_val', FLAGS.pct_val),
                         "--cluster_add_result_suffix",
                         ('--result_dir', FLAGS.result_dir_prefix + FLAGS.eicu_cohort),
                         ('--eicu_cohort', FLAGS.eicu_cohort),
                         ('--dataname', dataname),
                         ('--global_model_fn', FLAGS.global_model_fn),
                         ('--result_suffix', '_' + expname)] +
                        setting for setting in cluster_settings]
    model_settings = [[('--model_type', 'SNAPSHOT'),
                       ('--pct_val', FLAGS.pct_val), # new
                       ('--result_dir', FLAGS.result_dir_prefix + FLAGS.eicu_cohort),
                       ('--eicu_cohort', FLAGS.eicu_cohort),
                       ('--dataname', dataname),
                       ('--result_suffix', '_' + expname),
                       ('--global_model_fn', FLAGS.global_model_fn),
                       ('--cohort_filepath', str(i) + '_' + expname + '.npy')] +
                      setting for i, setting in enumerate(model_settings)]

    # acknowledge the temporal dependence between the runs
    # first run cluster_settings, followed by model_settings
    # also make sure model_settings uses cluster settings' model
    run('cluster_moe.py', cluster_settings, gpus=gpus, n_concurrent_process=FLAGS.nc)
    if test_time:
        model_settings = [['--test_time',
                           # '--bootstrap'
        ] + setting for setting in model_settings]
    run('moe.py', model_settings, gpus=gpus, n_concurrent_process=FLAGS.nc)

def experiment8(FLAGS, expname='snapshot_oi', test_time=False,
                debug=None, dataname='eicu'):
    '''
    snapshot outcome independent (AE)
    '''
    pct = FLAGS.train_data_subset_path.split('pct_')[1].split('_')[0]
    pct_num = os.path.basename(FLAGS.train_data_subset_path)[:-4] # remove '.pkl'
    cluster_expname = "pct{}_{}_{}".format(pct, pct_num, "mtl_oi") # share the cluster
    expname = "pct{}_{}_{}".format(pct, pct_num, expname)

    if FLAGS.global_model_fn is None: return
    cluster_settings, model_settings = create_cluster_model_settings(FLAGS)

    if debug is not None:
        if type(debug) is list:
            cluster_settings = [cluster_settings[idx] for idx in debug]
            model_settings = [model_settings[idx] for idx in debug]
        else:
            idx = debug
            cluster_settings = cluster_settings[idx:idx+1]
            model_settings = model_settings[idx:idx+1]

    # cluster_settings = [[('--model_type', 'AE'),
    #                      ('--train_data_subset_path', FLAGS.train_data_subset_path),
    #                      "--cluster_add_result_suffix",
    #                      ('--result_dir', FLAGS.result_dir_prefix + FLAGS.eicu_cohort),
    #                      ('--eicu_cohort', FLAGS.eicu_cohort),
    #                      ('--dataname', dataname),
    #                      ('--global_model_fn', FLAGS.global_model_fn),
    #                      ('--result_suffix', '_' + cluster_expname)] +
    #                     setting for setting in cluster_settings]
    model_settings = [[('--model_type', 'SNAPSHOT'),
                       ('--train_data_subset_path', FLAGS.train_data_subset_path),
                       ('--result_dir', FLAGS.result_dir_prefix + FLAGS.eicu_cohort),
                       ('--eicu_cohort', FLAGS.eicu_cohort),
                       ('--dataname', dataname),
                       ('--result_suffix', '_' + expname),
                       ('--global_model_fn', FLAGS.global_model_fn),
                       ('--cohort_filepath', str(i) + '_' + cluster_expname + '.npy')] +
                      setting for i, setting in enumerate(model_settings)]

    # acknowledge the temporal dependence between the runs
    # first run cluster_settings, followed by model_settings
    # also make sure model_settings uses cluster settings' model
    # don't need to run cluster setting b/c exp5 should already ran this
    # run('cluster_moe.py', cluster_settings, gpus=gpus, n_concurrent_process=FLAGS.nc)
    if test_time:
        model_settings = [['--test_time', '--bootstrap'] + setting for setting in model_settings]
    run('moe.py', model_settings, gpus=gpus, n_concurrent_process=FLAGS.nc)

def main():
    FLAGS = get_args()
    '''
    experiments I need to run for eicu
    1. moe
    2. global
    3. mtl with [ae|global|val_curve]
    4. snapshot with [ae|global|val_curve]
    '''
    # experiment1(FLAGS)
    # experiment2(FLAGS)
    # #### need global model
    # experiment3(FLAGS)
    # experiment4(FLAGS)
    # experiment5(FLAGS) # slowest
    # experiment6(FLAGS)
    experiment7(FLAGS)    
    # experiment7(FLAGS, debug=[0,1,2,3], test_time=True)
    # experiment8(FLAGS) # also slow but once 5 is done, can reuse

if __name__ == '__main__':
    main()

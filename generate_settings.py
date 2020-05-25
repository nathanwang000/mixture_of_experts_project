'''
a setting is a list of argument like the following
[('--lr', 0.001), ('--wd', 0), '--pmt', ('--runname', runname)]
'''
import numpy as np
import os
from sklearn.externals import joblib
import argparse
from tune import run

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global_model_fn", default="22_global_exp.m", type=str,
                        help="best global model file name in /mortality_test/models/, default=10_global_exp.m")
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

def create_joint_settings(n_settings=30):
    '''
    this applies to global and moe because they don't require the
    clustering function to be given
    RETURN: a list of settings
    setting: [('--lr', 0.001), ('--wd', 0), '--pmt', ('--runname', runname)]
    '''
    setting_dir = 'settings'
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

def create_cluster_model_settings(n_settings=30):
    '''
    uses create_joint settings as base, assumes global model is given
    return model_settings, cluster_settings

    caution: user need to provide '--cohort_filepath' and '--global_model_fn'
    '''

    cluster_settings, model_settings = [], []
    settings = create_joint_settings(n_settings)

    fname = 'settings/cluster_model_settings.pkl'
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
        if np.random.choice(2) == 1:
            cluster_setting.append('--pmt')
        if np.random.choice(2) == 1:
            cluster_setting.append('--not_pt') # only affect validation curve approach
        cluster_settings.append(cluster_setting)

        # create remaining model settings: continue from create_joint_settings
        # note model_setting already have runname, so no need to add again
        model_setting = settings[i] # based on create_joint_settings
        model_setting.append(('--cohorts', 'custom'))
        if np.random.choice(2) == 1:
            model_setting.append('--sample_weights')
        # if np.random.choice(2) == 1: # todo: this increases dim, in conflict with pmt
        #     model_setting.append('--include_cohort_as_feature')
        if np.random.choice(2) == 1:
            model_setting.append('--pmt')
        model_settings.append(model_setting)

    # save the settings
    print('saving {}'.format(fname))
    joblib.dump((cluster_settings, model_settings), fname)
    return cluster_settings, model_settings

##### specific experiments
# experiments that don't need clustering
def experiment1(FLAGS, expname='moe_exp', test_time=False, dataname='mimic'):
    '''
    MoE experiment
    '''
    settings = create_joint_settings()
    tasks = [[('--model_type', 'MOE'),
              ('--dataname', dataname),
              ('--result_suffix', '_' + expname)] +
             setting for setting in settings]
    if test_time:
        tasks = [['--test_time'] + setting for setting in tasks]
    run('moe.py', tasks, gpus=[5, 6], n_concurrent_process=FLAGS.nc)

def experiment2(FLAGS, expname='global_exp', test_time=False, dataname='mimic'):
    '''
    Global model only experiment
    '''
    settings = create_joint_settings()
    tasks = [[('--model_type', 'GLOBAL'),
              ('--dataname', dataname),              
              ('--result_suffix', '_' + expname)] +
             setting for setting in settings]
    if test_time:
        tasks = [['--test_time'] + setting for setting in tasks]
    run('moe.py', tasks, gpus=[5, 6], n_concurrent_process=FLAGS.nc)

def experiment6(FLAGS, expname='MTL_careunit_exp', test_time=False):
    '''
    careunit MTL
    '''
    settings = create_joint_settings()
    tasks = [[('--model_type', 'MULTITASK'),
              ('--cohorts', 'careunit'),
              ('--result_suffix', '_' + expname)] +
             setting for setting in settings]
    if test_time:
        tasks = [['--test_time'] + setting for setting in tasks]
    run('moe.py', tasks, gpus=[5, 6], n_concurrent_process=FLAGS.nc)

def experiment9(FLAGS, expname='MTL_saps_exp', test_time=False):
    '''
    saps quartile based MTL
    '''
    settings = create_joint_settings()
    tasks = [[('--model_type', 'MULTITASK'),
              ('--result_suffix', '_' + expname),
              ('--cohorts', 'saps')] +
             setting for setting in settings]
    if test_time:
        tasks = [['--test_time'] + setting for setting in tasks]
    run('moe.py', tasks, gpus=[5, 6], n_concurrent_process=FLAGS.nc)

def experiment12(FLAGS, expname='separate_careunit_exp', test_time=False):
    '''
    careunit separate models
    '''
    settings = create_joint_settings()
    tasks = [[('--model_type', 'SEPARATE'),
              ('--cohorts', 'careunit'),
              ('--result_suffix', '_' + expname)] +
             setting for setting in settings]
    if test_time:
        tasks = [['--test_time'] + setting for setting in tasks]
    run('moe.py', tasks, gpus=[5, 6], n_concurrent_process=FLAGS.nc)

def experiment13(FLAGS, expname='separate_saps_exp', test_time=False):
    '''
    saps quartile separate models
    '''
    settings = create_joint_settings()
    tasks = [[('--model_type', 'SEPARATE'),
              ('--cohorts', 'saps'),
              ('--result_suffix', '_' + expname)] +
             setting for setting in settings]
    if test_time:
        tasks = [['--test_time'] + setting for setting in tasks]
    run('moe.py', tasks, gpus=[5, 6], n_concurrent_process=FLAGS.nc)

# experiments that require global model but not clustering
def experiment10(FLAGS, expname='snapshot_careunit_exp', test_time=False):
    '''
    careunit snapshot
    '''
    settings = create_joint_settings()
    tasks = [[('--model_type', 'SNAPSHOT'),
              ('--cohorts', 'careunit'),
              ('--global_model_fn', FLAGS.global_model_fn),
              ('--result_suffix', '_' + expname)] +
             setting for setting in settings]
    if test_time:
        tasks = [['--test_time'] + setting for setting in tasks]
    run('moe.py', tasks, gpus=[5, 6], n_concurrent_process=FLAGS.nc)

def experiment11(FLAGS, expname='snapshot_saps_exp', test_time=False):
    '''
    saps quartile based snapshot
    '''
    settings = create_joint_settings()
    tasks = [[('--model_type', 'SNAPSHOT'),
              ('--global_model_fn', FLAGS.global_model_fn),
              ('--result_suffix', '_' + expname),
              ('--cohorts', 'saps')] +
             setting for setting in settings]
    if test_time:
        tasks = [['--test_time'] + setting for setting in tasks]
    run('moe.py', tasks, gpus=[5, 6], n_concurrent_process=FLAGS.nc)

def experiment_debug(FLAGS, expname='debug', test_time=False, viz_time=False):
    '''
    debug settings: see read me for manually tuned performance
    '''
    settings = create_joint_settings()

    #### debug
    # idx = 10
    # settings = settings[idx:idx+1]
    settings = [[
        ('--lr', 0.001), ('--wd', 1e-4)
    ]]

    tasks = [[('--model_type', 'MULTITASK'),
              ('--epochs', 100),              
              ('--global_model_fn', FLAGS.global_model_fn),
              ('--result_suffix', '_' + expname),
              ('--cohorts', 'saps')] +
             setting for setting in settings]
    
    # tasks = [[('--model_type', 'MULTITASK'), # auroc 0.852
    #           ('--epochs', 100),
    #           ('--global_model_fn', FLAGS.global_model_fn),
    #           ('--result_suffix', '_' + expname),
    #           ('--cohort_filepath', 'sample_y_quartile.npy'),
    #           ('--cohorts', 'custom')] +
    #          setting for setting in settings]

    # tasks = [[('--model_type', 'GLOBAL'), # test auc: 0.872, val auc: 0.880
    #           ('--epochs', 100),
    #           ('--global_model_fn', FLAGS.global_model_fn),
    #           ('--result_suffix', '_' + expname),
    #           '--include_cohort_as_feature',
    #           ('--cohorts', 'saps')] +
    #          setting for setting in settings]
        
    if test_time:
        tasks = [['--test_time'] + setting for setting in tasks]
    if viz_time:
        tasks = [['--viz_time'] + setting for setting in tasks]
    run('moe.py', tasks, gpus=[5, 6], n_concurrent_process=FLAGS.nc)

# experiments that requires clustering
def experiment3(FLAGS, expname='global_plus_mtl_exp', test_time=False, debug=None, dataname='mimic'):
    '''
    global clustering followed by mtl
    '''
    if FLAGS.global_model_fn is None: return
    cluster_settings, model_settings = create_cluster_model_settings()

    if debug is not None:
        if type(debug) is list:
            cluster_settings = [cluster_settings[idx] for idx in debug]
            model_settings = [model_settings[idx] for idx in debug]
        else:
            idx = debug
            cluster_settings = cluster_settings[idx:idx+1]
            model_settings = model_settings[idx:idx+1]

    cluster_settings = [[('--model_type', 'GLOBAL'),
                         ('--dataname', dataname),                         
                         ('--global_model_fn', FLAGS.global_model_fn),
                         ('--result_suffix', '_' + expname)] +
                        setting for setting in cluster_settings]
    model_settings = [[('--model_type', 'MULTITASK'),
                       ('--dataname', dataname),                       
                       ('--result_suffix', '_' + expname),
                       ('--global_model_fn', FLAGS.global_model_fn),
                       ('--cohort_filepath', str(i) + '_' + expname + '.npy')] +
                      setting for i, setting in enumerate(model_settings)]

    # acknowledge the temporal dependence between the runs
    # first run cluster_settings, followed by model_settings
    # also make sure model_settings uses cluster settings' model
    run('cluster_moe.py', cluster_settings, gpus=[5, 6], n_concurrent_process=FLAGS.nc)
    if test_time:
        model_settings = [['--test_time'] + setting for setting in model_settings]
    run('moe.py', model_settings, gpus=[5, 6], n_concurrent_process=FLAGS.nc)

def experiment4(FLAGS, expname='ae_plus_mtl_exp', test_time=False, debug=None, dataname='mimic'):
    '''
    AE clustering followed by mtl
    '''
    if FLAGS.global_model_fn is None: return
    cluster_settings, model_settings = create_cluster_model_settings()

    if debug is not None:
        if type(debug) is list:
            cluster_settings = [cluster_settings[idx] for idx in debug]
            model_settings = [model_settings[idx] for idx in debug]
        else:
            idx = debug
            cluster_settings = cluster_settings[idx:idx+1]
            model_settings = model_settings[idx:idx+1]

    cluster_settings = [[('--model_type', 'AE'),
                         ('--global_model_fn', FLAGS.global_model_fn),
                         ('--dataname', dataname),                         
                         ('--result_suffix', '_' + expname)] +
                        setting for setting in cluster_settings]
    model_settings = [[('--model_type', 'MULTITASK'),
                       ('--dataname', dataname),                       
                       ('--result_suffix', '_' + expname),
                       ('--global_model_fn', FLAGS.global_model_fn),
                       ('--cohort_filepath', str(i) + '_' + expname + '.npy')] +
                      setting for i, setting in enumerate(model_settings)]

    # acknowledge the temporal dependence between the runs
    # first run cluster_settings, followed by model_settings
    # also make sure model_settings uses cluster settings' model
    run('cluster_moe.py', cluster_settings, gpus=[5, 6], n_concurrent_process=FLAGS.nc)
    if test_time:
        model_settings = [['--test_time'] + setting for setting in model_settings]
    run('moe.py', model_settings, gpus=[5, 6], n_concurrent_process=FLAGS.nc)

def experiment5(FLAGS, expname='val_curve_plus_mtl_exp', test_time=False, debug=None):
    '''
    val_curve clustering followed by mtl
    '''
    if FLAGS.global_model_fn is None: return
    cluster_settings, model_settings = create_cluster_model_settings()

    if debug is not None:
        if type(debug) is list:
            cluster_settings = [cluster_settings[idx] for idx in debug]
            model_settings = [model_settings[idx] for idx in debug]
        else:
            idx = debug
            cluster_settings = cluster_settings[idx:idx+1]
            model_settings = model_settings[idx:idx+1]

    cluster_settings = [[('--model_type', 'VAL_CURVE'),
                         ('--global_model_fn', FLAGS.global_model_fn),
                         ('--result_suffix', '_' + expname)] +
                        setting for setting in cluster_settings]
    model_settings = [[('--model_type', 'MULTITASK'),
                       ('--result_suffix', '_' + expname),
                       ('--global_model_fn', FLAGS.global_model_fn),
                       ('--cohort_filepath', str(i) + '_' + expname + '.npy')] +
                      setting for i, setting in enumerate(model_settings)]

    # acknowledge the temporal dependence between the runs
    # first run cluster_settings, followed by model_settings
    # also make sure model_settings uses cluster settings' model
    run('cluster_moe.py', cluster_settings, gpus=[5, 6], n_concurrent_process=FLAGS.nc)
    if test_time:
        model_settings = [['--test_time'] + setting for setting in model_settings]
    run('moe.py', model_settings, gpus=[5, 6], n_concurrent_process=FLAGS.nc)

def experiment7(FLAGS, expname='global_plus_snapshot_exp', test_time=False, debug=None):
    '''
    global clustering followed by snapshot
    '''
    if FLAGS.global_model_fn is None: return
    cluster_settings, model_settings = create_cluster_model_settings()

    if debug is not None:
        if type(debug) is list:
            cluster_settings = [cluster_settings[idx] for idx in debug]
            model_settings = [model_settings[idx] for idx in debug]
        else:
            idx = debug
            cluster_settings = cluster_settings[idx:idx+1]
            model_settings = model_settings[idx:idx+1]
            
    cluster_settings = [[('--model_type', 'GLOBAL'),
                         ('--global_model_fn', FLAGS.global_model_fn),
                         ('--result_suffix', '_' + expname)] +
                        setting for setting in cluster_settings]
    model_settings = [[('--model_type', 'SNAPSHOT'),
                       ('--result_suffix', '_' + expname),
                       ('--global_model_fn', FLAGS.global_model_fn),
                       ('--cohort_filepath', str(i) + '_' + expname + '.npy')] +
                      setting for i, setting in enumerate(model_settings)]

    # acknowledge the temporal dependence between the runs
    # first run cluster_settings, followed by model_settings
    # also make sure model_settings uses cluster settings' model
    run('cluster_moe.py', cluster_settings, gpus=[5, 6], n_concurrent_process=FLAGS.nc)
    if test_time:
        model_settings = [['--test_time'] + setting for setting in model_settings]
    run('moe.py', model_settings, gpus=[5, 6], n_concurrent_process=FLAGS.nc)

def experiment8(FLAGS, expname='val_curve_plus_snapshot_exp', test_time=False, debug=None):
    '''
    val_curve clustering followed by snapshot
    '''
    if FLAGS.global_model_fn is None: return
    cluster_settings, model_settings = create_cluster_model_settings()

    if debug is not None:
        if type(debug) is list:
            cluster_settings = [cluster_settings[idx] for idx in debug]
            model_settings = [model_settings[idx] for idx in debug]
        else:
            idx = debug
            cluster_settings = cluster_settings[idx:idx+1]
            model_settings = model_settings[idx:idx+1]
            
    cluster_settings = [[('--model_type', 'VAL_CURVE'),
                         ('--global_model_fn', FLAGS.global_model_fn),
                         ('--result_suffix', '_' + expname)] +
                        setting for setting in cluster_settings]
    model_settings = [[('--model_type', 'SNAPSHOT'),
                       ('--result_suffix', '_' + expname),
                       ('--global_model_fn', FLAGS.global_model_fn),
                       ('--cohort_filepath', str(i) + '_' + expname + '.npy')] +
                      setting for i, setting in enumerate(model_settings)]

    # acknowledge the temporal dependence between the runs
    # first run cluster_settings, followed by model_settings
    # also make sure model_settings uses cluster settings' model
    run('cluster_moe.py', cluster_settings, gpus=[5, 6], n_concurrent_process=FLAGS.nc)
    if test_time:
        model_settings = [['--test_time'] + setting for setting in model_settings]
    run('moe.py', model_settings, gpus=[5, 6], n_concurrent_process=FLAGS.nc)

def main():
    FLAGS = get_args()
    # experiment_debug(FLAGS, viz_time=False)

    # experiment1(FLAGS, test_time=False, dataname='eicu') # must
    # experiment2(FLAGS, test_time=False, dataname='eicu') # must
    # experiment6(FLAGS, test_time=False)
    # experiment9(FLAGS, test_time=False)
    # experiment12(FLAGS, test_time=False)
    # experiment13(FLAGS, test_time=False)

    #### global model required
    # experiment10(FLAGS, test_time=False)
    # experiment11(FLAGS, test_time=False)

    #### cluster and models
    experiment3(FLAGS, test_time=False, dataname='eicu') # must
    experiment4(FLAGS, test_time=False, dataname='eicu') # must
    # experiment5(FLAGS, test_time=False) # d, good to have
    # experiment7(FLAGS, test_time=False) # good to have
    # experiment8(FLAGS, test_time=False) # d, good to have

if __name__ == '__main__':
    main()

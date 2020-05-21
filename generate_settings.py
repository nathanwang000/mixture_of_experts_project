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
    parser.add_argument("--global_model_fn", default=None, type=str,
                        help="best global model file name in /mortality_test/models/, default=None")
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
            ('--lstm_layer_size', np.random.choice([16, 100, 300])),
            ('--num_dense_shared_layers', np.random.choice([0, 1, 2, 3])),
            ('--dense_shared_layer_size', np.random.choice([16, 100, 300])),
            ('--num_multi_layers', np.random.choice([0, 1, 2, 3])),
            ('--multi_layer_size', np.random.choice([16, 100, 300]))]
        settings.append(setting)

    print('saving {}'.format(fname))
    joblib.dump(settings, fname)
    return settings

def create_model_cluster_settings(global_model_fn, n_settings=30):
    '''
    uses create_joint settings as base, assumes global model is given
    return model_settings, cluster_settings
    '''
    assert global_model_fn is not None, "need global_model_fn"

    model_settings, cluster_settings = [], []
    settings = create_joint_settings(n_settings)

    fname = 'settings/model_cluster_settings.pkl'
    if os.path.exists(fname):
        print('settings already exists in {}, loading...'.format(fname))
        model_settings, cluster_settings = joblib.load(fname)
        if n_settings <= len(model_settings):
            return model_settings[:n_settings], cluster_settings[:n_settings]
        else:
            n_settings -= len(model_settings) # continue generating
            settings = settings[-n_settings:]

    # create remaining model settings: continue from create_joint_settings
    for setting in settings:
        setting.append(('--global_model_fn', global_model_fn))         
        if np.random.choice(2) == 1:
            setting.append('--sample_weights')
        if np.random.choice(2) == 1:
            setting.append('--include_cohort_as_feature')
        if np.random.choice(2) == 1:
            setting.append('--pmt')
        model_settings.append(setting)

    # create remaining cluster settings
    for _ in range(n_settings):
        setting = [
            ('--global_model_fn', global_model_fn),
            ('--runname', str(len(cluster_settings))), # name of the saved model            
            ('--lr', 10**np.random.uniform(-2,-4)),
            ('--wd', 10**np.random.uniform(-3,-10)),
            ('--num_clusters', np.random.choice([2, 3, 4, 5])),
            ('--latent_dim', np.random.choice([16, 100, 300, 500]))]
        if np.random.choice(2) == 1:
            setting.append('--pmt')
        if np.random.choice(2) == 1:
            setting.append('--not_pt') # only affect validation curve approach
        cluster_settings.append(setting)

    # save the settings
    print('saving {}'.format(fname))
    joblib.dump((model_settings, cluster_settings), fname)
    return model_settings, cluster_settings

##### specific experiments
# experiments that don't need clustering
def experiment1(FLAGS, expname='moe_exp'):
    '''
    MoE experiment
    '''
    settings = create_joint_settings()
    tasks = [[('--model_type', 'MOE'), ('--result_suffix', '_' + expname)] +
             setting for setting in settings]
    run('moe.py', tasks, gpus=[5, 6], n_concurrent_process=FLAGS.nc)

def experiment2(FLAGS, expname='global_exp'):
    '''
    Global model only experiment
    '''
    settings = create_joint_settings()
    tasks = [[('--model_type', 'GLOBAL'), ('--result_suffix', '_' + expname)] +
             setting for setting in settings]
    run('moe.py', tasks, gpus=[5, 6], n_concurrent_process=FLAGS.nc)

def experiment7(FLAGS, expname='MTL_first_unit_exp'):
    '''
    first unit MTL
    '''
    settings = create_joint_settings()
    tasks = [[('--model_type', 'MULTITASK'), ('--result_suffix', '_' + expname)] +
             setting for setting in settings]
    run('moe.py', tasks, gpus=[5, 6], n_concurrent_process=FLAGS.nc)

# experiments that requires clustering
def experiment3(FLAGS, expname='mtl_and_global_exp'):
    '''
    MTL + global
    '''
    # todo: add custom
    if FLAGS.global_model_fn is not None: return
    model_settings, cluster_settings = create_model_cluster_settings(FLAGS.global_model_fn)
    model_settings = [[('--model_type', 'MULTITASK'), ('--result_suffix', '_' + expname)] +
                      setting for setting in model_settings]
    cluster_settings = [[('--model_type', 'GLOBAL'), ('--result_suffix', '_' + expname)] +
                        setting for setting in cluster_settings]

    # todo: acknowledge the temporal dependence between the runs
    # first run cluster_settings, followed by model_settings
    # also make sure model_settings uses cluster settings' model
    idx = 11
    print(model_settings[idx])
    print(cluster_settings[idx])
    # run('moe.py', cluster_settings, gpus=[5, 6], n_concurrent_process=FLAGS.nc)
    # run('moe.py', model_settings, gpus=[5, 6], n_concurrent_process=FLAGS.nc)

    
def main():
    FLAGS = get_args()
    # experiment1(FLAGS)
    experiment2(FLAGS)
    # experiment3(FLAGS)
    # experiment4(FLAGS)
    # experiment5(FLAGS)
    # experiment6(FLAGS)
    experiment7(FLAGS)
    # experiment8(FLAGS)
    # experiment9(FLAGS)
        
if __name__ == '__main__':
    main()

def old_main():
    FLAGS = get_args()

    # setting_dir = 'settings'
    # if not os.path.exists(setting_dir):
    #     os.makedirs(setting_dir)

    # ### model running using first unit as cohorts
    # ### e.g. mtl_pytorch, separate, global_pytorch, moe
    # fname = 'model_settings.pkl'
    # if os.path.exists(fname):
    #         print('settings already exists in {}, skipping...'.format(fname))
    # else:
    #     print('creating {}'.format(fname))
    #     settings = create_model_without_global_settings()
    #     joblib.dump(settings, fname)

    # ### model running using first unit as cohorts and requires a global model
    # ### eg. snapshot, mtl_pt; pmt + snaphsot, pmt + mtl_pt,
    # ### pmt + global_pytorch, pmt + mtl_pytorch, pmt + separate, pmt + moe
    # '''
    # need to add "--global_model_fn" to the setting already saved
    # need to add "--pmt" to the setting already saved
    # need to add "--not_pt" to the setting already saved
    # '''

    # '''clustering approaches must be followed by custom models
    # for simplicity, I still assume global model_fn even if approaches like
    # AE, and INPUT don't require a global model because down stream model may still require
    # a global model: e.g., AE + SNAPSHOT
    # '''
    # ### clustering models that doesn't need global model
    # ### e.g., AE, INPUT (too slow to run, maybe ommit)
    # fname = 'cluster_settings.pkl'
    # if os.path.exists(fname):
    #         print('settings already exists in {}, skipping...'.format(fname))
    # else:
    #     print('creating {}'.format(fname))
    #     settings = create_cluster_without_global_settings()
    #     joblib.dump(settings, fname)

    # ### clustering models that need a global model
    # ### e.g., GLOBAL, VAL_CURVE, pmt + AE, pmt + INPUT, pmt + GLOBAL, pmt + VAL_CURVE
    # '''
    # need to add "--custom" to the setting already saved
    # need to add "--global_model_fn" to the setting already saved
    # need to add "--pmt" to the setting already saved
    # need to add "--not_pt" to the setting already saved
    # '''

    # ### model running using custom cohorts
    # ### e.g. mtl_pytorch, separate # no need to run global and moe b/c it's the same as before
    # '''
    # need to add "--custom" to the setting already saved
    # need to add "--global_model_fn" to the setting already saved
    # need to add "--pmt" to the setting already saved
    # '''

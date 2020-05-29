from joblib import Parallel, delayed
import numpy as np
from sklearn import metrics, utils

def bootstrap_helper(y, y_pred, metric, random_state=None):
    y, y_pred = utils.resample(y, y_pred, replace=True, random_state=random_state)
    return metric(y, y_pred)
    
def bootstrap_metric(y, y_pred, num_bootstrap_samples=100, metric=metrics.roc_auc_score, n_jobs=4):
    '''
    ARGS:
        y: numpy array of output
        y_pred: numpy array of prediction output
        metric: a function taking (y, y_pred) as input and returns a number
        num_bootstrap_samples: number of bootstrap samples
        n_jobs: number of parallel jobs
    RETURNS:
        scores: numpy array of the size (num_bootstrap_samples, )
    '''
    scores = Parallel(n_jobs=n_jobs)(
        delayed(bootstrap_helper)(y, y_pred, metric, i) for i in range(num_bootstrap_samples))
    return scores
    
#### for plotting
def bootstrap_func(y, y_pred, random_state=None):
    y, y_pred = utils.resample(y, y_pred, replace=True, random_state=random_state)
    return metrics.roc_curve(y, y_pred), metrics.roc_auc_score(y, y_pred)
        
def get_roc_CI(y, y_pred, num_bootstrap_samples=100, n_jobs=4):
    '''
    y: numpy array of true y
    y_pred: numpy array of predicted y
    '''
    roc_curves, auc_scores = zip(*Parallel(n_jobs=n_jobs)(
        delayed(bootstrap_func)(y, y_pred, i) for i in range(num_bootstrap_samples)))
    
    print('Test AUC: {:.3f}'.format(metrics.roc_auc_score(y, y_pred)))
    print('Test AUC: ({:.3f}, {:.3f}) percentile 95% CI'.format(np.percentile(auc_scores, 2.5),
                                                                np.percentile(auc_scores, 97.5)))

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for fpr, tpr, _ in roc_curves:
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(metrics.auc(fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + 1.96 * std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - 1.96 * std_tpr, 0)
    return roc_curves, auc_scores, mean_fpr, tprs_lower, tprs_upper

def plot_roc_CI(y, y_pred, label=""):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 4))
    fpr, tpr, thresholds = metrics.roc_curve(y, y_pred)
    plt.plot(fpr, tpr, lw=1.25, label=label) 
    roc_curves, auc_scores, mean_fpr, tprs_lower, tprs_upper = get_roc_CI(y, y_pred)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=.1)
    plt.plot([0,1], [0,1], 'k:')
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.legend()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
import warnings
warnings.filterwarnings("ignore")

from IPython.core.interactiveshell import InteractiveShell # multipl outputs in same cells
InteractiveShell.ast_node_interactivity = "all"

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_regression


def mrmr(X_train, y_train, k):
    F = pd.Series(f_regression(X_train, y_train)[0], index = X_train.columns)
    corr = pd.DataFrame(.00001, index = X_train.columns, columns = X_train.columns)

    # initialize list of selected features and list of excluded features
    selected = []
    not_selected = X_train.columns.to_list()

    # repeat K times
    for i in range(3):
    
        # compute (absolute) correlations between the last selected feature and all the (currently) excluded features
        if i > 0:
            last_selected = selected[-1]
            corr.loc[not_selected, last_selected] = X_train[not_selected].corrwith(X_train[last_selected]).abs().clip(.00001)
            
        # compute FCQ score for all the (currently) excluded features (this is Formula 2)
        score = F.loc[not_selected] / corr.loc[not_selected, selected].mean(axis = 1).fillna(.00001)
        # find best feature, add it to selected and remove it from not_selected
        best = score.index[score.argmax()]
        selected.append(best)
        not_selected.remove(best)
    
    result = score.sort_values(ascending=False)
    placeholder = np.array(result)
    placeholder = placeholder / placeholder.sum()
    feature_names, importances = list(result.index)[:k], placeholder[:k]
    return feature_names, importances


def permutation_importances(rf, X_train, y_train, metric):
    baseline = metric(rf, X_train, y_train)
    imp = []
    for col in X_train.columns:
        save = X_train[col].copy()
        X_train[col] = np.random.permutation(X_train[col])
        m = metric(rf, X_train, y_train)
        X_train[col] = save
        imp.append(baseline - m)
    return np.array(imp)

def oob_regression_r2_score(rf, X_train, y_train):
    """
    Compute out-of-bag (OOB) R^2 for a scikit-learn random forest
    regressor. We learned the guts of scikit's RF from the BSD licensed
    code:
    https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/ensemble/forest.py#L702
    """
    X = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y = y_train.values if isinstance(y_train, pd.Series) else y_train

    n_samples = len(X)
    predictions = np.zeros(n_samples)
    n_predictions = np.zeros(n_samples)
    for tree in rf.estimators_:
        unsampled_indices = _get_unsampled_indices(tree, n_samples)
        tree_preds = tree.predict(X[unsampled_indices, :])
        predictions[unsampled_indices] += tree_preds
        n_predictions[unsampled_indices] += 1

    if (n_predictions == 0).any():
        warnings.warn("Too few trees; some variables do not have OOB scores.")
        n_predictions[n_predictions == 0] = 1

    predictions /= n_predictions




def imp_plot(feature_names, importances, std=None):

    fig, ax = plt.subplots(figsize=(4,2))  # make one subplot (ax) on the figure
    if std is None:
        std = np.repeat(1, len(feature_names))
    data_plot = list( zip(feature_names, importances, std) )
    data_plot = pd.DataFrame(sorted(data_plot, key=lambda x: x[1], reverse=True))
    feature_names, importances, std = data_plot[0], data_plot[1], data_plot[2]

    if std[0] == 1: 
        barcontainers = ax.bar(feature_names, importances, color='#FEE08F')
    else:
        barcontainers = ax.bar(feature_names, importances, color='#FEE08F', yerr=std)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)
    ax.set_ylabel("Importance", fontsize=7)

    for rect in barcontainers.patches:
        rect.set_linewidth(.5)
        rect.set_edgecolor('grey')
        
    # ax.set_xticks(feature_names)                     # make sure we have a ticket for every cyl value
    ax.set_xticklabels([n for n in feature_names], rotation=45, fontsize=7)
    plt.show();

def pca_var_plot(feature_names, data):

    fig, ax = plt.subplots(figsize=(4,2)) 
    pca = PCA().fit(data)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('')
    plt.ylabel('cumulative variance')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)
    ax.set_xticklabels([n for n in feature_names], rotation=45, fontsize=7)
    plt.show();
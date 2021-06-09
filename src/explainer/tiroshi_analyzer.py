import os, argparse, random, time, csv;

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import auc, roc_curve, roc_auc_score, accuracy_score

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC;
from scipy import stats

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.backends.backend_pdf import PdfPages
from enum import Enum

class Type(Enum):
    Prediction = 1
    Correlation = 2

def get_feat_correlation(cell_profiles):
    corr_mat = 1.0 - pairwise_distances(cell_profiles.T, metric='correlation');
    # logger.info('corr_mat={}'.format(corr_mat.shape));
    # figure = plt.figure(figsize=(12, 12));
    # ax = figure.add_subplot(1, 1, 1);
    # ax.hist(corr_mat.ravel(), bins=200, );
    # ax.set_xlabel('Correlation', fontsize=25);
    # ax.set_ylabel('# of feature pair', fontsize=25);
    # plt.show();

    corr_mat = np.squeeze(corr_mat);
    corr_mat = corr_mat[~np.isnan(corr_mat)];
    return np.mean(corr_mat), np.std(corr_mat);


def classify_cell_clust(cell_profiles, clust_ids, target_clustid, cv_fold=3, total_repeat=1):
    assert target_clustid in np.unique(clust_ids);
    # param_grid = {'C': np.power(5.0, np.arange(-5, 6))};
    param_grid = {'C': np.power(3.0, np.arange(-3, 4))};

    true_indices = np.where(clust_ids == target_clustid)[0];
    false_indices = np.where(clust_ids != target_clustid)[0];
    logger.info('clustid={}\ttrue_indices={}\tfalse_indices={}'.format(target_clustid, len(true_indices), len(false_indices) ));
    cell_labels = np.zeros(len(clust_ids)); cell_labels[true_indices] = 1;

    auc_list = [];
    for repeat in range(total_repeat):
        kf = StratifiedKFold(n_splits=cv_fold, shuffle=True, random_state=repeat);
        for train_indices, test_indices in kf.split(cell_profiles, cell_labels):
            logger.info('train_positive={}/{}\ttest_positive={}/{}'.format(np.sum(cell_labels[train_indices] > 0), len(train_indices), np.sum(cell_labels[test_indices] > 0), len(test_indices),  ));

            cell_profiles_train = cell_profiles[train_indices]; cell_labels_train = cell_labels[train_indices];
            cell_profiles_test = cell_profiles[test_indices]; cell_labels_test = cell_labels[test_indices];

            svm = SVC(kernel='rbf', random_state=0, class_weight='balanced', probability=False);
            searchCV = GridSearchCV(svm, param_grid, scoring='roc_auc', cv=cv_fold);
            searchCV = searchCV.fit(cell_profiles_train, cell_labels_train);
            optC = searchCV.best_params_['C'];

            svm = SVC(kernel='rbf', random_state=0, C=searchCV.best_params_['C'], class_weight='balanced', probability=False);
            svm.fit(cell_profiles_train, cell_labels_train);
            auc_score = roc_auc_score(y_true=cell_labels_test, y_score=svm.decision_function(cell_profiles_test));
            logger.info('auc_score={}\toptC={}'.format(auc_score, optC ));
            auc_list.append(auc_score);
    logger.info('auc_list={}'.format(auc_list ));
    return auc_list;


def process_cell_clust_by_feat_subset(gene_relevance_url, cell_profiles, gene_ids, clust_ids, target_clustid, output_url, process_type, use_abs=False ):
    gene_score_list = [];
    with open(gene_relevance_url) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t');
        for row in reader:
            gene = row['Gene'];
            raw_score = float(row['Score']);
            abs_score = np.fabs(raw_score);
            gene_score_list.append((gene, raw_score, abs_score));

    if use_abs: gene_score_list.sort(key=lambda x: -x[2]);
    else: gene_score_list.sort(key=lambda x: x[1]);

    ranked_gene_list = np.asarray([x[0] for x in gene_score_list]);

    logger.info('ranked_gene_list={}'.format(len(ranked_gene_list)));
    logger.info('gene_ids but not ranked_gene_list={}'.format(np.setdiff1d(gene_ids, ranked_gene_list)));
    logger.info('ranked_gene_list but not gene_ids={}'.format(np.setdiff1d(ranked_gene_list, gene_ids)));


    assert len(np.unique(gene_ids)) == len(np.intersect1d(np.unique(ranked_gene_list), np.unique(gene_ids)));
    gene_to_index_map = dict(zip(gene_ids, np.arange(len(gene_ids)) ));

    # percent_arr = np.concatenate((np.asarray(range(1, 10), dtype=float), np.asarray(range(10, 40, 2), dtype=float), np.asarray(range(40, 101, 4), dtype=float), )) / 100;
    # percent_arr = np.concatenate((np.asarray(range(1, 10), dtype=float), np.asarray(range(10, 20, 2), dtype=float), np.asarray(range(20, 40, 4), dtype=float), np.asarray(range(40, 101, 10), dtype=float), )) / 100;
    percent_arr = np.concatenate((np.asarray(range(1, 10, 1), dtype=float) / 1000, np.asarray(range(1, 11, 1), dtype=float) / 100, ));

    feat_subset_arr = np.asarray(np.round(percent_arr * len(gene_ids)), dtype=int);
    logger.info('feat_subset_arr={}'.format(feat_subset_arr));

    f = open(output_url, "w");
    if process_type == Type.Prediction:
        f.write("Selected_gene_percent\tSelected_gene_cnt\tAuc_scores\n");
    elif process_type == Type.Correlation:
        f.write("Selected_gene_percent\tSelected_gene_cnt\tCorr_mean\tCorr_std\n");
    else: assert False;

    for feat_subset_percent, feat_subset_cnt in zip(percent_arr, feat_subset_arr):
        logger.info('feat_subset_percent={}\tfeat_subset_cnt={}'.format(feat_subset_percent, feat_subset_cnt ));
        ranked_gene_subset = ranked_gene_list[:feat_subset_cnt];
        ranked_gene_indices = np.unique([gene_to_index_map[str(gene)] for gene in ranked_gene_subset if gene in gene_to_index_map]);

        if process_type == Type.Prediction:
            auc_list = classify_cell_clust(cell_profiles[:, ranked_gene_indices], clust_ids, target_clustid, );
            f.write("{}\t{}\t{}\n".format(feat_subset_percent, feat_subset_cnt, ','.join(['{:.4f}'.format(x) for x in auc_list]), ));
        elif process_type == Type.Correlation:
            corr_mean, corr_std = get_feat_correlation(cell_profiles[:, ranked_gene_indices]);
            f.write("{}\t{}\t{}\t{}\n".format(feat_subset_percent, feat_subset_cnt, corr_mean, corr_std, ));
        else: assert False;

    f.close();



def plot_subset_pred():
    result_url = '../data/tiroshi/results_kmeans/results_kmeans_all_subset_pred.csv';
    assert os.path.isfile(result_url);

    label_list = ['GCE', 'GRS', 'JSD', 'DESeq2', 'SmoothGrad', 'SHAP', 'ACE(feat=128)', 'ACE(feat=256)', 'ACE(feat=512)'];
    color_list = ['#78b62b', '#fc8e43', '#ffa1ad', '#825996', '#0087ad', '#00187c', '#f92b3f', '#f9394c', '#fa5263'];

    percent_arr = np.asarray(range(1, 11, 1), dtype=float) / 1000;
    percent_arr = np.asarray(percent_arr, dtype=str);
    feat_num_arr = [24, 47, 71, 95, 118, 142, 166, 189, 213, 237 ];

    label_percent_auc_map = {};
    for label in label_list:
        label_percent_auc_map[label] = {};
        for percent in percent_arr:
            label_percent_auc_map[label][percent] = np.asarray([], dtype=float);

    with open(result_url) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t');
        for row in reader:
            percent = row['Selected_gene_percent'];
            if percent not in percent_arr: continue;

            for label in label_list:
                auc_arr = np.asarray(row[label].split(','), dtype=float);
                label_percent_auc_map[label][percent] = np.concatenate((label_percent_auc_map[label][percent], auc_arr ));

    figure = plt.figure(figsize=(12, 12));
    ax = figure.add_subplot(1, 1, 1);
    ax.autoscale();

    for label, color in zip(label_list, color_list):
        mean_list = []; std_list = [];
        for percent in percent_arr:
            mean_list.append(np.mean(label_percent_auc_map[label][percent]));
            # std_list.append(np.std(label_percent_auc_map[label][percent]));
            std_list.append(stats.sem(label_percent_auc_map[label][percent], axis=None));

        ax.errorbar(feat_num_arr, mean_list, yerr=std_list, color=color, linewidth=4, elinewidth=0.2, capsize=2, label=label,  ); # yerr=std_list,

    ax.legend(fontsize=32);
    ax.set_xlabel('# of gene included', fontsize=32);
    ax.set_ylabel('AUROC', fontsize=32);
    ax.xaxis.set_tick_params(labelsize=25);
    ax.yaxis.set_tick_params(labelsize=25);
    # ax.set_xscale('log');

    # ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x,y: '{:d}%'.format(int(x)) ));
    ax.grid(linestyle=':');
    ax.set_title('From most to least important', fontsize=35);

    # plt.show()
    pp = PdfPages(result_url.replace('.csv', '.pdf'));
    pp.savefig(figure, bbox_inches='tight');
    pp.close();
    plt.close(figure);
    del figure;

def plot_subset_corr():
    result_url = '../data/tiroshi/results_kmeans/results_kmeans_all_subset_corr.csv';
    assert os.path.isfile(result_url);

    label_list = ['GCE', 'GRS', 'JSD', 'DESeq2', 'SmoothGrad', 'SHAP', 'ACE(feat=128)', 'ACE(feat=256)', 'ACE(feat=512)'];
    color_list = ['#78b62b', '#fc8e43', '#ffa1ad', '#825996', '#0087ad', '#00187c', '#f92b3f', '#f9394c', '#fa5263'];

    percent_arr = np.asarray(range(1, 11, 1), dtype=float) / 1000;
    percent_arr = np.asarray(percent_arr, dtype=str);
    feat_num_arr = [24, 47, 71, 95, 118, 142, 166, 189, 213, 237 ];

    label_percent_mean_map = {}; label_percent_std_map = {};
    for label in label_list:
        label_percent_mean_map[label] = {};
        label_percent_std_map[label] = {};
        for percent in percent_arr:
            label_percent_mean_map[label][percent] = [];
            label_percent_std_map[label][percent] = [];

    with open(result_url) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t');
        for row in reader:
            percent = row['Selected_gene_percent'];
            if percent not in percent_arr: continue;

            for label in label_list:
                mean_val = float(row['{}_mean'.format(label)]);
                std_val = float(row['{}_std'.format(label)]);
                label_percent_mean_map[label][percent].append(mean_val);
                label_percent_std_map[label][percent].append(std_val);


    figure = plt.figure(figsize=(12, 12));
    ax = figure.add_subplot(1, 1, 1);
    ax.autoscale();

    for label, color in zip(label_list, color_list):
        mean_list = []; std_list = [];
        for percent, feat_num in list(zip(percent_arr, feat_num_arr)):
            mean_list.append(np.mean(label_percent_mean_map[label][percent]));
            std_list.append(np.mean(label_percent_std_map[label][percent]) / np.sqrt(feat_num));

        ax.errorbar(feat_num_arr, mean_list, yerr=std_list, color=color, linewidth=4, elinewidth=0.2, capsize=2, label=label,  ); # yerr=std_list,

    ax.legend(fontsize=32);
    ax.set_xlabel('# of gene included', fontsize=32);
    ax.set_ylabel('Pearson correlation', fontsize=32);
    ax.xaxis.set_tick_params(labelsize=25);
    ax.yaxis.set_tick_params(labelsize=25);
    # ax.set_xscale('log');

    # ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x,y: '{:d}%'.format(int(x)) ));
    ax.grid(linestyle=':');
    ax.set_title('From most to least important', fontsize=35);

    # plt.show()
    pp = PdfPages(result_url.replace('.csv', '.pdf'));
    pp.savefig(figure, bbox_inches='tight');
    pp.close();
    plt.close(figure);
    del figure;

def get_data(args, take_log=True, normalize=False):
    logger.info('input_url={}'.format(args.input_url)); assert os.path.exists(args.input_url);
    save_url = args.input_url.replace('.txt', '_log{}_norm{}.npz'.format(int(take_log), int(normalize))); logger.info('save_url={}'.format(save_url));
    assert os.path.isfile(save_url);

    data = np.load(save_url);
    gene_ids = data['gene_ids'];
    cell_ids = data['cell_ids'];
    clust_ids = data['clust_ids'];
    cell_profiles = data['cell_profiles'];

    logger.info('gene_ids={}'.format(len(gene_ids), ));
    logger.info('cell_ids={}'.format(len(cell_ids), ));
    logger.info('clust_ids={}'.format(clust_ids.shape, ));
    logger.info('cell_profiles={}'.format(cell_profiles.shape, ));

    return cell_profiles, cell_ids, np.asarray(gene_ids, dtype=str), clust_ids;


def main(args):
    assert os.path.isfile(args.gene_relevance_url);
    cell_profiles, cell_ids, gene_ids, clust_ids = get_data(args);

    ### subset feature classification
    # output_url = args.gene_relevance_url.replace('.csv', '.subset_pred.clustid{}_abs{}.csv'.format(args.target_clustid, args.use_abs));
    # if not os.path.isfile(output_url): process_cell_clust_by_feat_subset(args.gene_relevance_url, cell_profiles, gene_ids, clust_ids, args.target_clustid, output_url, process_type=Type.Prediction, use_abs=args.use_abs);

    ### subset feature correlation
    # output_url = args.gene_relevance_url.replace('.csv', '.subset_corr.clustid{}_abs{}.csv'.format(args.target_clustid, args.use_abs));
    # if not os.path.isfile(output_url): process_cell_clust_by_feat_subset(args.gene_relevance_url, cell_profiles, gene_ids, clust_ids, args.target_clustid, output_url, process_type=Type.Correlation, use_abs=args.use_abs);

    ### plotting
    # plot_subset_pred();
    plot_subset_corr();


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description');
    parser.add_argument('--input_url', type=str, help='input_url');
    parser.add_argument('--gene_relevance_url', type=str, help='gene_relevance_url', default='');

    parser.add_argument('--target_clustid', type=int, help='target_clustid');
    parser.add_argument('--use_abs', type=int, help='use_abs');

    args = parser.parse_args();
    main(args);


"""example command"""
# python explainer/tiroshi_analyzer.py --input_dir ../data/tiroshi --gene_relevance_url ../data/PBMC/results/results_pbmc_beta0.004_margin0_fd-1_latent512,256,128,2/genes_cw_onevsrest_lamda100.0_iter3000_lr0.001_clustid1_abs0.csv --target_clustid 1 --use_abs 1

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
    return np.mean(corr_mat), np.std(corr_mat);


def classify_cell_clust(cell_profiles, clust_ids, target_clustid, cv_fold=3, total_repeat=1):
    assert target_clustid in np.unique(clust_ids);
    # param_grid = {'C': np.power(5.0, np.arange(-5, 6))};
    param_grid = {'C': np.power(3.0, np.arange(-7, 8))};

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

    ranked_gene_list = [x[0] for x in gene_score_list];
    assert len(gene_ids) == len(np.intersect1d(ranked_gene_list, gene_ids));
    gene_to_index_map = dict(zip(gene_ids, np.arange(len(gene_ids)) ));

    # percent_arr = np.concatenate((np.asarray(range(1, 10), dtype=float), np.asarray(range(10, 101, 2), dtype=float) )) / 100;
    # percent_arr = np.asarray(range(1, 101), dtype=float) / 100;
    percent_arr = np.concatenate((np.asarray(range(1, 10), dtype=float), np.asarray(range(10, 40, 2), dtype=float), np.asarray(range(40, 101, 4), dtype=float), )) / 100;

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
        ranked_gene_indices = np.unique([gene_to_index_map[str(gene)] for gene in ranked_gene_subset]);

        if process_type == Type.Prediction:
            auc_list = classify_cell_clust(cell_profiles[:, ranked_gene_indices], clust_ids, target_clustid, );
            f.write("{}\t{}\t{}\n".format(feat_subset_percent, feat_subset_cnt, ','.join(['{:.4f}'.format(x) for x in auc_list]), ));
        elif process_type == Type.Correlation:
            corr_mean, corr_std = get_feat_correlation(cell_profiles[:, ranked_gene_indices]);
            f.write("{}\t{}\t{}\t{}\n".format(feat_subset_percent, feat_subset_cnt, corr_mean, corr_std, ));
        else: assert False;

    f.close();


def plot_subset_upsetplot(target_clustid, top_k=100):

    def get_topk_feat(gene_relevance_url, top_k, use_abs=False, ):
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
        ranked_gene_list = [x[0] for x in gene_score_list];
        return ranked_gene_list[:top_k];

    parent_dir = '../data/PBMC/results_kmeans';

    label_gce = 'GCE'; use_abs_gce = True;
    relevance_url_gce = os.path.join(parent_dir, 'gce_kmeans_clustid{}.csv'.format(target_clustid));
    gene_list_gce = get_topk_feat(relevance_url_gce, top_k, use_abs=use_abs_gce, );

    label_grs = 'GRS'; use_abs_grs = True;
    relevance_url_grs = os.path.join(parent_dir, 'grs_kmeans_clustid{}.csv'.format(target_clustid));
    gene_list_grs = get_topk_feat(relevance_url_grs, top_k, use_abs=use_abs_grs, );

    label_jsd = 'JSD'; use_abs_jsd = False;
    relevance_url_jsd = os.path.join(parent_dir, 'jsd_kmeans_clustid{}.csv'.format(target_clustid));
    gene_list_jsd = get_topk_feat(relevance_url_jsd, top_k, use_abs=use_abs_jsd, );

    label_deseq2 = 'DESeq2'; use_abs_deseq2 = False;
    relevance_url_deseq2 = os.path.join(parent_dir, 'deseq2_kmeans_clustid{}.csv'.format(target_clustid));
    gene_list_deseq2 = get_topk_feat(relevance_url_deseq2, top_k, use_abs=use_abs_deseq2, );

    data_dir = os.path.join(parent_dir, 'results_pbmc_beta0.01_margin0_fd256_latent512,256,128,2/');

    label_smoothgrad = 'SmoothGrad'; use_abs_smoothgrad = True;
    relevance_url_smoothgrad = os.path.join(data_dir, 'genes_smoothgrad_onevsrest_clustid{}_abs0.csv'.format(target_clustid, ));
    gene_list_smoothgrad = get_topk_feat(relevance_url_smoothgrad, top_k, use_abs=use_abs_smoothgrad, );

    label_shap = 'SHAP'; use_abs_shap = True;
    relevance_url_shap = os.path.join(data_dir, 'genes_shap_onevsrest_clustid{}_abs0.csv'.format(target_clustid, ));
    gene_list_shap = get_topk_feat(relevance_url_shap, top_k, use_abs=use_abs_shap, );

    label_ace = 'ACE'; use_abs_ace = True;
    relevance_url_ace = os.path.join(data_dir, 'genes_cw_onevsrest_lamda100.0_iter5000_lr0.001_clustid{}_abs0.csv'.format(target_clustid, ));
    gene_list_ace = get_topk_feat(relevance_url_ace, top_k, use_abs=use_abs_ace, );

    gene_list_all = np.unique(np.concatenate((gene_list_gce, gene_list_grs, gene_list_jsd, gene_list_deseq2, gene_list_smoothgrad, gene_list_shap, gene_list_ace )));
    logger.info('gene_list_all={}'.format(len(gene_list_all) ));
    label_genes_map = {label_gce:gene_list_gce, label_grs:gene_list_grs, label_jsd:gene_list_jsd, label_deseq2:gene_list_deseq2, label_smoothgrad: gene_list_smoothgrad, label_shap:gene_list_shap, label_ace:gene_list_ace };

    other_label_list = [label_gce, label_grs, label_jsd, label_deseq2, label_smoothgrad, label_shap];
    f = open(os.path.join(parent_dir, 'results_subset_uniqgenes_clustid{}_top{}.txt'.format(target_clustid, top_k )), "w");
    for gene in gene_list_ace:
        if not np.any([(gene in label_genes_map[label]) for label in other_label_list]):
            f.write("{}\n".format(gene));
    f.close();

    from upsetplot import UpSet, plot, from_contents
    figure = plt.figure(figsize=(20, 8));
    data_pd = from_contents(label_genes_map);
    plot(data_pd, show_counts='%d', sort_by='cardinality', subset_size='count', );

    plt.ylabel('Intersection size', fontsize=20);
    plt.yticks(fontsize=18)
    # plt.show();
    fig_url = os.path.join(parent_dir, 'results_subset_upsetplot_clustid{}_top{}.pdf'.format(target_clustid, top_k ));
    plt.savefig(fig_url);


### I know the simulation results shouldn't be plotted here, but just for the convenience.
def plot_simulation_subset_upsetplot(target_clustid, top_k=20):

    def get_topk_feat(gene_relevance_url, top_k, target_clustid ):
        gene_list = [];
        with open(gene_relevance_url) as fp:
            for cnt, line in enumerate(fp):
                arr = line.rstrip().split(',');

                clustid = int(arr[0]); gene = arr[1];
                if clustid != target_clustid: continue;
                gene_list.append(gene);
                # logger.info('clustid={}\tgene={}'.format(clustid, gene, ));
        return gene_list[:top_k];

    # parent_dir = '../data/simulation/clean/one_vs_rest/';
    parent_dir = '../data/simulation/hard/one_vs_rest/';

    label_gce = 'GCE'; relevance_url_gce = os.path.join(parent_dir, 'gce_one_vs_rest.csv');
    gene_list_gce = get_topk_feat(relevance_url_gce, top_k, target_clustid, );

    label_grs = 'GRS'; relevance_url_grs = os.path.join(parent_dir, 'grs_one_vs_rest.csv');
    gene_list_grs = get_topk_feat(relevance_url_grs, top_k, target_clustid, );

    label_jsd = 'JSD'; relevance_url_jsd = os.path.join(parent_dir, 'jsd_one_vs_rest.csv');
    gene_list_jsd = get_topk_feat(relevance_url_jsd, top_k, target_clustid, );

    label_deseq2 = 'DESeq2'; relevance_url_deseq2 = os.path.join(parent_dir, 'deseq2_one_vs_rest.csv');
    gene_list_deseq2 = get_topk_feat(relevance_url_deseq2, top_k, target_clustid, );

    label_smoothgrad = 'SmoothGrad'; relevance_url_smoothgrad = os.path.join(parent_dir, 'ace_one_vs_rest_saliency_smoothgrad_post1.csv');
    gene_list_smoothgrad = get_topk_feat(relevance_url_smoothgrad, top_k, target_clustid, );

    label_shap = 'SHAP'; relevance_url_shap = os.path.join(parent_dir, 'ace_one_vs_rest_shap_post1.csv');
    gene_list_shap = get_topk_feat(relevance_url_shap, top_k, target_clustid, );

    label_ace = 'ACE'; relevance_url_ace = os.path.join(parent_dir, 'ace_one_vs_rest_carlini_wagner_lamda100.0_iter2000_lr0.001_post1.csv');
    gene_list_ace = get_topk_feat(relevance_url_ace, top_k, target_clustid, );

    gene_list_all = np.unique(np.concatenate((gene_list_gce, gene_list_grs, gene_list_jsd, gene_list_deseq2, gene_list_smoothgrad, gene_list_shap, gene_list_ace )));
    logger.info('gene_list_all={}'.format(len(gene_list_all) ));
    label_genes_map = {label_gce:gene_list_gce, label_grs:gene_list_grs, label_jsd:gene_list_jsd, label_deseq2:gene_list_deseq2, label_smoothgrad: gene_list_smoothgrad, label_shap:gene_list_shap, label_ace:gene_list_ace };

    other_label_list = [label_gce, label_grs, label_jsd, label_deseq2, label_smoothgrad, label_shap];
    f = open(os.path.join(parent_dir, 'results_subset_uniqgenes_clustid{}_top{}.txt'.format(target_clustid, top_k )), "w");
    for gene in gene_list_ace:
        if not np.any([(gene in label_genes_map[label]) for label in other_label_list]):
            f.write("{}\n".format(gene));
    f.close();

    from upsetplot import UpSet, plot, from_contents
    figure = plt.figure(figsize=(20, 8));
    data_pd = from_contents(label_genes_map);
    plot(data_pd, show_counts='%d', sort_by='cardinality', subset_size='count', );

    plt.ylabel('Intersection size', fontsize=20);
    plt.yticks(fontsize=18)
    # plt.show();
    fig_url = os.path.join(parent_dir, 'results_subset_upsetplot_clustid{}_top{}.pdf'.format(target_clustid, top_k ));
    plt.savefig(fig_url);



def plot_subset_pred():
    # result_url = '../data/PBMC/results/results_all_subset_pred.csv';
    result_url = '../data/PBMC/results_kmeans/results_kmeans_all_subset_pred.csv';
    assert os.path.isfile(result_url);

    label_list = ['GCE', 'GRS', 'JSD', 'DESeq2', 'SmoothGrad', 'SHAP', 'ACE'];
    color_list = ['#78b62b', '#fc8e43', '#ffa1ad', '#825996', '#0087ad', '#00187c', '#f92b3f']
    percent_arr = np.concatenate((np.asarray(range(1, 10), dtype=int), np.asarray(range(10, 40, 2), dtype=int), np.asarray(range(40, 101, 4), dtype=int),));

    label_percent_auc_map = {};
    for label in label_list:
        label_percent_auc_map[label] = {};
        for percent in percent_arr:
            label_percent_auc_map[label][percent] = np.asarray([], dtype=float);

    with open(result_url) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t');
        for row in reader:
            percent = int(float(row['Selected_gene_percent'])*100);
            clustid = int(row['clust_id']);
            # if clustid == 4: continue;

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

        ax.errorbar(percent_arr, mean_list, yerr=std_list, color=color, linewidth=4, elinewidth=0.2, capsize=2, label=label,  ); # yerr=std_list,

    ax.legend(fontsize=32);
    ax.set_xlabel('# of gene included', fontsize=32);
    ax.set_ylabel('AUROC', fontsize=32);
    ax.xaxis.set_tick_params(labelsize=25);
    ax.yaxis.set_tick_params(labelsize=25);
    ax.set_xscale('log');

    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x,y: '{:d}%'.format(int(x)) ));
    ax.grid(linestyle=':');
    ax.set_title('From most to least important', fontsize=35);

    # plt.show()
    pp = PdfPages(result_url.replace('.csv', '.pdf'));
    pp.savefig(figure, bbox_inches='tight');
    pp.close();
    plt.close(figure);
    del figure;

def plot_subset_corr():
    # result_url = '../data/PBMC/results/results_all_subset_corr.csv';
    result_url = '../data/PBMC/results_kmeans/results_kmeans_all_subset_corr.csv';
    assert os.path.isfile(result_url);

    label_list = ['GCE', 'GRS', 'JSD', 'DESeq2', 'SmoothGrad', 'SHAP', 'ACE'];
    color_list = ['#78b62b', '#fc8e43', '#ffa1ad', '#825996', '#0087ad', '#00187c', '#f92b3f']
    percent_arr = np.concatenate((np.asarray(range(1, 10), dtype=int), np.asarray(range(10, 40, 2), dtype=int), np.asarray(range(40, 101, 4), dtype=int),));

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
            percent = int(float(row['Selected_gene_percent'])*100);
            clustid = int(row['clust_id']);

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
        for percent in percent_arr:
            mean_list.append(np.mean(label_percent_mean_map[label][percent]));
            # std_list.append(np.mean(label_percent_std_map[label][percent]));
            std_list.append(stats.sem(label_percent_std_map[label][percent], axis=None));

        ax.errorbar(percent_arr, mean_list, yerr=std_list, color=color, linewidth=4, elinewidth=0.2, capsize=2, label=label,  ); # yerr=std_list,

    ax.legend(fontsize=32);
    ax.set_xlabel('# of gene included', fontsize=32);
    ax.set_ylabel('Pearson correlation', fontsize=32);
    ax.xaxis.set_tick_params(labelsize=25);
    ax.yaxis.set_tick_params(labelsize=25);
    ax.set_xscale('log');

    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x,y: '{:d}%'.format(int(x)) ));
    ax.grid(linestyle=':');
    ax.set_title('From most to least important', fontsize=35);

    # plt.show()
    pp = PdfPages(result_url.replace('.csv', '.pdf'));
    pp.savefig(figure, bbox_inches='tight');
    pp.close();
    plt.close(figure);
    del figure;




def get_data(args, ):
    logger.info('input_dir={}'.format(args.input_dir)); assert os.path.exists(args.input_dir);

    cell_profiles = np.genfromtxt(os.path.join(args.input_dir, '3K_PBMC_matrix.txt'), delimiter=',');
    gene_ids = np.genfromtxt(os.path.join(args.input_dir, '3K_PBMC_gene_ids.txt'), dtype=None);
    cell_ids = np.genfromtxt(os.path.join(args.input_dir, '3K_PBMC_cell_ids.txt'), dtype=None);

    if 'kmeans' in args.gene_relevance_url: clust_ids = np.genfromtxt(os.path.join(args.input_dir, '3K_PBMC_clust_ids_kmeans.txt')).astype(int) + 1;
    else: clust_ids = np.genfromtxt(os.path.join(args.input_dir, '3K_PBMC_clust_ids.txt')).astype(int) + 1;

    logger.info('cell_profiles={}'.format(cell_profiles.shape));
    logger.info('cell_ids={}'.format(len(cell_ids)));
    logger.info('gene_ids={}'.format(len(gene_ids)));
    logger.info('clust_ids={}'.format(len(clust_ids)));

    return cell_profiles, cell_ids, np.asarray(gene_ids, dtype=str), clust_ids;


def main(args):
    # assert os.path.isfile(args.gene_relevance_url);
    # cell_profiles, cell_ids, gene_ids, clust_ids = get_data(args);

    ### subset feature classification
    # output_url = args.gene_relevance_url.replace('.csv', '.subset_pred.clustid{}_abs{}.csv'.format(args.target_clustid, args.use_abs));
    # if not os.path.isfile(output_url): process_cell_clust_by_feat_subset(args.gene_relevance_url, cell_profiles, gene_ids, clust_ids, args.target_clustid, output_url, process_type=Type.Prediction, use_abs=args.use_abs);

    ### subset feature correlation
    # output_url = args.gene_relevance_url.replace('.csv', '.subset_corr.clustid{}_abs{}.csv'.format(args.target_clustid, args.use_abs));
    # if not os.path.isfile(output_url): process_cell_clust_by_feat_subset(args.gene_relevance_url, cell_profiles, gene_ids, clust_ids, args.target_clustid, output_url, process_type=Type.Correlation, use_abs=args.use_abs);

    ### plotting
    # plot_subset_pred();
    # plot_subset_corr();
    # for top_k in [18, 92, 184]:  plot_subset_upsetplot(target_clustid=args.target_clustid, top_k=top_k);
    for clustid in [1,2,3,4,5]: plot_simulation_subset_upsetplot(target_clustid=clustid, top_k=20);


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description');
    parser.add_argument('--input_dir', type=str, help='input_dir');
    parser.add_argument('--gene_relevance_url', type=str, help='gene_relevance_url', default='');

    parser.add_argument('--target_clustid', type=int, help='target_clustid');
    parser.add_argument('--use_abs', type=int, help='use_abs');

    args = parser.parse_args();
    main(args);


"""example command"""
# python explainer/pbmc_analyzer.py --input_dir ../data/PBMC --gene_relevance_url ../data/PBMC/results/results_pbmc_beta0.004_margin0_fd-1_latent512,256,128,2/genes_cw_onevsrest_lamda100.0_iter3000_lr0.001_clustid1_abs0.csv --target_clustid 1 --use_abs 1

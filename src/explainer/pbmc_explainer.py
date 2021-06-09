import os, argparse, random, time, csv;

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import numpy as np;
from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA;

import tensorflow as tf; # tim: using tensorflow==1.3.1

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

from SAUCIE.model import SAUCIE
from SAUCIE.loader import Loader


def output_relevance_one_vs_rest(output_dir, clustid_relevance_map, gene_ids, use_abs=False, tag=''):
    """output the relevance for each clust_id"""
    for clust_id in clustid_relevance_map:
        logger.info('clust_id={}'.format(clust_id, ));
        if use_abs:
            relevance_arr = np.mean(np.fabs(clustid_relevance_map[clust_id]), axis=0);
            gene_relev_list = zip(relevance_arr, gene_ids);
            gene_relev_list.sort(key=lambda x: -x[0]);
        else:
            relevance_arr = np.mean(clustid_relevance_map[clust_id], axis=0);
            gene_relev_list = zip(relevance_arr, gene_ids);
            gene_relev_list.sort(key=lambda x: x[0]);

        f = open(os.path.join(output_dir, "genes_{}_clustid{}_abs{}.csv".format(tag, clust_id, int(use_abs) )), "w");
        f.write("Rank\tGene\tScore\n");
        for idx in range(len(gene_relev_list)):
            score, gene = gene_relev_list[idx];
            f.write("{}\t{}\t{}\n".format(idx+1, gene, score));
        f.close();


def output_relevance_one_vs_one(output_dir, clustid_pair_relevance_map, gene_ids, use_abs=False, tag=''):
    """output the relevance for each pair of clust_id"""

    for clustid_tp in clustid_pair_relevance_map:
        clust_c_id, clust_k_id = clustid_tp;
        logger.info('clust_c_id={}\tclust_k_id={}'.format(clust_c_id, clust_k_id, ));

        if use_abs:
            relevance_arr = np.mean(np.fabs(clustid_pair_relevance_map[clustid_tp]), axis=0);
            gene_relev_list = zip(relevance_arr, gene_ids);
            gene_relev_list.sort(key=lambda x: -x[0]);
        else:
            relevance_arr = np.mean(clustid_pair_relevance_map[clustid_tp], axis=0);
            gene_relev_list = zip(relevance_arr, gene_ids);
            gene_relev_list.sort(key=lambda x: x[0]);

        f = open(os.path.join(output_dir, "genes_{}_clustid{}_{}_abs{}.csv".format(tag, clust_c_id, clust_k_id, int(use_abs))), "w");
        for idx in range(len(gene_relev_list)):
            score, gene = gene_relev_list[idx];
            f.write("{},{},{}\n".format(idx + 1, gene, score));
        f.close();


def get_data(args, ):
    logger.info('input_dir={}'.format(args.input_dir)); assert os.path.exists(args.input_dir);

    cell_profiles = np.genfromtxt(os.path.join(args.input_dir, '3K_PBMC_matrix.txt'), delimiter=',');
    gene_ids = np.genfromtxt(os.path.join(args.input_dir, '3K_PBMC_gene_ids.txt'), dtype=None);
    cell_ids = np.genfromtxt(os.path.join(args.input_dir, '3K_PBMC_cell_ids.txt'), dtype=None);

    if 'kmeans' in args.output_dir: clust_ids = np.genfromtxt(os.path.join(args.input_dir, '3K_PBMC_clust_ids_kmeans.txt')).astype(int) + 1;
    else: clust_ids = np.genfromtxt(os.path.join(args.input_dir, '3K_PBMC_clust_ids.txt')).astype(int) + 1;

    logger.info('cell_profiles={}'.format(cell_profiles.shape));
    logger.info('cell_ids={}'.format(len(cell_ids)));
    logger.info('gene_ids={}'.format(len(gene_ids)));
    logger.info('clust_ids={}'.format(len(clust_ids)));

    return cell_profiles, cell_ids, gene_ids, clust_ids;

def visualize_x_or_embedding(x_mat, clust_ids, save_url, use_umap=True ):
    assert len(x_mat) == len(clust_ids);

    if use_umap:
        import umap;
        mapper = umap.UMAP(random_state=1).fit(x_mat);
        x_mat_2d = mapper.transform(x_mat);
    else:
        mapper = PCA(n_components=2);
        mapper.fit(x_mat);
        x_mat_2d = mapper.transform(x_mat);

    figure = plt.figure(figsize=(12, 12));
    sc = plt.scatter(x_mat_2d[:, 0], x_mat_2d[:, 1], s=5, c=clust_ids, cmap='Spectral');
    # plt.show();

    pp = PdfPages(save_url);
    pp.savefig(figure, bbox_inches='tight');
    pp.close();
    plt.close(figure);
    del figure;


def get_gene_clust_proteinatlas(ground_truth_url):
    gene_clust_values_map = {};
    with open(ground_truth_url) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t');
        for row in reader:
            gene = row['Gene name'];
            cell = row['Blood cell'];
            # TPM = float(row['TPM']);
            # pTPM = float(row['pTPM']);
            pTPM = float(row['NX']);

            if gene not in gene_clust_values_map:
                gene_clust_values_map[gene] = {};
                for id in range(1,8): gene_clust_values_map[gene][id] = [];

            clust_id = -1;
            if 'memory CD4 T-cell' == cell or 'naive CD4 T-cell' == cell : clust_id = 1;
            elif 'classical monocyte' == cell: clust_id = 2;
            elif 'memory B-cell' == cell or 'naive B-cell' == cell: clust_id = 3;
            elif 'memory CD8 T-cell' == cell or 'naive CD8 T-cell' == cell: clust_id = 4;
            elif 'NK-cell' == cell: clust_id = 5;
            elif 'intermediate monocyte' == cell or 'non-classical monocyte' == cell: clust_id = 6;
            elif 'plasmacytoid DC' == cell: clust_id = 7;

            if clust_id < 0: continue;
            # logger.info('gene={}\tcell={}\tclust_id={}\tTPM={}\tpTPM={}'.format(gene, cell, clust_id, TPM, pTPM ));
            gene_clust_values_map[gene][clust_id].append(pTPM);

    gene_clust_maxvalues_map = {};
    for gene in gene_clust_values_map:
        gene_clust_maxvalues_map[gene] = {};
        for clust_id in gene_clust_values_map[gene]:
            gene_clust_maxvalues_map[gene][clust_id] = np.max(gene_clust_values_map[gene][clust_id]);

    return gene_clust_maxvalues_map;

def get_gene_clust_cellmarker(ground_truth_url):

    bcell_list = []; tmp_list = [];
    with open(ground_truth_url) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t');
        for row in reader:
            tissuetype = row['tissueType'].lower();
            if 'blood' not in tissuetype: continue;

            cancertype = row['cancerType'].lower();
            if 'normal' not in cancertype: continue;

            cellname = row['cellName'];
            cell_markers = [x.strip() for x in row['cellMarker'].split(',')];

            # if 'CD4+ T cell' in cellname | 'CD4+ T cell' == cellname
            # if 'CD8+ T cell' in cellname | 'CD8+ T cell' == cellname
            # if 'B cell' in cellname:

            if 'B cell' in cellname:
                bcell_list = np.concatenate((bcell_list, cell_markers ));
                tmp_list.append(cellname)


    logger.info('bcell_list={}'.format(np.unique(bcell_list)));
    logger.info(np.unique(tmp_list))

def eval_gene_relevance(relevance_url, gene_clust_maxvalues_map, clust_id):
    logger.info(relevance_url); assert os.path.isfile(relevance_url);

    clustid_type_map = {1: 'CD4 T-cell', 2: 'CD14 Monocyte', 3: 'B-cell', 4: 'CD8 T-cell', 5: 'NK-cell', 6: 'FCGR3A Monocyte', 7: 'Dendritic-cell'}

    gene_clust_sum_map = {};
    for gene in gene_clust_maxvalues_map: gene_clust_sum_map[gene] = np.sum([gene_clust_maxvalues_map[gene][clust_id] for clust_id in gene_clust_maxvalues_map[gene]]);

    values_list_pos = []; values_list_neg = [];
    with open(relevance_url) as input:
        for line in input.readlines():
            arr = line.strip().split(',');
            gene = arr[1];
            relevance_score = float(arr[2]);

            if gene not in gene_clust_maxvalues_map: continue;
            if gene.startswith('RP') and '-' in gene: continue;
            values_list_pos.append((gene, relevance_score, gene_clust_maxvalues_map[gene][clust_id], (gene_clust_sum_map[gene]-gene_clust_maxvalues_map[gene][clust_id]) / 6.0  ));
            values_list_neg.append((gene, relevance_score, gene_clust_maxvalues_map[gene][clust_id], (gene_clust_sum_map[gene]-gene_clust_maxvalues_map[gene][clust_id]) / 6.0  ));

    ### maximum positive perturbation
    values_list_pos.sort(key=lambda x: -x[1]);
    ### minimum negative perturbation
    values_list_neg.sort(key=lambda x: x[1]);


    figure = plt.figure(figsize=(20, 10));
    ax = figure.add_subplot(1, 2, 1);
    ax.plot([x[1] for x in values_list_pos], [x[2] for x in values_list_pos], 's', marker='o', label='pTPM');
    ax.set_xlabel('Perturbation', fontsize=25);
    ax.set_ylabel('pTPM', fontsize=25);

    subset_num = 30;
    ax = figure.add_subplot(1, 2, 2);
    ax.plot([x[2] for x in values_list_pos[:subset_num]], [x[3] for x in values_list_pos[:subset_num]], 's', marker='o', label='Positive perturbation');
    ax.plot([x[2] for x in values_list_neg[:subset_num]], [x[3] for x in values_list_neg[:subset_num]], 's', marker='o', label='Negative perturbation');
    ax.set_xlabel('pTPM in target type', fontsize=25);
    ax.set_ylabel('pTPM in remaining types', fontsize=25);
    ax.legend(fontsize=28);

    min_value = min(np.min([x[2] for x in values_list_pos[:subset_num]]), np.min([x[3] for x in values_list_pos[:subset_num]]), np.min([x[2] for x in values_list_neg[:subset_num]]), np.min([x[3] for x in values_list_neg[:subset_num]]), );
    max_value = max(np.max([x[2] for x in values_list_pos[:subset_num]]), np.max([x[3] for x in values_list_pos[:subset_num]]), np.max([x[2] for x in values_list_neg[:subset_num]]), np.max([x[3] for x in values_list_neg[:subset_num]]), );
    ax.plot([min_value, max_value], [min_value, max_value], linestyle='--', linewidth=2, color='black');

    max_pos_thres = np.mean(np.maximum([x[2] for x in values_list_pos[:subset_num]], [x[3] for x in values_list_pos[:subset_num]]));
    max_neg_thres = np.mean(np.maximum([x[2] for x in values_list_neg[:subset_num]], [x[3] for x in values_list_neg[:subset_num]]));

    for gene, val_x, val_y in zip([x[0] for x in values_list_pos[:subset_num]], [x[2] for x in values_list_pos[:subset_num]], [x[3] for x in values_list_pos[:subset_num]] ):
        if max(val_x, val_y) < max_pos_thres: continue;
        ax.annotate(gene, xy=(val_x, val_y), xytext=(val_x+1, val_y+1), );

    for gene, val_x, val_y in zip([x[0] for x in values_list_neg[:subset_num]], [x[2] for x in values_list_neg[:subset_num]], [x[3] for x in values_list_neg[:subset_num]] ):
        if max(val_x, val_y) < max_neg_thres: continue;
        ax.annotate(gene, xy=(val_x, val_y), xytext=(val_x+1, val_y+1), );

    plt.title(clustid_type_map[clust_id], fontdict={'fontsize' : 25});
    plt.show();


class LRP:
    def __init__(self, clust_ids, cell_profiles, cell_embeddings, saucie):
        self.clust_ids = clust_ids;
        self.uniq_clust_ids = np.unique(self.clust_ids);
        self.clust_num = len(self.uniq_clust_ids);

        self.cell_profiles = cell_profiles;
        self.cell_embeddings = cell_embeddings;
        assert len(self.clust_ids) == self.cell_embeddings.shape[0];

        self.saucie = saucie;
        self.embed_weights, self.embed_biases, self.embed_activations = self.saucie.get_embedder_tensors();
        self.clust_weights, self.clust_biases, self.clust_activations = self.saucie.get_clust_tensors();
        self.X = self.embed_activations[0];

        self.clustid_pos_sample_profiles_map = self.relevance_preparation();


    def calc_saliency(self, saliency_type=0 ):
        clustid_relevance_map = {};
        for clust_c_id in self.uniq_clust_ids:
            _, _, _, clust_c_minpoolAct_tensor = self.get_tensor_by_clustid(clust_c_id);
            clust_grads_tensor = tf.gradients(clust_c_minpoolAct_tensor, self.X)[0];

            x0 = np.copy(self.clustid_pos_sample_profiles_map[clust_c_id]);

            try:

                if saliency_type == 0:
                    """vanilla gradient"""
                    clust_grads = self.saucie.sess.run(clust_grads_tensor, feed_dict={self.X: x0});
                elif saliency_type == 1:
                    """Smooth Gradient"""
                    clust_grads = np.zeros_like(x0); repeat=20; stdev = np.std(x0) * 0.2;
                    for iter in range(repeat):
                        noise = np.random.normal(0, stdev, x0.shape);
                        clust_grads += self.saucie.sess.run(clust_grads_tensor, feed_dict={self.X: x0 + noise});
                elif saliency_type == 2:
                    """Integrated Gradients"""
                    clust_grads = np.zeros_like(x0); baseline = np.zeros_like(x0); steps=20;
                    for alpha in np.linspace(0, 1, steps):
                        clust_grads += self.saucie.sess.run(clust_grads_tensor, feed_dict={self.X: alpha * x0});
                elif saliency_type == 3:
                    """SHAP: conda install -c conda-forge shap """
                    import shap;
                    e = shap.GradientExplainer((self.X, clust_c_minpoolAct_tensor), x0, session=self.saucie.sess, local_smoothing=0);
                    clust_grads = e.shap_values(x0);
                else: assert False;
                clust_grads_mean = np.sum(clust_grads, axis=1);
                inf_or_nan_indices = np.where(np.isinf(clust_grads_mean) | np.isnan(clust_grads_mean))[0];
                logger.info('x0={}\tinf_or_nan_indices={}'.format(x0.shape, inf_or_nan_indices.shape ));
                if len(inf_or_nan_indices) == len(x0): raise RuntimeError('explanation failed by encountering Inf/Nan values! Try a smaller beta value! ');
                valid_indices = np.setdiff1d(list(range(len(x0))), inf_or_nan_indices);

                clustid_relevance_map[clust_c_id] = clust_grads[valid_indices];
                logger.info('clust_id={}\tclust_grads={}'.format(clust_c_id, clustid_relevance_map[clust_c_id].shape ))

            except: logger.info("Exception encoutered!");

        return clustid_relevance_map;

    def calc_carlini_wagner_one_vs_rest(self, lamda=1e2, max_iter=5000, lr=2e-3, margin=0 ):

        """explain each cluster """
        clustid_relevance_map = {};
        for clust_c_id in self.uniq_clust_ids:
            _, _, _, clust_c_minpoolAct_tensor = self.get_tensor_by_clustid(clust_c_id);

            clust_k_minpoolAct_tensor_list = [];
            for clust_k_id in self.uniq_clust_ids:
                if clust_c_id == clust_k_id: continue;
                logger.info('clust_c_id={}\tclust_k_id={}'.format(clust_c_id, clust_k_id, ));
                _, _, _, clust_k_minpoolAct_tensor = self.get_tensor_by_clustid(clust_k_id);
                clust_k_minpoolAct_tensor_list.append(clust_k_minpoolAct_tensor);
            logger.info('clust_k_minpoolAct_tensor_list={}'.format(len(clust_k_minpoolAct_tensor_list)));

            clust_k_minpoolAct_tensor = clust_k_minpoolAct_tensor_list[0];
            for clust_k_idx in range(1, len(clust_k_minpoolAct_tensor_list)):
                clust_k_minpoolAct_tensor = tf.math.maximum(clust_k_minpoolAct_tensor, clust_k_minpoolAct_tensor_list[clust_k_idx]);

            clust_loss_tensor = tf.math.maximum(margin + clust_c_minpoolAct_tensor - clust_k_minpoolAct_tensor, 0);
            clust_loss_grads_tensor = tf.gradients(clust_loss_tensor, self.X)[0];

            """starting carlini_wagner optimization"""
            x0 = np.copy(self.clustid_pos_sample_profiles_map[clust_c_id]); curr_x = x0 + 1e-2;
            for iter in range(max_iter):
                clust_loss_grads = self.saucie.sess.run(clust_loss_grads_tensor, feed_dict={self.X: curr_x});

                if np.any([np.isinf(x) for x in clust_loss_grads]) or np.any([np.isnan(x) for x in clust_loss_grads]):
                    clust_loss_grads_mean = np.sum(clust_loss_grads, axis=1);
                    inf_or_nan_indices = np.where(np.isinf(clust_loss_grads_mean) | np.isnan(clust_loss_grads_mean) )[0];
                    logger.info('x0={}\tinf_or_nan_indices={}'.format(x0.shape, inf_or_nan_indices.shape ));

                    if len(inf_or_nan_indices) == len(x0): raise RuntimeError('explanation failed by encountering Inf/Nan values! Try a smaller beta value! ');
                    valid_indices = np.setdiff1d(list(range(len(x0))), inf_or_nan_indices );
                    x0 = x0[valid_indices]; curr_x = x0 + 1e-10;
                    continue;

                x_to_x0_sgn = np.asarray((curr_x - x0) >= 0, dtype=float);
                x_to_x0_sgn[x_to_x0_sgn <= 0] = -1;

                curr_x = curr_x - lr * (x_to_x0_sgn + lamda * clust_loss_grads);
                x_to_x0_loss = np.sum(np.fabs(curr_x - x0));
                clust_loss = np.sum(np.fabs(self.saucie.sess.run(clust_loss_tensor, feed_dict={self.X: curr_x})));
                if (iter+1) % 50 == 0: logger.info('iter={}\tx_to_x0_loss={}\tclust_loss={}'.format(iter, x_to_x0_loss, clust_loss, ));

            clustid_relevance_map[clust_c_id] = (curr_x - x0);
        return clustid_relevance_map;

    def calc_carlini_wagner_one_vs_one(self, lamda=1e2, max_iter=5000, lr=2e-3, margin=0 ):

        """explain each cluster pair"""
        clustid_pair_relevance_map = {};
        for clust_c_id in self.uniq_clust_ids:
            _, _, _, clust_c_minpoolAct_tensor = self.get_tensor_by_clustid(clust_c_id);

            for clust_k_id in self.uniq_clust_ids:
                if clust_c_id == clust_k_id: continue;
                logger.info('clust_c_id={}\tclust_k_id={}'.format(clust_c_id, clust_k_id, ));
                _, _, _, clust_k_minpoolAct_tensor = self.get_tensor_by_clustid(clust_k_id);

                clust_loss_tensor = tf.math.maximum(margin + clust_c_minpoolAct_tensor - clust_k_minpoolAct_tensor, 0);
                clust_loss_grads_tensor = tf.gradients(clust_loss_tensor, self.X)[0];

                """starting carlini_wagner optimization"""
                x0 = np.copy(self.clustid_pos_sample_profiles_map[clust_c_id]); curr_x = x0 + 1e-2;
                for iter in range(max_iter):
                    clust_loss_grads = self.saucie.sess.run(clust_loss_grads_tensor, feed_dict={self.X: curr_x});

                    if np.any([np.isinf(x) for x in clust_loss_grads]) or np.any([np.isnan(x) for x in clust_loss_grads]):
                        clust_loss_grads_mean = np.sum(clust_loss_grads, axis=1);
                        inf_or_nan_indices = np.where(np.isinf(clust_loss_grads_mean) | np.isnan(clust_loss_grads_mean))[0];
                        logger.info('x0={}\tinf_or_nan_indices={}'.format(x0.shape, inf_or_nan_indices.shape));

                        if len(inf_or_nan_indices) == len(x0): raise RuntimeError('explanation failed by encountering Inf/Nan values! Try a smaller beta value! ');
                        valid_indices = np.setdiff1d(list(range(len(x0))), inf_or_nan_indices );
                        x0 = x0[valid_indices]; curr_x = x0 + 1e-10;
                        continue;

                    x_to_x0_sgn = np.asarray((curr_x - x0) >= 0, dtype=float);
                    x_to_x0_sgn[x_to_x0_sgn <= 0] = -1;

                    curr_x = curr_x - lr * (x_to_x0_sgn + lamda * clust_loss_grads);
                    x_to_x0_loss = np.sum(np.fabs(curr_x - x0));
                    clust_loss = np.sum(np.fabs(self.saucie.sess.run(clust_loss_tensor, feed_dict={self.X: curr_x})));
                    if (iter+1) % 50 == 0: logger.info('iter={}\tx_to_x0_loss={}\tclust_loss={}'.format(iter, x_to_x0_loss, clust_loss, ));

                clustid_pair_relevance_map[(clust_c_id, clust_k_id)] = (curr_x - x0);
        return clustid_pair_relevance_map;

    def relevance_preparation(self, debug_mode=True):
        """neuralizing the cluster results"""
        clust_logits = [];
        for clust_c_id in self.uniq_clust_ids: clust_logits.append(self.neuralize_by_clustid(clust_c_id, debug_mode=debug_mode));
        if debug_mode: logger.info('clust_logits={}'.format(clust_logits));

        """sanity check of neuralization"""
        clustid_pos_sample_profiles_map = {};
        for clust_c_id in self.uniq_clust_ids:
            pos_sample_c_profiles = self.get_samples_with_positive_clust_logits(clust_c_id, clust_logits, debug_mode=debug_mode)
            clustid_pos_sample_profiles_map[clust_c_id] = pos_sample_c_profiles;
        return clustid_pos_sample_profiles_map;

    def neuralize_by_clustid(self, clust_c_id, debug_mode=False):
        sample_c_indices, sample_c_profiles, clust_c_centroid = self.get_clust_profiles(clust_c_id);

        new_clust_weight = np.zeros((self.saucie.get_embed_dim(), self.clust_num - 1));
        new_clust_bias = np.zeros(self.clust_num - 1);

        clust_idx = 0;
        for clust_k_id in self.uniq_clust_ids:
            if clust_k_id == clust_c_id: continue;
            _, _, clust_k_centroid = self.get_clust_profiles(clust_k_id);

            new_clust_weight[:, clust_idx] = 2*(clust_c_centroid - clust_k_centroid);
            new_clust_bias[clust_idx] = np.sum(np.square(clust_k_centroid)) - np.sum(np.square(clust_c_centroid));

            clust_idx += 1;

        if debug_mode: logger.info('clust_c_id={}\tnew_clust_weight={}\tnew_clust_bias={}'.format(clust_c_id, new_clust_weight, new_clust_bias));

        clust_weight, clust_bias, clust_linearAct_tensor, clust_minpoolAct_tensor = self.get_tensor_by_clustid(clust_c_id);

        """overwrite the clust-related tensor by the centroid information"""
        self.saucie.set_clust_tensors(clust_weight, clust_bias, new_clust_weight, new_clust_bias, debug_mode=debug_mode);

        return clust_minpoolAct_tensor;

    def get_samples_with_positive_clust_logits(self, clust_c_id, clust_logits, debug_mode=False):
        sample_c_indices, sample_c_profiles, clust_c_centroid = self.get_clust_profiles(clust_c_id);

        logger.info('--------------------------------------------------------------------');
        logger.info('neuralization sanity check:\tclust_c_id={}\tsample_c_profiles={}'.format(clust_c_id, sample_c_profiles.shape));

        # for clust_logits_arr in self.saucie.sess.run(clust_logits, feed_dict={self.X: sample_c_profiles}):
        #     if debug_mode: logger.info('clust_logits_arr={}'.format(clust_logits_arr));
        #     if np.any([np.isinf(x) for x in clust_logits_arr]): return False;

        clust_logits_arr_list = self.saucie.sess.run(clust_logits, feed_dict={self.X: sample_c_profiles});
        clust_logits_arr = clust_logits_arr_list[clust_c_id - 1];

        pos_indices = np.where(clust_logits_arr > 0)[0];
        neg_indices = np.where(clust_logits_arr < 0)[0];
        inf_indices = np.where(np.isinf(clust_logits_arr))[0];

        pos_indices = np.setdiff1d(pos_indices, inf_indices);
        neg_indices = np.setdiff1d(neg_indices, inf_indices);
        if len(pos_indices) <= 1: RuntimeError('No positive clust logits! Try a smaller beta value! ');
        pos_sample_c_profiles = sample_c_profiles[pos_indices];

        if debug_mode:
            logger.info('clust_c_id={}\tclust_logits_arr={}\tpos_sample_c_profiles={}'.format(clust_c_id, clust_logits_arr.shape, pos_sample_c_profiles.shape ));
            logger.info('pos_indices={}\tneg_indices={}\tinf_indices={}\tpos_mean={}'.format(pos_indices.shape, neg_indices.shape, inf_indices.shape, np.mean(clust_logits_arr[pos_indices]) ));

        return pos_sample_c_profiles;


    def get_clust_profiles(self, clust_id):
        sample_indices = np.where(self.clust_ids == clust_id)[0];
        sample_profiles = self.cell_profiles[sample_indices, :];
        clust_centroid = np.mean(self.cell_embeddings[sample_indices, :], axis=0);
        logger.info('clust_id={}\tsample_indices={}\tsample_profiles={}\tclust_centroid={}'.format(clust_id, sample_indices.shape, sample_profiles.shape, clust_centroid.shape));
        return sample_indices, sample_profiles, clust_centroid;

    def get_tensor_by_clustid(self, clust_id):
        clust_weight = [x for x in self.clust_weights if 'clust{}'.format(clust_id) in x.name];
        clust_bias = [x for x in self.clust_biases if 'clust{}'.format(clust_id) in x.name];
        clust_linearAct_tensor = [x for x in self.clust_activations if 'clust{}'.format(clust_id) in x.name and 'linears' in x.name ];
        clust_minpoolAct_tensor = [x for x in self.clust_activations if 'clust{}'.format(clust_id) in x.name and 'minpools' in x.name ];

        logger.info('clust_weight={}'.format(clust_weight));
        logger.info('clust_bias={}'.format(clust_bias));
        logger.info('clust_linearAct_tensor={}'.format(clust_linearAct_tensor));
        logger.info('clust_minpoolAct_tensor={}'.format(clust_minpoolAct_tensor));
        assert len(clust_weight) == 1;
        assert len(clust_bias) == 1;
        assert len(clust_linearAct_tensor) == 1;
        assert len(clust_minpoolAct_tensor) == 1;

        clust_weight = clust_weight[0];
        clust_bias = clust_bias[0];
        clust_linearAct_tensor = clust_linearAct_tensor[0];
        clust_minpoolAct_tensor = clust_minpoolAct_tensor[0];

        logger.info('clust_weight={}\n\n clust_bias={}\n\n clust_linearAct_tensor={}\n\n clust_minpoolAct_tensor={}\n\n'.format(clust_weight, clust_bias, clust_linearAct_tensor, clust_minpoolAct_tensor));
        return clust_weight, clust_bias, clust_linearAct_tensor, clust_minpoolAct_tensor;


def get_model_description(args):
    return '{}_beta{}_margin{}_fd{}_latent{}'.format(args.label, args.beta, args.margin, args.feat_dim, args.latent_dims );

def main(args):
    model_dir = os.path.join(args.output_dir, 'models_{}'.format(get_model_description(args)));
    if not os.path.exists(model_dir): os.makedirs(model_dir);
    output_dir = os.path.join(args.output_dir, 'results_{}'.format(get_model_description(args)));
    if not os.path.exists(output_dir): os.makedirs(output_dir);

    latent_dim_list = [int(x) for x in args.latent_dims.split(',')];
    logger.info('latent_dim_list={}'.format(latent_dim_list)); assert len(latent_dim_list) > 0;

    use_umap = False;
    cell_profiles, cell_ids, gene_ids, clust_ids = get_data(args);
    visualize_x_or_embedding(cell_profiles, clust_ids, os.path.join(output_dir, 'x_origin_umap{}.pdf'.format(int(use_umap))), use_umap=use_umap);

    tf.reset_default_graph();
    loadtrain = Loader(cell_profiles, labels=clust_ids, shuffle=False);

    if os.path.isfile(os.path.join(model_dir, 'SAUCIE.meta')) and args.overwrite <= 0:
        logger.info('Loading pretrained model at {}'.format(model_dir));
        saucie = SAUCIE(input_dim=cell_profiles.shape[1], feat_dim=args.feat_dim, clust_num=len(np.unique(clust_ids)), beta=args.beta, layers=latent_dim_list, restore_folder=model_dir);
    else:
        saucie = SAUCIE(input_dim=cell_profiles.shape[1], feat_dim=args.feat_dim, clust_num=len(np.unique(clust_ids)), beta=args.beta, layers=latent_dim_list, );
        saucie.train(loadtrain, steps=5000);
        saucie.save(folder=model_dir);

    # """LRP explanation"""
    cell_embeddings = saucie.get_embedding(loadtrain)[0]; logger.info('cell_embeddings={}'.format(cell_embeddings.shape));
    visualize_x_or_embedding(cell_embeddings, clust_ids, os.path.join(output_dir, 'x_embedding_umap{}.pdf'.format(int(use_umap))), use_umap=use_umap);

    lrp = LRP(clust_ids, cell_profiles, cell_embeddings, saucie);


    tag = 'cw_onevsrest_lamda{}_iter{}_lr{}'.format(args.lamda, args.max_iter, args.lr );
    relevance_url = os.path.join(output_dir, '{}.npy'.format(tag));
    if os.path.isfile(relevance_url): clustid_relevance_map = np.load(relevance_url, allow_pickle=True).item();
    else:
        clustid_relevance_map = lrp.calc_carlini_wagner_one_vs_rest(lamda=args.lamda, max_iter=args.max_iter, lr=args.lr, margin=args.margin );
        np.save(relevance_url, clustid_relevance_map);
    output_relevance_one_vs_rest(output_dir, clustid_relevance_map, gene_ids, tag=tag);


    tag = 'smoothgrad_onevsrest';
    relevance_url = os.path.join(output_dir, '{}.npy'.format(tag));
    if os.path.isfile(relevance_url): clustid_relevance_map = np.load(relevance_url, allow_pickle=True).item();
    else:
        clustid_relevance_map = lrp.calc_saliency(saliency_type=1);
        np.save(relevance_url, clustid_relevance_map);
    output_relevance_one_vs_rest(output_dir, clustid_relevance_map, gene_ids, tag=tag);

    tag = 'shap_onevsrest';
    relevance_url = os.path.join(output_dir, '{}.npy'.format(tag));
    if os.path.isfile(relevance_url): clustid_relevance_map = np.load(relevance_url, allow_pickle=True).item();
    else:
        clustid_relevance_map = lrp.calc_saliency(saliency_type=3);
        np.save(relevance_url, clustid_relevance_map);
    output_relevance_one_vs_rest(output_dir, clustid_relevance_map, gene_ids, tag=tag);




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description');
    parser.add_argument('--input_dir', type=str, help='input_dir');
    parser.add_argument('--output_dir', type=str, help='output_dir');
    parser.add_argument('--beta', type=float, help='beta', default=1.0);

    parser.add_argument('--lamda', type=float, help='lamda', default=1e2);
    parser.add_argument('--lr', type=float, help='lr', default=1e-3);
    parser.add_argument('--max_iter', type=int, help='max_iter', default=5000);

    parser.add_argument('--margin', type=float, help='margin', default=0);
    parser.add_argument('--latent_dims', type=str, help='latent_dims', default='512,256,128,2');

    parser.add_argument('--feat_dim', type=int, help='feat_dim', default=-1);
    parser.add_argument('--label', type=str, help='label', default='');
    parser.add_argument('--overwrite', type=int, help='overwrite', default=0);

    args = parser.parse_args();
    main(args);


"""example command"""
# python explainer/pbmc_explainer.py --input_dir ../data/PBMC --output_dir ../data/PBMC/results_kmeans --label pbmc --feat_dim -1 --beta 0.004

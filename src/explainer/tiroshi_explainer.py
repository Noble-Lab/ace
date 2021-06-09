import os, argparse, random, time, csv;

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import numpy as np;
from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA;
from sklearn.metrics import pairwise_distances;

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
            relevance_arr = np.squeeze(np.mean(clustid_relevance_map[clust_id], axis=0));
            gene_relev_list = list(zip(relevance_arr, gene_ids));
            gene_relev_list.sort(key=lambda x: x[0]);

        f = open(os.path.join(output_dir, "genes_{}_clustid{}_abs{}.csv".format(tag, clust_id, int(use_abs) )), "w");
        f.write("Rank\tGene\tScore\n");
        for idx in range(len(gene_relev_list)):
            score, gene = gene_relev_list[idx];
            f.write("{}\t{}\t{}\n".format(idx+1, gene, score));
        f.close();

def get_data(args, take_log=True, normalize=False):
    logger.info('input_url={}'.format(args.input_url)); assert os.path.exists(args.input_url);
    save_url = args.input_url.replace('.txt', '_log{}_norm{}.npz'.format(int(take_log), int(normalize))); logger.info('save_url={}'.format(save_url));

    if os.path.isfile(save_url):
        data = np.load(save_url);
        gene_ids = data['gene_ids'];
        cell_ids = data['cell_ids'];
        clust_ids = data['clust_ids'];
        type_ids = data['type_ids'];
        cell_profiles = data['cell_profiles'];
    else:
        gene_ids = np.genfromtxt(args.input_url, delimiter='\t', skip_header=1, usecols=(0), dtype=None, encoding='ascii');
        gene_ids = np.asarray(gene_ids, dtype=str)[3:];

        cell_ids = np.genfromtxt(args.input_url, delimiter='\t', max_rows=1, dtype=None);
        cell_ids = np.asarray(cell_ids[1:], dtype=str)

        clust_ids = np.genfromtxt(args.input_url, delimiter='\t', max_rows=3, dtype=None);
        clust_ids = np.squeeze(np.asarray(clust_ids[2:], dtype=str));
        clust_ids = np.asarray(clust_ids[1:], dtype=int);

        type_ids = np.genfromtxt(args.input_url, delimiter='\t', max_rows=4, dtype=None);
        type_ids = np.squeeze(np.asarray(type_ids[3:], dtype=str));
        type_ids = np.asarray(type_ids[1:], dtype=int);

        cell_profiles = np.genfromtxt(args.input_url, delimiter='\t', skip_header=4, encoding='ascii');
        cell_profiles = np.asarray(cell_profiles[:, 1:], dtype=float);
        cell_profiles = cell_profiles.T;

        if take_log: cell_profiles = np.log(cell_profiles + 1.0);
        if normalize: cell_profiles = preprocessing.scale(cell_profiles);

        valid_cell_indices = np.where(clust_ids > 0)[0];
        logger.info('valid_cell_indices={}'.format(len(valid_cell_indices), ));
        cell_ids = cell_ids[valid_cell_indices];
        clust_ids = clust_ids[valid_cell_indices];
        type_ids = type_ids[valid_cell_indices];
        cell_profiles = cell_profiles[valid_cell_indices];

        np.savez(save_url, gene_ids=gene_ids, cell_ids=cell_ids, clust_ids=clust_ids, type_ids=type_ids, cell_profiles=cell_profiles);

    logger.info('gene_ids={}\t{}'.format(len(gene_ids), gene_ids, ));
    logger.info('cell_ids={}\t{}'.format(len(cell_ids), cell_ids));
    logger.info('clust_ids={}\tmin={}\tmax={}'.format(clust_ids.shape, np.min(clust_ids), np.max(clust_ids) ));
    logger.info('type_ids={}\t{}'.format(type_ids.shape, type_ids));
    logger.info('cell_profiles={}'.format(cell_profiles.shape, ));

    return cell_profiles, cell_ids, gene_ids, clust_ids, type_ids;

def summarize_cluster(x_mat, clust_ids):
    assert len(x_mat) == len(clust_ids);

    cluster_info_map = {};
    radius_indices_all = np.asarray([], dtype=int); centroid_mat = np.array([], dtype=float).reshape(0, x_mat.shape[1]);
    for clust_id in np.unique(clust_ids):
        sample_indices = np.where(clust_ids == clust_id)[0];
        x_mat_sub = x_mat[sample_indices];
        clust_centroid = np.mean(x_mat_sub, axis=0);
        dist = np.squeeze(pairwise_distances(clust_centroid.reshape(1, -1), x_mat_sub));

        ### only choose samples within the median distance
        dist_thres = np.percentile(dist, 50);

        radius_indices = sample_indices[np.where(dist <= dist_thres)];
        radius_indices_all = np.concatenate((radius_indices_all, radius_indices));
        centroid_mat = np.vstack([centroid_mat, clust_centroid]);
        cluster_info_map[clust_id] = (sample_indices, radius_indices, clust_centroid);
        logger.info('clust_id={}\tsample_indices={}\tradius_indices={}'.format(clust_id, len(sample_indices), len(radius_indices) ));

    cluster_info_map[-1] = (list(range(len(clust_ids))), np.unique(radius_indices_all), centroid_mat);
    logger.info('sample_indices_all={}\tradius_indices={}'.format(len(clust_ids), len(radius_indices_all) ));
    return cluster_info_map;

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
    sc = plt.scatter(x_mat_2d[:, 0], x_mat_2d[:, 1], s=5, c=clust_ids, cmap='Spectral', marker='.');
    # plt.show();

    ### add centroid info
    cluster_info_map = summarize_cluster(x_mat, clust_ids);
    _, _, centroid_mat = cluster_info_map[-1];
    centroid_2d = mapper.transform(centroid_mat);

    logger.info('centroid_mat={}\tcentroid_2d={}'.format(centroid_mat.shape, centroid_2d.shape));
    sc = plt.scatter(centroid_2d[:, 0], centroid_2d[:, 1], s=100, c='black', marker='v');

    pp = PdfPages(save_url);
    pp.savefig(figure, bbox_inches='tight');
    pp.close();
    plt.close(figure);
    del figure;

def get_model_description(args):
    return '{}_beta{}_margin{}_fd{}_latent{}'.format(args.label, args.beta, args.margin, args.feat_dim, args.latent_dims );

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

    def calc_lrp_one_vs_rest(self, debug_mode=True):
        """explain each cluster """
        clustid_relevance_map = {};
        for clust_c_id in self.uniq_clust_ids:
            clust_weight, clust_bias, clust_linearAct_tensor, clust_minpoolAct_tensor = self.get_tensor_by_clustid(clust_c_id);

            """back-propagate relevance top-down"""
            all_activations = [clust_minpoolAct_tensor, clust_linearAct_tensor];
            all_weights_biases = [(None, None), (clust_weight, clust_bias)];

            """The first activation is clust_minpool layer, the last activation is input layer"""
            for tmp_activation, tmp_weight, tmp_bias in zip(self.embed_activations[:0:-1], self.embed_weights[::-1], self.embed_biases[::-1]):
                all_activations.append(tmp_activation);
                all_weights_biases.append((tmp_weight, tmp_bias));
            all_activations.append(self.embed_activations[0]); all_weights_biases.append((None, None));

            """Attaching subgraph for calculating LRP"""
            all_relevances = [all_activations[0], ];
            for i in range(1, len(all_activations)):
                name = all_activations[i - 1].name;

                if 'clust' in name and 'minpool' in name:
                    all_relevances.append(self.backprop_clust_minpool(all_activations[i], self.saucie.beta, all_relevances[-1]));
                elif 'clust' in name and 'linear' in name:
                    all_relevances.append(self.backprop_clust_linear(all_activations[i], all_weights_biases[i-1][0], all_weights_biases[i-1][1], all_relevances[-1]));
                elif 'encoder' in name or 'embedding' in name:
                    all_relevances.append(self.backprop_dense(all_activations[i], all_weights_biases[i-1][0], all_weights_biases[i-1][1], all_relevances[-1]));
                else: assert False;
                logger.info('output_relevance={}'.format(all_relevances[-1]));

            relevance_scores = self.saucie.sess.run(all_relevances[-1], feed_dict={self.X: self.clustid_pos_sample_profiles_map[clust_c_id]});
            if debug_mode: logger.info('relevance_scores={}'.format(relevance_scores.shape));
            relevance_scores[np.isnan(relevance_scores)] = 0;

            if np.any([np.isinf(x) for x in relevance_scores]) or np.any([np.isnan(x) for x in relevance_scores]):
                raise RuntimeError('explanation failed by encountering Inf/Nan values! Try a smaller beta value! ');

            clustid_relevance_map[clust_c_id] = relevance_scores;
        return clustid_relevance_map;

    def backprop_dense(self, activation, kernel, bias, relevance, gamma=10.0):
        logger.info('backprop_dense\tactivation={}\tkernel={}\tbias={}\trelevance={}'.format(activation, kernel, bias, relevance));
        kernel_pos = tf.maximum(0.0, kernel)*tf.constant(gamma) + kernel;
        bias_pos = tf.maximum(0., bias)*tf.constant(gamma) + bias;
        z = tf.matmul(activation, kernel_pos) + bias_pos + tf.constant(1e-10);
        s = relevance / z;
        c = tf.matmul(s, tf.transpose(kernel_pos));
        return c * activation;

    def backprop_clust_linear(self, activation, kernel, bias, relevance):
        logger.info('backprop_clust_linear\tactivation={}\tkernel={}\tbias={}\trelevance={}'.format(activation, kernel, bias, relevance));
        z = tf.matmul(activation, kernel) + bias + tf.constant(1e-10);
        s = relevance / z;
        c = tf.matmul(s, tf.transpose(kernel));
        return c * activation;

    def backprop_clust_minpool(self, activation, beta, relevance):
        logger.info('backprop_clust_minpool\tactivation={}\tbeta={}\trelevance={}'.format(activation, beta, relevance));
        s = tf.math.exp(tf.constant(-beta) * activation) / tf.reduce_sum(tf.math.exp(tf.constant(-beta) * activation));
        return tf.expand_dims(relevance, axis=1) * s;

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



def main(args):
    model_dir = os.path.join(args.output_dir, 'models_{}'.format(get_model_description(args)));
    if not os.path.exists(model_dir): os.makedirs(model_dir);
    output_dir = os.path.join(args.output_dir, 'results_{}'.format(get_model_description(args)));
    if not os.path.exists(output_dir): os.makedirs(output_dir);

    latent_dim_list = [int(x) for x in args.latent_dims.split(',')];
    logger.info('latent_dim_list={}'.format(latent_dim_list)); assert len(latent_dim_list) > 0;

    ### Load the data
    take_log=True; normalize=False; use_umap=False;
    cell_profiles, cell_ids, gene_ids, clust_ids, type_ids = get_data(args, take_log=take_log, normalize=normalize);

    ### visualize the data
    viz_url = os.path.join(args.output_dir, 'x_origin_log{}_norm{}_umap{}.pdf'.format(int(take_log), int(normalize), int(use_umap)));
    visualize_x_or_embedding(cell_profiles, clust_ids, viz_url, use_umap=use_umap);

    ### train or load SAUCIE model
    tf.reset_default_graph();
    loadtrain = Loader(cell_profiles, labels=clust_ids, shuffle=False);

    if os.path.isfile(os.path.join(model_dir, 'SAUCIE.meta')) and args.overwrite <= 0:
        logger.info('Loading pretrained model at {}'.format(model_dir));
        saucie = SAUCIE(input_dim=cell_profiles.shape[1], feat_dim=args.feat_dim, clust_num=len(np.unique(clust_ids)), beta=args.beta, layers=latent_dim_list, restore_folder=model_dir);
    else:
        saucie = SAUCIE(input_dim=cell_profiles.shape[1], feat_dim=args.feat_dim, clust_num=len(np.unique(clust_ids)), beta=args.beta, layers=latent_dim_list, );
        saucie.train(loadtrain, steps=10000);
        saucie.save(folder=model_dir);

    ### get embedding
    cell_embeddings = saucie.get_embedding(loadtrain)[0]; logger.info('cell_embeddings={}'.format(cell_embeddings.shape));
    viz_url = os.path.join(output_dir, 'x_embedding_log{}_norm{}_umap{}.pdf'.format(int(take_log), int(normalize), int(use_umap)));
    visualize_x_or_embedding(cell_embeddings, clust_ids, viz_url, use_umap=use_umap);

    ### get embedding for only reliable samples (within each cluster radius)
    cluster_info_map = summarize_cluster(cell_embeddings, clust_ids);
    _, radius_indices, _ = cluster_info_map[-1];
    viz_url = os.path.join(output_dir, 'x_embedding_radius_log{}_norm{}_umap{}.pdf'.format(int(take_log), int(normalize), int(use_umap)));
    visualize_x_or_embedding(cell_embeddings[radius_indices], clust_ids[radius_indices], viz_url, use_umap=use_umap);


    lrp = LRP(clust_ids[radius_indices], cell_profiles[radius_indices], cell_embeddings[radius_indices], saucie);

    tag = 'cw_onevsrest_lamda{}_iter{}_lr{}'.format(args.lamda, args.max_iter, args.lr );
    relevance_url = os.path.join(output_dir, '{}.npy'.format(tag));
    if os.path.isfile(relevance_url): clustid_relevance_map = np.load(relevance_url, allow_pickle=True).item();
    else:
        clustid_relevance_map = lrp.calc_carlini_wagner_one_vs_rest(lamda=args.lamda, max_iter=args.max_iter, lr=args.lr, margin=args.margin );
        np.save(relevance_url, clustid_relevance_map);
    output_relevance_one_vs_rest(output_dir, clustid_relevance_map, gene_ids, tag=tag);
    #
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

    tag = 'lrp_onevsrest';
    relevance_url = os.path.join(output_dir, '{}.npy'.format(tag));
    if os.path.isfile(relevance_url): clustid_relevance_map = np.load(relevance_url, allow_pickle=True).item();
    else:
        clustid_relevance_map = lrp.calc_lrp_one_vs_rest();
        np.save(relevance_url, clustid_relevance_map);
    output_relevance_one_vs_rest(output_dir, clustid_relevance_map, gene_ids, tag=tag);


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description');
    parser.add_argument('--input_url', type=str, help='input_url');
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
# python explainer/tiroshi_explainer.py --input_url ../data/tiroshi/GSE72056_melanoma_single_cell_revised_v2.txt --output_dir ../data/tiroshi/results_kmeans --label tiroshi --feat_dim -1 --beta 0.1
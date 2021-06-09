import os, sys, argparse, random, time;

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import numpy as np;
from sklearn import preprocessing
import tensorflow as tf; # tim: using tensorflow==1.3.1

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

import scprep
from SAUCIE.model import SAUCIE
from SAUCIE.loader import Loader


def plot_relevance(output_fig_url, relevance_mat):
    figure = plt.figure(figsize=(12, 12));
    ax = figure.add_subplot(1, 1, 1);
    sns.heatmap(relevance_mat, linewidth=0.5);
    pp = PdfPages(output_fig_url);
    pp.savefig(figure, bbox_inches='tight');
    pp.close();
    plt.close(figure);
    del figure;


def get_data(args, normalize=True):

    logger.info('input_url={}'.format(args.input_url)); assert os.path.isfile(args.input_url);

    cell_ids = np.genfromtxt(args.input_url, delimiter=',', skip_header=1, usecols=(0), dtype=None, encoding='ascii');
    cell_ids = np.asarray(cell_ids, dtype=str)
    logger.info('cell_ids={}'.format(len(cell_ids)));

    gene_ids = np.genfromtxt(args.input_url, delimiter=',', max_rows=1, dtype=None);
    gene_ids = np.asarray(gene_ids[1:], dtype=str)
    logger.info('gene_ids={}'.format(len(gene_ids)));

    cell_profiles = np.genfromtxt(args.input_url, delimiter=',', skip_header=1, encoding='ascii');
    cell_profiles = np.asarray(cell_profiles[:, 1:], dtype=float);
    cell_profiles = np.log(cell_profiles + 1.0);
    if normalize: cell_profiles = preprocessing.scale(cell_profiles);
    logger.info('cell_profiles={}'.format(cell_profiles.shape));

    clust_ids = np.asarray([int(id.replace('"', '').split('_')[1]) for id in cell_ids], dtype=int);
    logger.info('clust_ids={}'.format(len(clust_ids)));

    return cell_profiles, cell_ids, gene_ids, clust_ids;


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

        self.relevance_preparation();


    def calc_saliency(self, saliency_type=0 ):
        clustid_relevance_map = {};
        for clust_c_id in self.uniq_clust_ids:
            _, _, _, clust_c_minpoolAct_tensor = self.get_tensor_by_clustid(clust_c_id);
            clust_grads_tensor = tf.gradients(clust_c_minpoolAct_tensor, self.X)[0];

            if saliency_type == 0:
                """vanilla gradient"""
                clust_grads = self.saucie.sess.run(clust_grads_tensor, feed_dict={self.X: self.cell_profiles});
            elif saliency_type == 1:
                """Smooth Gradient"""
                clust_grads = np.zeros_like(self.cell_profiles); repeat=20; stdev = np.std(self.cell_profiles) * 0.2;
                for iter in range(repeat):
                    noise = np.random.normal(0, stdev, self.cell_profiles.shape);
                    clust_grads += self.saucie.sess.run(clust_grads_tensor, feed_dict={self.X: self.cell_profiles + noise});
            elif saliency_type == 2:
                """Integrated Gradients"""
                clust_grads = np.zeros_like(self.cell_profiles); baseline = np.zeros_like(self.cell_profiles); steps=20;
                for alpha in np.linspace(0, 1, steps):
                    clust_grads += self.saucie.sess.run(clust_grads_tensor, feed_dict={self.X: alpha * self.cell_profiles});

            elif saliency_type == 3:
                """SHAP: conda install -c conda-forge shap """
                import shap;
                e = shap.GradientExplainer((self.X, clust_c_minpoolAct_tensor), self.cell_profiles, session=self.saucie.sess, local_smoothing=0);
                clust_grads = e.shap_values(self.cell_profiles);

            else: assert False;
            clustid_relevance_map[clust_c_id] = clust_grads;
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
            x0 = np.copy(self.cell_profiles); curr_x = x0 + 1e-6;
            for iter in range(max_iter):
                clust_loss_grads = self.saucie.sess.run(clust_loss_grads_tensor, feed_dict={self.X: curr_x});
                if np.any([np.isinf(x) for x in clust_loss_grads]) or np.any([np.isnan(x) for x in clust_loss_grads]):
                    raise RuntimeError('explanation failed by encountering Inf/Nan values! Try a smaller beta value! ');

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
                x0 = np.copy(self.cell_profiles); curr_x = x0 + 1e-6;
                for iter in range(max_iter):
                    clust_loss_grads = self.saucie.sess.run(clust_loss_grads_tensor, feed_dict={self.X: curr_x});
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
        for clust_c_id in self.uniq_clust_ids:
            if not self.sanity_check_neuralization_by_clustid(clust_c_id, clust_logits, debug_mode=debug_mode):
                raise RuntimeError('Neuralization failed by encountering Inf clust logits! Try a smaller beta value! ');

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

    def sanity_check_neuralization_by_clustid(self, clust_c_id, clust_logits, debug_mode=False):
        sample_c_indices, sample_c_profiles, clust_c_centroid = self.get_clust_profiles(clust_c_id);

        logger.info('--------------------------------------------------------------------');
        logger.info('neuralization sanity check:\tclust_c_id={}\tsample_c_profiles={}'.format(clust_c_id, sample_c_profiles.shape));

        for clust_logits_arr in self.saucie.sess.run(clust_logits, feed_dict={self.X: sample_c_profiles}):
            if debug_mode: logger.info('clust_logits_arr={}'.format(clust_logits_arr));
            if np.any([np.isinf(x) for x in clust_logits_arr]): return False;
        return True;


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
    return '{}_louvain_beta{}_margin{}_fd{}_latent{}'.format(args.label, args.beta, args.margin, args.feat_dim, args.latent_dim );

def main(args):
    model_dir = os.path.join(args.output_dir, 'models_{}'.format(get_model_description(args)));
    if not os.path.exists(model_dir): os.makedirs(model_dir);
    output_dir = os.path.join(args.output_dir, 'figs_{}'.format(get_model_description(args)));
    if not os.path.exists(output_dir): os.makedirs(output_dir);

    cell_profiles, cell_ids, gene_ids, clust_ids = get_data(args, normalize=True);

    tf.reset_default_graph();
    loadtrain = Loader(cell_profiles, labels=clust_ids, shuffle=False);

    if os.path.isfile(os.path.join(model_dir, 'SAUCIE.meta')) and args.overwrite <= 0:
        logger.info('Loading pretrained model at {}'.format(model_dir));
        saucie = SAUCIE(input_dim=cell_profiles.shape[1], feat_dim=args.feat_dim, clust_num=len(np.unique(clust_ids)), beta=args.beta, layers=[512,256,128,args.latent_dim], restore_folder=model_dir);
    else:
        saucie = SAUCIE(input_dim=cell_profiles.shape[1], feat_dim=args.feat_dim, clust_num=len(np.unique(clust_ids)), beta=args.beta, layers=[512,256,128,args.latent_dim], );
        saucie.train(loadtrain, steps=1000);
        saucie.save(folder=model_dir);

    """LRP explanation"""
    cell_embeddings = saucie.get_embedding(loadtrain)[0]; logger.info('cell_embeddings={}'.format(cell_embeddings.shape));
    if args.plot_embedding > 0 and args.latent_dim == 2:
        scprep.plot.scatter2d(cell_embeddings, c=clust_ids, ticks=False, label_prefix="SAUCIE", legend_title="Group", legend_anchor=(1, 1));
        plt.show();



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description');
    parser.add_argument('--input_url', type=str, help='input_url');
    parser.add_argument('--output_dir', type=str, help='output_dir');
    parser.add_argument('--beta', type=float, help='beta', default=1.0);

    parser.add_argument('--lamda', type=float, help='lamda', default=1e2);
    parser.add_argument('--lr', type=float, help='lr', default=1e-3);
    parser.add_argument('--max_iter', type=int, help='max_iter', default=3000);

    parser.add_argument('--margin', type=float, help='margin', default=0);
    parser.add_argument('--latent_dim', type=int, help='latent_dim', default=2);

    parser.add_argument('--feat_dim', type=int, help='feat_dim', default=-1);
    parser.add_argument('--label', type=str, help='label', default='');
    parser.add_argument('--plot_embedding', type=int, help='plot_embedding', default=0);
    parser.add_argument('--overwrite', type=int, help='overwrite', default=0);

    args = parser.parse_args();
    main(args);


"""example command"""
"""reference to carlini_wagner and deepfool: Adversarial Attacks and Defenses in Images, Graphs and Text: A Review"""
# python explainer/saucie_louvain_explainer.py --input_url ../data/130var_clean.csv --output_dir ../data/ --label 130var_clean --feat_dim -1 --beta 0.01 --plot_embedding 0
# python explainer/saucie_louvain_explainer.py --input_url ../data/130var_clean.csv --output_dir ../data/ --label 130var_clean --feat_dim 25 --beta 0.04 --plot_embedding 0

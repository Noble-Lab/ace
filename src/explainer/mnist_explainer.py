import os, sys, argparse, random, time;

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import numpy as np;
from sklearn.decomposition import PCA;
from sklearn.manifold import TSNE;
from sklearn.preprocessing import MinMaxScaler;
import tensorflow as tf;
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from PIL import Image
except ImportError:
    import Image

import utils.io_utils as io_utils;
import utils.viz_utils as viz_utils;

from mnist.loader import Loader
from mnist.model import MNIST


def plot_embedding(embeddings, labels, use_tsne=True):
    assert len(embeddings) == len(labels);

    if use_tsne:
        embeddings_2d = TSNE(n_components=2, init='pca').fit_transform(embeddings);
    else:
        pca = PCA(n_components=2);
        pca.fit(embeddings);
        embeddings_2d = pca.transform(embeddings);

    figure = plt.figure(figsize=(12, 12));
    ax = figure.add_subplot(1, 1, 1);

    for curr_label in np.unique(labels):
        curr_indices = np.where(labels == curr_label)[0];
        logger.info('curr_label={}\tcurr_indices={}'.format(curr_label, len(curr_indices) ));
        ax.plot(embeddings_2d[curr_indices, 0], embeddings_2d[curr_indices, 1], 's', marker='o', alpha=0.7, label="label={}".format(curr_label), );

    ax.legend(prop={'size': 20});
    plt.show();


def plot_sampled_importance(tag, labels, data, scores, result_dir, sample_size=100):
    assert len(labels) == len(data) and data.shape == scores.shape;
    data_sub = data[:sample_size];
    scores_sub = scores[:sample_size];
    logger.info('tag={}\tdata_sub={}\tscores_sub={}'.format(tag, data_sub.shape, scores_sub.shape, ));
    logger.info('data_sub\tmax={}\tmin={}'.format(np.max(data_sub), np.min(data_sub), )),
    logger.info('scores_sub\tmax={}\tmin={}'.format(np.max(scores_sub), np.min(scores_sub), )),

    orig_img_url = os.path.join(result_dir, 'fig_orig_{}.png'.format(tag));
    rel_img_url = os.path.join(result_dir, 'fig_rel_{}.png'.format(tag));
    blend_img_url = os.path.join(result_dir, 'fig_blend_{}.png'.format(tag));

    viz_utils.save_image(viz_utils.create_score_sprite(data_sub, normalize=False), orig_img_url, cmap='binary');
    viz_utils.save_image(viz_utils.create_score_sprite(scores_sub, normalize=False), rel_img_url, cmap='bwr');

    """superimpose two images together"""
    background = Image.open(orig_img_url);
    overlay = Image.open(rel_img_url);
    background = background.convert("RGBA");
    overlay = overlay.convert("RGBA");
    new_img = Image.blend(background, overlay, 0.7);
    new_img.save(blend_img_url, "PNG");

    os.remove(orig_img_url);
    os.remove(rel_img_url);


def compare_importance(args, result_dir):
    fig_dir = os.path.join(result_dir, 'figs');
    if not os.path.exists(fig_dir): os.makedirs(fig_dir);

    uniq_labels = list(range(10));
    for clust_c_id in uniq_labels:
        logger.info('processing clust_c_id={}'.format(clust_c_id));

        tag_list = [];
        tag_list.append('{}_label_{}'.format('saliency_smoothgrad', clust_c_id, ));
        tag_list.append('{}_label_{}'.format('shap', clust_c_id, ));
        for clust_k_id in uniq_labels:
            if clust_c_id == clust_k_id: continue;
            tag = 'cw_margin{}_lamda{}_iter{}_lr{}'.format(args.margin, args.lamda, args.max_iter, args.lr);
            tag_list.append('{}_label_from{}_to{}'.format(tag, clust_c_id, clust_k_id, ));

        for tag in tag_list:
            relevance_url = os.path.join(result_dir, '{}.npz'.format(tag, ));
            assert os.path.isfile(relevance_url);

            relevance_data = np.load(relevance_url);
            label_sub = relevance_data['labels'];
            data_sub = relevance_data['data'];
            scores_sub = relevance_data['scores'];
            plot_sampled_importance(tag, label_sub, data_sub, scores_sub, fig_dir);


def compare_logodds(data_mat, score_mat, mnist, clust_c_id, clust_k_id, output_dir):
    logger.info('data_mat={}\tmin={}\tmax={}'.format(data_mat.shape, np.min(data_mat), np.max(data_mat) ));
    logger.info('score_mat={}\tmin={}\tmax={}'.format(score_mat.shape, np.min(score_mat), np.max(score_mat) ));

    predict_labels = mnist.predict_logits(data_mat);
    # # logger.info('clust_c_id={}\tpredicted_logits={}'.format(clust_c_id, predict_labels[:, clust_c_id] ));
    # # logger.info('clust_k_id={}\tpredicted_logits={}'.format(clust_k_id, predict_labels[:, clust_k_id] ));
    #
    attacked_data_mat = data_mat + score_mat;
    # logger.info('attacked_data_mat={}\tmin={}\tmax={}'.format(attacked_data_mat.shape, np.min(attacked_data_mat), np.max(attacked_data_mat) ));
    #
    attacked_labels = mnist.predict_logits(attacked_data_mat);
    # # logger.info('clust_c_id={}\tpredicted_logits={}'.format(clust_c_id, attacked_labels[:, clust_c_id] ));
    # # logger.info('clust_k_id={}\tattacked_logits={}'.format(clust_k_id, attacked_labels[:, clust_k_id] ));
    #
    attacked_logodds = np.log(attacked_labels[:, clust_k_id]) - np.log(predict_labels[:, clust_k_id]);
    logger.info('clust_k_id={}\tattacked_logodds={}'.format(clust_k_id, attacked_logodds));

    logodds_all = []; method_all = [];
    logodds_all = np.concatenate((logodds_all, attacked_logodds));
    method_all = np.concatenate((method_all, ['ACE']*len(attacked_logodds) ));

    ### calculate saliency and shap
    for saliency_type, saliency_name in [(1, 'SmoothGrad'), (3, 'SHAP')]:
        logger.info('saliency_type={}'.format(saliency_type));

        grads_c = mnist.calc_saliency(data_mat, clust_id=clust_c_id, saliency_type=1);
        grads_k = mnist.calc_saliency(data_mat, clust_id=clust_k_id, saliency_type=1);
        grads_ensemble = grads_k - grads_c;

        grads_ensemble = (grads_ensemble - grads_ensemble.min()) / (grads_ensemble.max() - grads_ensemble.min()) * 2 -1;
        logger.info('grads_ensemble={}\tmin={}\tmax={}'.format(grads_ensemble.shape, np.min(grads_ensemble), np.max(grads_ensemble) ));

        saliency_data_mat = data_mat + grads_ensemble;
        saliency_data_mat = (saliency_data_mat - saliency_data_mat.min()) / (saliency_data_mat.max() - saliency_data_mat.min());
        logger.info('saliency_data_mat={}\tmin={}\tmax={}'.format(saliency_data_mat.shape, np.min(saliency_data_mat), np.max(saliency_data_mat) ));

        saliency_labels = mnist.predict_logits(saliency_data_mat);
        saliency_logodds = np.log(saliency_labels[:, clust_k_id]) - np.log(predict_labels[:, clust_k_id]);
        logger.info('clust_k_id={}\tsaliency_logodds={}'.format(clust_k_id, saliency_logodds));

        logodds_all = np.concatenate((logodds_all, saliency_logodds));
        method_all = np.concatenate((method_all, [saliency_name] * len(attacked_logodds)));

    data_pd = pd.DataFrame({'Logodds': logodds_all, 'Method': method_all});
    figure = plt.figure(figsize=(12, 8));
    ax = sns.boxplot(x='Method', y='Logodds', data=data_pd, whis=2,
                     order=['ACE', 'SmoothGrad', 'SHAP', ], );
    ax.legend(prop={'size': 30}, bbox_to_anchor=(1.05, 1), loc='upper left', );
    ax.set_ylabel('Change in log-odds', fontsize=35);
    ax.set_xlabel('');
    ax.tick_params(axis='both', which='major', labelsize=30,);

    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages(os.path.join(output_dir, 'fig_logodds.pdf'));
    pp.savefig(figure, bbox_inches='tight');
    pp.close()


class LRP:
    def __init__(self, labels, x_data, x_embeddings, mnist):
        self.labels = labels;
        self.uniq_labels = np.unique(self.labels);
        self.clust_num = len(self.uniq_labels);
        assert self.clust_num == 10;

        self.x_data = x_data;
        self.x_embeddings = x_embeddings;
        assert len(self.labels) == len(self.x_data) and len(self.x_data) == len(self.x_embeddings);

        self.mnist = mnist;
        self.embed_weights, self.embed_biases, self.embed_activations = self.mnist.get_embedder_tensors();
        self.clust_weights, self.clust_biases, self.clust_activations = self.mnist.get_clust_tensors();
        self.X = self.embed_activations[0];

        ### classification-based attack
        self.clust_output = {};
        for clust_id in self.uniq_labels: self.clust_output[clust_id] = self.mnist.get_targeted_output_tensor([clust_id]);

        self.relevance_preparation();

    def relevance_preparation(self, debug_mode=True):
        """neuralizing the cluster results"""
        clust_logits = [];
        for clust_c_id in self.uniq_labels: clust_logits.append(self.neuralize_by_clustid(clust_c_id, debug_mode=debug_mode));
        if debug_mode: logger.info('clust_c_id={}\tclust_logits={}'.format(clust_c_id, clust_logits));

        """sanity check of neuralization"""
        for clust_c_id in self.uniq_labels:
            if not self.sanity_check_neuralization_by_clustid(clust_c_id, clust_logits, debug_mode=debug_mode):
                raise RuntimeError('Neuralization failed by encountering Inf clust logits! Try a smaller beta value! ');

    def neuralize_by_clustid(self, clust_c_id, debug_mode=False):
        sample_c_indices, sample_c_profiles, clust_c_centroid = self.get_clust_profiles(clust_c_id);

        new_clust_weight = np.zeros((self.mnist.get_embed_dim(), self.clust_num - 1));
        new_clust_bias = np.zeros(self.clust_num - 1);

        clust_idx = 0;
        for clust_k_id in self.uniq_labels:
            if clust_k_id == clust_c_id: continue;
            _, _, clust_k_centroid = self.get_clust_profiles(clust_k_id);

            new_clust_weight[:, clust_idx] = 2*(clust_c_centroid - clust_k_centroid);
            new_clust_bias[clust_idx] = np.sum(np.square(clust_k_centroid)) - np.sum(np.square(clust_c_centroid));

            clust_idx += 1;

        if debug_mode: logger.info('clust_c_id={}\tnew_clust_weight={}\tnew_clust_bias={}'.format(clust_c_id, new_clust_weight, new_clust_bias));

        clust_weight, clust_bias, clust_linearAct_tensor, clust_minpoolAct_tensor = self.get_tensor_by_clustid(clust_c_id);

        """overwrite the clust-related tensor by the centroid information"""
        self.mnist.set_clust_tensors(clust_weight, clust_bias, new_clust_weight, new_clust_bias, debug_mode=debug_mode);

        return clust_minpoolAct_tensor;

    def sanity_check_neuralization_by_clustid(self, clust_c_id, clust_logits, debug_mode=False):
        sample_c_indices, sample_c_profiles, clust_c_centroid = self.get_clust_profiles(clust_c_id);

        logger.info('--------------------------------------------------------------------');
        logger.info('neuralization sanity check:\tclust_c_id={}\tsample_c_profiles={}'.format(clust_c_id, sample_c_profiles.shape));

        for clust_logits_arr in self.mnist.sess.run(clust_logits, feed_dict={self.X: sample_c_profiles}):
            if debug_mode: logger.info('clust_logits_arr={}'.format(clust_logits_arr));
            if np.any([np.isinf(x) for x in clust_logits_arr]): return False;
        return True;

    def get_clust_profiles(self, clust_id):
        sample_indices = np.where(self.labels == clust_id)[0];
        sample_profiles = self.x_data[sample_indices];
        clust_centroid = np.mean(self.x_embeddings[sample_indices], axis=0);
        logger.info('clust_id={}\tsample_indices={}\tsample_profiles={}\tclust_centroid={}'.format(clust_id, sample_indices.shape, sample_profiles.shape, clust_centroid.shape));
        return sample_indices, sample_profiles, clust_centroid;

    def get_tensor_by_clustid(self, clust_id):
        clust_weight = [x for x in self.clust_weights if 'clust{}'.format(clust_id) in x.name];
        clust_bias = [x for x in self.clust_biases if 'clust{}'.format(clust_id) in x.name];
        clust_linearAct_tensor = [x for x in self.clust_activations if 'clust{}'.format(clust_id) in x.name and 'linear_op' in x.name ];
        clust_minpoolAct_tensor = [x for x in self.clust_activations if 'clust{}'.format(clust_id) in x.name and 'minpool_op' in x.name ];

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

    def calc_carlini_wagner(self, clust_c_id, clust_k_id, lamda=1e2, max_iter=5000, lr=2e-3, margin=0, min_val=0.0, max_val=1.0):
        assert clust_c_id in self.uniq_labels and clust_k_id in self.uniq_labels;
        assert clust_c_id != clust_k_id;

        sample_indices = np.where(self.labels == clust_c_id)[0];
        x_data_sub = self.x_data[sample_indices];
        labels_sub = self.labels[sample_indices];

        logger.info('clust_c_id={}\tclust_k_id={}'.format(clust_c_id, clust_k_id, ));
        _, _, _, clust_c_minpoolAct_tensor = self.get_tensor_by_clustid(clust_c_id);
        _, _, _, clust_k_minpoolAct_tensor = self.get_tensor_by_clustid(clust_k_id);

        clust_loss_tensor = tf.math.maximum(margin + clust_c_minpoolAct_tensor - clust_k_minpoolAct_tensor, 0);
        clust_loss_grads_tensor = tf.gradients(clust_loss_tensor, self.X)[0];

        """starting carlini_wagner optimization"""
        x0 = np.copy(x_data_sub); curr_x = x0 + 1e-10;
        for iter in range(max_iter):
            clust_loss_grads = self.mnist.sess.run(clust_loss_grads_tensor, feed_dict={self.X: curr_x});
            x_to_x0_sgn = np.asarray((curr_x - x0) >= 0, dtype=float);
            x_to_x0_sgn[x_to_x0_sgn <= 0] = -1;

            curr_x = curr_x - lr * (x_to_x0_sgn + lamda * clust_loss_grads);
            curr_x = np.maximum(curr_x, min_val);
            curr_x = np.minimum(curr_x, max_val);

            x_to_x0_loss = np.sum(np.fabs(curr_x - x0));
            clust_loss = np.sum(np.fabs(self.mnist.sess.run(clust_loss_tensor, feed_dict={self.X: curr_x})));
            if iter % 50 == 0: logger.info('iter={}\tx_to_x0_loss={}\tclust_loss={}'.format(iter, x_to_x0_loss, clust_loss, ));

        return curr_x - x0, x_data_sub, labels_sub;


    def calc_carlini_wagner_classification(self, clust_c_id, clust_k_id, lamda=1e2, max_iter=5000, lr=2e-3, margin=0, min_val=0.0, max_val=1.0):
        assert clust_c_id in self.uniq_labels and clust_k_id in self.uniq_labels;
        assert clust_c_id != clust_k_id;
        assert clust_c_id in self.clust_output and clust_k_id in self.clust_output;

        sample_indices = np.where(self.labels == clust_c_id)[0];
        x_data_sub = self.x_data[sample_indices];
        labels_sub = self.labels[sample_indices];

        logger.info('clust_c_id={}\tclust_k_id={}'.format(clust_c_id, clust_k_id, ));

        clust_c_output = self.clust_output[clust_c_id];
        clust_k_output = self.clust_output[clust_k_id];

        clust_loss_tensor = tf.math.maximum(margin + clust_c_output - clust_k_output, 0);
        clust_loss_grads_tensor = tf.gradients(clust_loss_tensor, self.X)[0];

        """starting carlini_wagner optimization"""
        x0 = np.copy(x_data_sub); curr_x = x0 + 1e-10;
        for iter in range(max_iter):
            clust_loss_grads = self.mnist.sess.run(clust_loss_grads_tensor, feed_dict={self.X: curr_x});
            x_to_x0_sgn = np.asarray((curr_x - x0) >= 0, dtype=float);
            x_to_x0_sgn[x_to_x0_sgn <= 0] = -1;

            curr_x = curr_x - lr * (x_to_x0_sgn + lamda * clust_loss_grads);
            curr_x = np.maximum(curr_x, min_val);
            curr_x = np.minimum(curr_x, max_val);

            x_to_x0_loss = np.sum(np.fabs(curr_x - x0));
            clust_loss = np.sum(np.fabs(self.mnist.sess.run(clust_loss_tensor, feed_dict={self.X: curr_x})));
            if iter % 50 == 0: logger.info('iter={}\tx_to_x0_loss={}\tclust_loss={}'.format(iter, x_to_x0_loss, clust_loss, ));

        return curr_x - x0, x_data_sub, labels_sub;


def get_model_description(args):
    return 'cnn{}_beta{}'.format(args.use_cnn, args.beta, );

def main(args):
    model_dir = os.path.join(args.output_dir, 'models_{}'.format(get_model_description(args)));
    if not os.path.exists(model_dir): os.makedirs(model_dir);
    result_dir = os.path.join(args.output_dir, 'results_{}'.format(get_model_description(args)));
    if not os.path.exists(result_dir): os.makedirs(result_dir);

    (x_train, y_train), (x_test, y_test), _, _ = io_utils.load_mnist(args.output_dir);
    ### subsampling the test data
    x_test = x_test[:1000]; y_test = y_test[:1000];

    label_train = np.argmax(y_train, axis=1); label_test = np.argmax(y_test, axis=1);
    logger.info('x_train={}\ty_train={}\tlabel_train={}\tx_test={}\ty_test={}\tlabel_test={}'.format(x_train.shape, y_train.shape, label_train.shape, x_test.shape, y_test.shape, label_test.shape ));

    loadtrain = Loader(x_train, y_train, shuffle=False);
    loadtest = Loader(x_test, y_test, shuffle=False);

    if os.path.isfile(os.path.join(model_dir, 'MNIST.meta')) and args.overwrite <= 0:
        logger.info('Loading pretrained model at {}'.format(model_dir));
        mnist = MNIST(clust_num=len(np.unique(label_train)), use_cnn=args.use_cnn, beta=args.beta, restore_folder=model_dir );
    else:
        mnist = MNIST(clust_num=len(np.unique(label_train)), use_cnn=args.use_cnn, beta=args.beta, );
        mnist.train(loadtrain, steps=200, batch_size=1024);
        mnist.predict(loadtest);
        mnist.save(folder=model_dir);

    embedding_test = mnist.get_embedding(loadtest); logger.info('embedding_test = {}'.format(embedding_test.shape ));
    if args.plot_embedding > 0: plot_embedding(embedding_test, label_test, use_tsne=True);

    # lrp = LRP(label_test, x_test, embedding_test, mnist);
    # clust_id_pairs = [(0,8), (1,4), (1,7), (2,3), (3,8), (4,5), (4,9), (5,0), (5,6), (5,8),(6,0), (7,0), (7,1), (7,2), (7,9), (8,0), (8,3), (8,5), (9,0), (9,8),];
    clust_id_pairs = [(1, 7), ];

    for clust_c_id, clust_k_id in clust_id_pairs:
        tag = 'cw_cluster_margin{}_lamda{}_iter{}_lr{}'.format(args.margin, args.lamda, args.max_iter, args.lr);
        relevance_url = os.path.join(result_dir, '{}_label_from{}_to{}.npz'.format(tag, clust_c_id, clust_k_id, ));
        if os.path.isfile(relevance_url):
            relevance_data = np.load(relevance_url);
            label_sub = relevance_data['labels']; data_sub = relevance_data['data']; scores_sub = relevance_data['scores'];
        else:
            scores_sub, data_sub, label_sub = lrp.calc_carlini_wagner(clust_c_id, clust_k_id, lamda=args.lamda, max_iter=args.max_iter, lr=args.lr, margin=args.margin, );
            np.savez(relevance_url, scores=scores_sub, data=data_sub, labels=label_sub);
        logger.info('tag={}\tclust_c_id={}\tclust_k_id={}\tscores_sub={}\tdata_sub={}\tlabel_sub={}'.format(tag, clust_c_id, clust_k_id, scores_sub.shape, data_sub.shape, label_sub.shape, ));
        compare_logodds(data_sub, scores_sub, mnist, clust_c_id, clust_k_id, args.output_dir);
        #
        # tag = 'cw_classify_margin{}_lamda{}_iter{}_lr{}'.format(args.margin, args.lamda, args.max_iter, args.lr);
        # relevance_url = os.path.join(result_dir, '{}_label_from{}_to{}.npz'.format(tag, clust_c_id, clust_k_id, ));
        # if os.path.isfile(relevance_url):
        #     relevance_data = np.load(relevance_url);
        #     label_sub = relevance_data['labels']; data_sub = relevance_data['data']; scores_sub = relevance_data['scores'];
        # else:
        #     scores_sub, data_sub, label_sub = lrp.calc_carlini_wagner_classification(clust_c_id, clust_k_id, lamda=args.lamda, max_iter=args.max_iter, lr=args.lr, margin=args.margin, );
        #     np.savez(relevance_url, scores=scores_sub, data=data_sub, labels=label_sub);
        # logger.info('tag={}\tclust_c_id={}\tclust_k_id={}\tscores_sub={}\tdata_sub={}\tlabel_sub={}'.format(tag, clust_c_id, clust_k_id, scores_sub.shape, data_sub.shape, label_sub.shape, ));
        # compare_logodds(data_sub, scores_sub, mnist, clust_c_id, clust_k_id);



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description');
    parser.add_argument('--output_dir', type=str, help='output_dir');
    parser.add_argument('--beta', type=float, help='beta', default=1.0);
    parser.add_argument('--use_cnn', type=int, help='use_cnn', default=1);

    parser.add_argument('--lamda', type=float, help='lamda', default=1e2);
    parser.add_argument('--lr', type=float, help='lr', default=1e-3);
    parser.add_argument('--max_iter', type=int, help='max_iter', default=3000);
    parser.add_argument('--margin', type=float, help='margin', default=0);

    parser.add_argument('--plot_embedding', type=int, help='plot_embedding', default=0);
    parser.add_argument('--overwrite', type=int, help='overwrite', default=0);

    args = parser.parse_args();
    main(args);


"""example command"""
# python explainer/mnist_explainer.py --output_dir ../data/mnist --use_cnn 1 --beta 0.05
# python explainer/mnist_explainer.py --output_dir ../data/mnist --use_cnn 0 --beta 0.1

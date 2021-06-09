import os, sys, argparse, random, time;

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import numpy as np;
from sklearn.decomposition import PCA, KernelPCA;
from sklearn.preprocessing import normalize
import tensorflow as tf;
import umap;
import matplotlib
import matplotlib.pyplot as plt

try:
    from PIL import Image
except ImportError:
    import Image

import utils.io_utils as io_utils;
import utils.viz_utils as viz_utils;
from matplotlib.backends.backend_pdf import PdfPages

from mnist.loader import Loader
from mnist.model import MNIST

# https://github.com/rasbt/mlxtend/blob/master/mlxtend/plotting/decision_regions.py
# http://localhost:8888/notebooks/mnist.ipynb

def get_x_grid(x_test, grid_bin = 100):
    x_test_reshape = x_test.reshape(-1, 784);

    mapper = PCA(n_components=2);
    mapper.fit(x_test_reshape);
    x_test_embedding = mapper.transform(x_test_reshape);

    grid_x, grid_y = np.meshgrid(np.linspace(np.min(x_test_embedding[:, 0]) - 0.1, np.max(x_test_embedding[:, 0]) + 0.1, grid_bin), np.linspace(np.min(x_test_embedding[:, 1]) - 0.1, np.max(x_test_embedding[:, 1]) + 0.1, grid_bin));
    x_grid_embedding = np.vstack([grid_x.reshape(-1), grid_y.reshape(-1)]).T;
    x_grid = mapper.inverse_transform(x_grid_embedding);
    x_grid = x_grid.reshape(-1, 28, 28, 1);

    return x_grid;

def visualize_x_embedding(x_test_embedding, x_grid_embedding, label_test, use_umap=True  ):

    if len(x_test_embedding.shape) > 2 and len(x_grid_embedding.shape) > 2:
        x_test_embedding = x_test_embedding.reshape(-1, 784);
        x_grid_embedding = x_grid_embedding.reshape(-1, 784);

    if use_umap:
        # mapper = umap.UMAP(random_state=1).fit(x_test_embedding);
        mapper = umap.UMAP(random_state=1).fit(np.vstack((x_test_embedding, x_grid_embedding)));
        x_test_2d = mapper.transform(x_test_embedding);
        x_grid_2d = mapper.transform(x_grid_embedding);
    else:
        mapper = PCA(n_components=2);
        # mapper.fit(x_test_embedding);
        mapper.fit(np.vstack((x_test_embedding, x_grid_embedding)));
        x_test_2d = mapper.transform(x_test_embedding);
        x_grid_2d = mapper.transform(x_grid_embedding);

    # figure = plt.figure(figsize=(12, 12));
    # sc = plt.scatter(x_test_2d[:, 0], x_test_2d[:, 1], s=5, c=label_test, cmap='Spectral');
    # plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10));
    # plt.scatter(x_grid_2d[:, 0], x_grid_2d[:, 1], color='black', s=1, alpha=0.3);
    # plt.show();

    return x_test_2d, x_grid_2d;


def plot_embedding_by_umap(x_test, label_test, grid_bin = 100, use_umap=True ):
    assert len(x_test) == len(label_test);

    x_test_reshape = x_test.reshape(-1, 784);

    if use_umap:
        mapper = umap.UMAP(random_state=1).fit(x_test_reshape);
        x_test_embedding = mapper.transform(x_test_reshape);
    else:
        mapper = PCA(n_components=2);
        mapper.fit(x_test_reshape);
        x_test_embedding = mapper.transform(x_test_reshape);


    grid_x, grid_y = np.meshgrid(np.linspace(np.min(x_test_embedding[:, 0])-0.1, np.max(x_test_embedding[:, 0])+0.1, grid_bin), np.linspace(np.min(x_test_embedding[:, 1])-0.1, np.max(x_test_embedding[:, 1])+0.1, grid_bin));
    x_grid_embedding = np.vstack([grid_x.reshape(-1), grid_y.reshape(-1)]).T;
    x_grid = mapper.inverse_transform(x_grid_embedding);

    x_grid = x_grid.reshape(-1, 28, 28, 1);
    logger.info('x_grid={}\tx_grid_embedding={}'.format(x_grid.shape, x_grid_embedding.shape ));

    figure = plt.figure(figsize=(12, 12));
    sc = plt.scatter(x_test_embedding[:, 0], x_test_embedding[:, 1], s=5, c=label_test, cmap='Spectral');
    plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10));
    # plt.scatter(x_grid_embedding[:, 0], x_grid_embedding[:, 1], color='black', s=2);
    plt.xlabel('UMAP 1', fontsize=25);
    plt.ylabel('UMAP 2', fontsize=25);
    plt.show();

    return x_grid, x_grid_embedding, x_test_embedding;


def plot_prediction_decision_regions(x_test_embedding, y_test_pred, x_grid_embedding, y_grid_pred, result_dir, grid_bin = 100, tag='' ):
    logger.info('x_test_embedding={}\ty_test_pred={}'.format(x_test_embedding.shape, y_test_pred.shape,  ));
    logger.info('x_grid_embedding={}\ty_grid_pred={}'.format(x_grid_embedding.shape, y_grid_pred.shape,  ));
    assert x_test_embedding.shape[1] == 2 and x_grid_embedding.shape[1] == 2 ;
    assert len(x_test_embedding) == len(y_test_pred) and len(x_grid_embedding) == len(y_grid_pred) ;

    contourf_kwargs = {'alpha': 0.1, 'antialiased': True};

    for curr_label in range(10):
        figure = plt.figure(figsize=(12, 12));
        # ax = plt.gca();
        ax = figure.add_subplot(1, 1, 1);

        x_grid_embedding_dim1 = x_grid_embedding[:, 0].reshape(-1, grid_bin);
        x_grid_embedding_dim2 = x_grid_embedding[:, 1].reshape(-1, grid_bin);

        sc = plt.scatter(x_test_embedding[:, 0], x_test_embedding[:, 1], s=8, c=np.argmax(y_test_pred, axis=1), cmap='Spectral' );
        plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10));
        label_rgba_map = {label: sc.to_rgba(label) for label in range(10)}
        label_hex_map = {label: matplotlib.colors.to_hex(label_rgba_map[label]) for label in range(10)}

        label_grid = y_grid_pred[:, curr_label].reshape(-1, grid_bin);
        cset = ax.contourf(x_grid_embedding_dim1, x_grid_embedding_dim2, label_grid, cmap='Greys', **contourf_kwargs )
        ax.contour(x_grid_embedding_dim1, x_grid_embedding_dim2, label_grid, cset.levels, colors=label_hex_map[curr_label], linewidths=2, antialiased=True)

        # for curr_label1 in range(10):
        #     label_grid = y_grid_pred[:, curr_label1].reshape(-1, grid_bin);
        #     contourf_kwargs = {'alpha': 0.02, 'antialiased': True};
        #     cset = ax.contourf(x_grid_embedding_dim1, x_grid_embedding_dim2, label_grid, cmap='Greys', **contourf_kwargs )
        #     ax.contour(x_grid_embedding_dim1, x_grid_embedding_dim2, label_grid, cset.levels, colors=label_hex_map[curr_label1], linewidths=1.5, antialiased=True)

        ax.xaxis.set_tick_params(labelsize=20);
        ax.yaxis.set_tick_params(labelsize=20);
        if 'umap0' in tag:
            ax.set_xlabel('PCA1', fontsize=25);
            ax.set_ylabel('PCA2', fontsize=25);
        elif 'umap1' in tag:
            ax.set_xlabel('UMAP1', fontsize=25);
            ax.set_ylabel('UMAP2', fontsize=25);
        else: assert False;

        pp = PdfPages(os.path.join(result_dir, 'fig_decisionreg_tag_{}_label_{}.pdf'.format(tag, curr_label)));
        pp.savefig(figure, bbox_inches='tight');
        pp.close();
        plt.close(figure);
        del figure;



def compare_importance(result_dir, sample_size=900):
    target_pair_list = [(0,8),(1,4),(1,7),(2,1),(2,3),(2,9),(3,8),(4,5),(4,6),(4,9),(5,0),(5,6),(5,8),(5,9),(6,0),(7,0),(7,1),(7,2),(7,9),(8,0),(8,3),(8,5),(9,0),(9,8)];

    for from_id, to_id in target_pair_list:
        label = 'from{}_to{}'.format(from_id, to_id);

        npz_url_list = [];
        for file in os.listdir(result_dir):
            if label in file: npz_url_list.append(os.path.join(result_dir, file));
        logger.info(npz_url_list);

        for npz_url in npz_url_list:
            is_clust = True;
            if 'classify' in npz_url: is_clust = False;

            fig_dir = os.path.join(result_dir, 'figs', 'fig_{}'.format(label));
            if not os.path.exists(fig_dir): os.makedirs(fig_dir);

            relevance_data = np.load(npz_url);
            label_sub = relevance_data['labels']; data_sub = relevance_data['data']; scores_sub = relevance_data['scores'];
            data_sub = data_sub[:sample_size]; scores_sub = scores_sub[:sample_size];
            logger.info('scores_sub={}\tmax={}\tmin={}'.format(scores_sub.shape, np.max(scores_sub), np.min(scores_sub), )),

            orig_img_url = os.path.join(fig_dir, 'fig_orig_{}_clust{}.png'.format(label, int(is_clust)));
            rel_img_url = os.path.join(fig_dir, 'fig_rel_{}_clust{}.png'.format(label, int(is_clust)));
            blend_img_url = os.path.join(fig_dir, 'fig_blend_{}_clust{}.png'.format(label, int(is_clust)));

            viz_utils.save_image(viz_utils.create_score_sprite(data_sub, normalize=False), orig_img_url, cmap='binary');
            viz_utils.save_image(viz_utils.create_score_sprite(scores_sub, normalize=False), rel_img_url, cmap='bwr');

            """superimpose two images together"""
            background = Image.open(orig_img_url);
            overlay = Image.open(rel_img_url);
            background = background.convert("RGBA");
            overlay = overlay.convert("RGBA");
            new_img = Image.blend(background, overlay, 0.7);
            new_img.save(blend_img_url, "PNG");

            os.remove(rel_img_url);



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

        self.relevance_preparation();

    def relevance_preparation(self, debug_mode=True):
        """neuralizing the cluster results"""
        self.clust_logits = [];
        for clust_c_id in self.uniq_labels: self.clust_logits.append(self.neuralize_by_clustid(clust_c_id, debug_mode=debug_mode));
        if debug_mode: logger.info('clust_c_id={}\tclust_logits={}'.format(clust_c_id, self.clust_logits));

        """sanity check of neuralization"""
        # for clust_c_id in self.uniq_labels:
        #     if not self.sanity_check_neuralization_by_clustid(clust_c_id, debug_mode=debug_mode):
        #         raise RuntimeError('Neuralization failed by encountering Inf clust logits! Try a smaller beta value! ');

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

    def sanity_check_neuralization_by_clustid(self, clust_c_id, debug_mode=False):
        sample_c_indices, sample_c_profiles, clust_c_centroid = self.get_clust_profiles(clust_c_id);

        logger.info('--------------------------------------------------------------------');
        logger.info('neuralization sanity check:\tclust_c_id={}\tsample_c_profiles={}'.format(clust_c_id, sample_c_profiles.shape));

        for clust_logits_arr in self.mnist.sess.run(self.clust_logits, feed_dict={self.X: sample_c_profiles}):
            # if debug_mode: logger.info('clust_logits_arr={}'.format(clust_logits_arr));
            if np.any([np.isinf(x) for x in clust_logits_arr]): return False;
        return True;

    def predict(self, new_data):
        clust_logits_list = self.mnist.sess.run(self.clust_logits, feed_dict={self.X: new_data});

        clust_logits_pred_arr = np.zeros((len(clust_logits_list[0]), len(clust_logits_list) ));
        for idx in range(self.clust_num): clust_logits_pred_arr[:, idx] = clust_logits_list[idx];
        logger.info('clust_logits_pred_arr={}'.format(clust_logits_pred_arr.shape, ));
        return clust_logits_pred_arr;


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
        mnist.save(folder=model_dir);

    embedding_test = mnist.get_embedding(loadtest); logger.info('embedding_test = {}'.format(embedding_test.shape ));
    if args.plot_embedding > 0: plot_embedding_by_umap(embedding_test, label_test);

    # compare_importance(result_dir);

    # grid_bin=250; use_umap=False;
    # saved_grid_url = os.path.join(result_dir, 'x_grid{}.npz'.format(grid_bin, ));
    # if os.path.isfile(saved_grid_url):
    #     data = np.load(saved_grid_url, allow_pickle=True);
    #     x_grid = data['x_grid']; embedding_grid = data['embedding_grid']; y_grid_pred = data['y_grid_pred'];
    # else:
    #     x_grid = get_x_grid(x_test, grid_bin=grid_bin, );
    #     loadgrid = Loader(x_grid, np.random.rand(len(x_grid), y_test.shape[1]), shuffle=False);
    #     embedding_grid = mnist.get_embedding(loadgrid);
    #     y_grid_pred = mnist.predict(loadgrid);
    #     np.savez(saved_grid_url, x_grid=x_grid, embedding_grid=embedding_grid, y_grid_pred=y_grid_pred, );
    # logger.info('x_grid={}\tembedding_grid={}'.format(x_grid.shape, embedding_grid.shape ));

    # # x_test_2d, x_grid_2d = visualize_x_embedding(embedding_test, embedding_grid, label_test, use_umap=use_umap);
    # x_test_2d, x_grid_2d = visualize_x_embedding(x_test, x_grid, label_test, use_umap=use_umap);
    #
    # y_test_pred = mnist.predict(loadtest);
    # y_test_pred = normalize(np.exp(y_test_pred), axis=1, norm='l1');
    # y_grid_pred = normalize(np.exp(y_grid_pred), axis=1, norm='l1');
    # plot_prediction_decision_regions(x_test_2d, y_test_pred, x_grid_2d, y_grid_pred, result_dir, grid_bin=grid_bin, tag='prediction_umap{}'.format(int(use_umap)) );
    #
    # lrp = LRP(label_test, x_test, embedding_test, mnist);
    # y_grid_pred = normalize(np.exp(lrp.predict(x_grid)), axis=1, norm='l1');
    # plot_prediction_decision_regions(x_test_2d, y_test_pred, x_grid_2d, y_grid_pred, result_dir, grid_bin=grid_bin, tag='clustering_umap{}'.format(int(use_umap)) );



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description');
    parser.add_argument('--output_dir', type=str, help='output_dir');
    parser.add_argument('--beta', type=float, help='beta', default=1.0);
    parser.add_argument('--use_cnn', type=int, help='use_cnn', default=1);

    parser.add_argument('--lamda', type=float, help='lamda', default=1e2);
    parser.add_argument('--lr', type=float, help='lr', default=1e-3);
    parser.add_argument('--max_iter', type=int, help='max_iter', default=3000);
    parser.add_argument('--margin', type=float, help='margin', default=0.5);

    parser.add_argument('--plot_embedding', type=int, help='plot_embedding', default=0);
    parser.add_argument('--overwrite', type=int, help='overwrite', default=0);

    args = parser.parse_args();
    main(args);


"""example command"""
# python explainer/mnist_plotter.py --output_dir ../data/mnist --use_cnn 1 --beta 0.05
# python explainer/mnist_plotter.py --output_dir ../data/mnist --use_cnn 0 --beta 0.1

import os, sys, argparse, random, time;

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import numpy as np;
from sklearn.decomposition import PCA;
from sklearn.manifold import TSNE;
from scipy import sparse;
import tensorflow as tf;

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

from sklearn.datasets import fetch_20newsgroups, fetch_20newsgroups_vectorized ;
from sklearn.feature_extraction import text

import utils.io_utils as io_utils;
import utils.viz_utils as viz_utils;

from newsgroup.loader import Loader, LoaderCNN
from newsgroup.model import NEWSG
from newsgroup.model_cnn import NEWSGCNN

MAX_SEQUENCE_LENGTH = 500
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100

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


class LRP:
    def __init__(self, labels, x_data, x_embeddings, mnist):
        self.labels = labels;
        self.uniq_labels = np.unique(self.labels);
        self.clust_num = len(self.uniq_labels);

        self.x_data = x_data;
        self.x_embeddings = x_embeddings;
        assert len(self.labels) == self.x_data.shape[0] and len(self.labels) == len(self.x_embeddings);

        self.is_data_sparse = False;
        if sparse.issparse(self.x_data): self.is_data_sparse = True;


        self.mnist = mnist;
        self.embed_weights, self.embed_biases, self.embed_activations = self.mnist.get_embedder_tensors();
        self.clust_weights, self.clust_biases, self.clust_activations = self.mnist.get_clust_tensors();
        self.X = self.embed_activations[0];

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

        if self.is_data_sparse: sample_profiles = sample_profiles.todense();
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

    def calc_saliency(self, clust_c_id, saliency_type=0 ):
        assert clust_c_id in self.uniq_labels;

        sample_indices = np.where(self.labels == clust_c_id)[0];
        x_data_sub = self.x_data[sample_indices];
        labels_sub = self.labels[sample_indices];
        if self.is_data_sparse: x_data_sub = x_data_sub.todense();

        _, _, _, clust_c_minpoolAct_tensor = self.get_tensor_by_clustid(clust_c_id);
        clust_grads_tensor = tf.gradients(clust_c_minpoolAct_tensor, self.X)[0];

        if saliency_type == 0:
            """vanilla gradient"""
            clust_grads = self.mnist.sess.run(clust_grads_tensor, feed_dict={self.X: x_data_sub});
        elif saliency_type == 1:
            """Smooth Gradient"""
            clust_grads = np.zeros_like(x_data_sub); repeat=20; stdev = np.std(x_data_sub) * 0.2;
            for iter in range(repeat):
                noise = np.random.normal(0, stdev, x_data_sub.shape);
                clust_grads += self.mnist.sess.run(clust_grads_tensor, feed_dict={self.X: x_data_sub + noise});
        elif saliency_type == 2:
            """Integrated Gradients"""
            clust_grads = np.zeros_like(x_data_sub); baseline = np.zeros_like(x_data_sub); steps=20;
            for alpha in np.linspace(0, 1, steps):
                clust_grads += self.mnist.sess.run(clust_grads_tensor, feed_dict={self.X: alpha * x_data_sub});
        elif saliency_type == 3:
            """SHAP: conda install -c conda-forge shap """
            import shap;
            e = shap.GradientExplainer((self.X, clust_c_minpoolAct_tensor), x_data_sub, session=self.mnist.sess, local_smoothing=0);
            clust_grads = e.shap_values(x_data_sub);
        else: assert False;
        return clust_grads, x_data_sub, labels_sub;

    def calc_carlini_wagner(self, clust_c_id, clust_k_id, lamda=1e2, max_iter=5000, lr=2e-3, margin=0, min_val=0.0, max_val=1.0):
        assert clust_c_id in self.uniq_labels and clust_k_id in self.uniq_labels;
        assert clust_c_id != clust_k_id;

        sample_indices = np.where(self.labels == clust_c_id)[0];
        x_data_sub = self.x_data[sample_indices];
        labels_sub = self.labels[sample_indices];
        if self.is_data_sparse: x_data_sub = x_data_sub.todense();

        logger.info('clust_c_id={}\tclust_k_id={}'.format(clust_c_id, clust_k_id, ));
        _, _, _, clust_c_minpoolAct_tensor = self.get_tensor_by_clustid(clust_c_id);
        _, _, _, clust_k_minpoolAct_tensor = self.get_tensor_by_clustid(clust_k_id);

        clust_loss_tensor = tf.math.maximum(margin + clust_c_minpoolAct_tensor - clust_k_minpoolAct_tensor, 0);
        clust_loss_grads_tensor = tf.gradients(clust_loss_tensor, self.X)[0];

        """starting carlini_wagner optimization"""
        x0 = np.copy(x_data_sub); curr_x = x0 + 1e-6;
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

def load_glove(glove_url):
    assert os.path.isfile(glove_url);
    glove_dict_url = glove_url.replace('.txt', '.npy');
    if os.path.isfile(glove_dict_url):
        glove_dict = np.load(glove_dict_url, allow_pickle=True).item();
    else:
        glove_dict = {};
        f = open(glove_url, encoding="utf-8");
        for line in f:
            values = line.split();
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32');
            glove_dict[word] = coefs;
        f.close();
        np.save(glove_dict_url, glove_dict);

    logger.info('glove_dict={}'.format(len(glove_dict)));
    return glove_dict;

def vectorize_data_tfidf(x_train, x_test):
    from sklearn.feature_extraction.text import TfidfVectorizer;
    vectorizer = TfidfVectorizer(stop_words={'english'}, min_df=2);
    vectorizer.fit(x_train + x_test);
    x_train_vec = vectorizer.transform(x_train);
    x_test_vec = vectorizer.transform(x_test);
    vocabulary = vectorizer.vocabulary_;
    logger.info('x_train_vec={}\tx_test_vec={}\tvocabulary={}'.format(x_train_vec.shape, x_test_vec.shape, len(vocabulary) ));
    return x_train_vec, x_test_vec, vocabulary;

def vectorize_data_glove(x_train, x_test, glove_dict):
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences

    x_ensemble = x_train + x_test;
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS);
    tokenizer.fit_on_texts(x_ensemble);
    sequences = tokenizer.texts_to_sequences(x_ensemble);
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post');
    logger.info('sequences={}\tdata={}'.format(len(sequences), data.shape, ));

    vocabulary = tokenizer.word_index;
    embedding_matrix = np.zeros((len(vocabulary) + 1, EMBEDDING_DIM));
    for word, i in vocabulary.items():
        embedding_vector = glove_dict.get(word);
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector;
    logger.info('vocabulary={}\tembedding_matrix={}'.format(len(vocabulary), embedding_matrix.shape, ));

    data_train = data[:len(x_train)];
    data_test = data[len(x_train):];
    logger.info('data_train={}\tdata_test={}'.format(data_train.shape, data_test.shape, ));

    return data_train, data_test, vocabulary, embedding_matrix;


def baseline_naive_bayes(x_train_vec, x_test_vec, label_train, label_test ):
    from sklearn.naive_bayes import MultinomialNB
    from sklearn import metrics
    clf = MultinomialNB(alpha=.01);
    clf.fit(x_train_vec, label_train);
    pred = clf.predict(x_test_vec);
    logger.info(metrics.accuracy_score(label_test, pred))

def main(args):
    model_dir = os.path.join(args.output_dir, 'models_{}'.format(get_model_description(args)));
    if not os.path.exists(model_dir): os.makedirs(model_dir);
    result_dir = os.path.join(args.output_dir, 'results_{}'.format(get_model_description(args)));
    if not os.path.exists(result_dir): os.makedirs(result_dir);

    # categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space'];
    # categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x'];
    # categories = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'];
    # categories = ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'];
    # categories = ['talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'];
    categories = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'];

    x_train, label_train = fetch_20newsgroups(data_home=args.output_dir, subset='train', categories=categories, remove=('headers', 'footers', 'quotes'), shuffle=False, return_X_y=True );
    y_train = np.zeros((len(x_train), len(np.unique(label_train)) ));
    y_train[range(len(x_train)), label_train] = 1;
    logger.info('x_train={}\tlabel_train={}\ty_train={}\t{}'.format(len(x_train), label_train.shape, y_train.shape, np.unique(label_train) ));

    x_test, label_test = fetch_20newsgroups(data_home=args.output_dir, subset='test', categories=categories, remove=('headers', 'footers', 'quotes'), shuffle=False, return_X_y=True);
    y_test = np.zeros((len(x_test), len(np.unique(label_test))));
    y_test[range(len(x_test)), label_test] = 1;
    logger.info('x_test={}\tlabel_test={}\ty_test={}\t{}'.format(len(x_test), label_test.shape, y_test.shape, np.unique(label_test) ));


    if args.use_cnn:
        data_train, data_test, vocabulary, embedding_matrix = vectorize_data_glove(x_train, x_test, load_glove(os.path.join(args.output_dir, 'glove.6B.100d.txt')));
        loadtrain = LoaderCNN(data_train, y_train, embedding_matrix, shuffle=False);
        loadtest = LoaderCNN(data_test, y_test, embedding_matrix, shuffle=False);
        if os.path.isfile(os.path.join(model_dir, 'NEWSG.meta')) and args.overwrite <= 0:
            logger.info('Loading pretrained model at {}'.format(model_dir));
            newsg = NEWSGCNN(input_dim=MAX_SEQUENCE_LENGTH, channel_dim=EMBEDDING_DIM, clust_num=len(np.unique(label_train)), beta=args.beta, restore_folder=model_dir );
        else:
            newsg = NEWSGCNN(input_dim=MAX_SEQUENCE_LENGTH, channel_dim=EMBEDDING_DIM, clust_num=len(np.unique(label_train)), beta=args.beta, );
            newsg.train(loadtrain, steps=200, batch_size=256);
            newsg.predict(loadtest, batch_size=256);
            newsg.save(folder=model_dir);

    else:
        x_train_vec, x_test_vec, vocabulary = vectorize_data_tfidf(x_train, x_test);
        loadtrain = Loader(x_train_vec, y_train, shuffle=False);
        loadtest = Loader(x_test_vec, y_test, shuffle=False);
        if os.path.isfile(os.path.join(model_dir, 'NEWSG.meta')) and args.overwrite <= 0:
            logger.info('Loading pretrained model at {}'.format(model_dir));
            newsg = NEWSG(input_dim=x_train_vec.shape[1], clust_num=len(np.unique(label_train)), beta=args.beta, restore_folder=model_dir );
        else:
            newsg = NEWSG(input_dim=x_train_vec.shape[1], clust_num=len(np.unique(label_train)), beta=args.beta, );
            newsg.train(loadtrain, steps=200, batch_size=256);
            newsg.predict(loadtest, batch_size=256);
            newsg.save(folder=model_dir);

    embedding_test = newsg.get_embedding(loadtest); logger.info('embedding_test = {}'.format(embedding_test.shape ));
    if args.plot_embedding > 0: plot_embedding(embedding_test, label_test, use_tsne=False);


    lrp = LRP(label_test, x_test_vec, embedding_test, newsg);



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description');
    parser.add_argument('--output_dir', type=str, help='output_dir');
    parser.add_argument('--beta', type=float, help='beta', default=1.0);
    parser.add_argument('--use_cnn', type=int, help='use_cnn', default=0);

    parser.add_argument('--lamda', type=float, help='lamda', default=1e2);
    parser.add_argument('--lr', type=float, help='lr', default=1e-3);
    parser.add_argument('--max_iter', type=int, help='max_iter', default=2000);
    parser.add_argument('--margin', type=float, help='margin', default=0.5);

    parser.add_argument('--plot_embedding', type=int, help='plot_embedding', default=0);
    parser.add_argument('--overwrite', type=int, help='overwrite', default=0);

    args = parser.parse_args();
    main(args);


"""example command"""
"""https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html"""
# python explainer/newsgroup_explainer.py --output_dir ../data/newsgroup --beta 0.4

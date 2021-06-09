import os, math
import numpy as np
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.python.keras import backend as K


"""LRP-related by Yang"""
class ClustMinPoolLayer(tf.keras.layers.Layer):
    def __init__(self, beta):
        super(ClustMinPoolLayer, self).__init__();
        self.beta = beta;

    def call(self, inputs, training=False):
        return - tf.math.log(tf.reduce_sum(tf.math.exp(inputs * tf.constant(-self.beta)), axis=1));

def nameop(op, name):
    """
    Give a name to a tensorflow op.

    :param op: a tensorflow op
    :param name: a string name for the op
    """
    op = tf.identity(op, name=name)
    return op

def tbn(name):
    """Get the tensor in the default graph of the given name."""
    return tf.get_default_graph().get_tensor_by_name(name)

def obn(name):
    """Get the operation node in the default graph of the given name."""
    return tf.get_default_graph().get_operation_by_name(name)


class MNIST(object):

    def __init__(self, clust_num, use_cnn=True, beta=1.0, learning_rate=.001, restore_folder='', limit_gpu_fraction=.3, no_gpu=False):

        self.use_cnn = use_cnn;
        self.learning_rate = learning_rate;
        self.iteration = 0;

        """LRP-related by Yang"""
        self.clust_num = clust_num;
        self.beta = beta;
        self.embed_dim = 128;

        if restore_folder:
            self._restore(restore_folder)
            return

        self.x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='x');
        self.y = tf.placeholder(tf.float32, shape=[None, self.clust_num], name='y');

        self._build();
        self.init_session(limit_gpu_fraction, no_gpu);
        self.graph_init(self.sess);

    def init_session(self, limit_gpu_fraction=.1, no_gpu=False):
        """
        Initialize a tensorflow session for MNIST.

        :param limit_gpu_fraction: float percentage of the avaiable gpu to use
        :param no_gpu: bool for whether or not to use the gpu if available
        """
        if no_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            config = tf.ConfigProto(device_count={'GPU': 0})
            self.sess = tf.Session(config=config)
        elif limit_gpu_fraction:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=limit_gpu_fraction)
            config = tf.ConfigProto(gpu_options=gpu_options)
            self.sess = tf.Session(config=config)
        else:
            self.sess = tf.Session()

    def graph_init(self, sess=None):
        """
        Initialize the tensorflow graph that's been created.

        :param sess: the session to use while initializing, if different from MNIST's sess member
        """
        if not sess: sess = self.sess
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        sess.run(tf.global_variables_initializer())

    def save(self, saver=None, sess=None, folder=None):
        """
        Save the current state of MNIST.

        :param saver: the saver instance to use
        :param sess: the session to save
        :param folder: the location to save MNIST's state to
        """
        if not saver: saver = self.saver
        if not sess: sess = self.sess
        if not folder: assert False;

        savefile = os.path.join(folder, 'MNIST')
        saver.save(sess, savefile, write_meta_graph=True)
        print("Model saved to {}".format(savefile))

    def _restore(self, restore_folder):
        """
        Restore the tensorflow graph stored in restore_folder.

        :param restore_folder: the location of the directory where the saved MNIST model resides.
        """
        tf.reset_default_graph()
        self.init_session()
        ckpt = tf.train.get_checkpoint_state(restore_folder)
        self.saver = tf.train.import_meta_graph('{}.meta'.format(ckpt.model_checkpoint_path))
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        print("Model restored from {}".format(restore_folder))

    def _build(self):
        """Coordinate the building of each part of MNIST."""
        if self.use_cnn: self._build_conv_layers();
        else: self._build_mlp_layers();
        self._build_clust_layers();

        self._build_losses();

        self._build_optimization();

    def _build_conv_layers(self):
        """Construct the layers of MNIST using CNN."""
        self.conv1 = tf.layers.conv2d(self.x, filters=32, kernel_size=[5, 5], padding='same', activation=tf.nn.relu, name='conv1', );
        self.conv1 = nameop(self.conv1, 'conv1_op');

        self.pool1 = tf.layers.max_pooling2d(self.conv1, pool_size=[2, 2], strides=2, padding='same', name='pool1', );
        self.pool1 = nameop(self.pool1, 'pool1_op');

        self.conv2 = tf.layers.conv2d(self.pool1, filters=32, kernel_size=[5,5], padding='same', activation=tf.nn.relu, name='conv2', );
        self.conv2 = nameop(self.conv2, 'conv2_op');

        self.pool2 = tf.layers.max_pooling2d(self.conv2, pool_size=[2, 2], strides=2, padding='same', name='pool2', );
        self.pool2 = nameop(self.pool2, 'pool2_op');

        self.flatten = tf.contrib.layers.flatten(self.pool2, );
        self.flatten = nameop(self.flatten, 'flatten_op');

        self.fc = tf.layers.dense(self.flatten, self.embed_dim, activation=tf.nn.relu, name='fc', );
        self.fc = nameop(self.fc, 'fc_op');

        self.output = tf.layers.dense(self.fc, self.clust_num, activation=tf.nn.softmax, name='output', );
        self.output = nameop(self.output, 'output_op');
        self.activation_names = ['conv1_op', 'pool1_op', 'conv2_op', 'pool2_op', 'fc_op'];


    def _build_mlp_layers(self):
        """Construct the layers of MNIST using MLP."""
        self.flatten = tf.contrib.layers.flatten(self.x, );
        self.flatten = nameop(self.flatten, 'flatten_op');

        self.fc1 = tf.layers.dense(self.flatten, 512, activation=tf.nn.relu, name='fc1', use_bias=True);
        self.fc1 = nameop(self.fc1, 'fc1_op');

        self.fc2 = tf.layers.dense(self.fc1, 256, activation=tf.nn.relu, name='fc2', use_bias=True);
        self.fc2 = nameop(self.fc2, 'fc2_op');

        self.fc = tf.layers.dense(self.fc2, self.embed_dim, activation=tf.nn.relu, name='fc', );
        self.fc = nameop(self.fc, 'fc_op');

        self.output = tf.layers.dense(self.fc, self.clust_num, activation=tf.nn.softmax, name='output', );
        self.output = nameop(self.output, 'output_op');
        self.activation_names = ['fc1_op', 'fc2_op', 'fc_op'];


    def _build_clust_layers(self):
        tf.add_to_collection('LRP', self.x);
        for name in self.activation_names: tf.add_to_collection('LRP', tbn("{}:0".format(name)));

        clust_pool_layer = ClustMinPoolLayer(self.beta);
        embedded = tbn("fc_op:0");
        for clust_id in range(self.clust_num):
            print('Add clustering layer...clust_id={}'.format(clust_id));
            clust_h = tf.layers.dense(embedded, self.clust_num - 1, activation=None, use_bias=True, kernel_initializer=tf.constant_initializer(0.0),
                              trainable=False, name='clust{}_linear'.format(clust_id));

            clust_h_name = 'clust{}_linear_op'.format(clust_id);
            clust_h = nameop(clust_h, clust_h_name);
            tf.add_to_collection('LRPClust', tbn("{}:0".format(clust_h_name)));

            clust_fc = clust_pool_layer(clust_h);
            clust_fc_name = 'clust{}_minpool_op'.format(clust_id);
            clust_fc = nameop(clust_fc, clust_fc_name);
            tf.add_to_collection('LRPClust', tbn("{}:0".format(clust_fc_name)));


    def _build_losses(self):
        """Build all the loss ops for the network."""
        self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=self.output, onehot_labels=self.y));
        self.loss = nameop(self.loss, 'loss');

    def _build_optimization(self, norm_clip=5.):
        """Build all the optimization ops for the network."""
        opt = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = opt.minimize(self.loss, name='train_op')

    def train(self, load, steps, batch_size=512):
        """
        Train MNIST.

        :param load: the loader object to yield batches from
        :param steps: the number of steps to train for
        :param batch_size: the number of points to train on in each step
        """

        start = self.iteration
        while (self.iteration - start) < steps:
            self.iteration += 1;
            print("MNIST train batch={}".format(self.iteration));

            data_batch, label_batch = load.next_batch(batch_size=batch_size);
            feed = {tbn('x:0'): data_batch, tbn('y:0'): label_batch, };
            ops = [obn('train_op')];
            self.sess.run(ops, feed_dict=feed);

    def predict(self, load, ):
        output_tensor = tbn("output_op:0");
        data_batch, label_batch = load.full_batch();
        feed = {tbn('x:0'): data_batch, tbn('y:0'): label_batch, };
        [pred_label] = self.sess.run([output_tensor], feed_dict=feed);
        assert label_batch.shape == pred_label.shape;
        score = accuracy_score(np.argmax(label_batch, axis=1), np.argmax(pred_label, axis=1)) * 100;
        print("Accuracy ={:.4f}\tpred_label={}".format(score, pred_label.shape ));
        return pred_label;


    def predict_logits(self, data_batch):
        feed = {tbn('x:0'): data_batch,  };
        [pred_label] = self.sess.run([tbn("output_op:0")], feed_dict=feed);
        print("pred_label={}".format(pred_label.shape ));
        return pred_label;


    def get_layer(self, load, name):
        """
        Get the actual values in array_like form from an abstract tensor.

        :param load: the loader object to iterate over
        :param name: the name of the tensor to evaluate for each point
        """
        tensor_name = "{}:0".format(name);
        tensor = tbn(tensor_name);

        data_batch, label_batch = load.full_batch();
        feed = {tbn('x:0'): data_batch, tbn('y:0'): label_batch, };
        [layer] = self.sess.run([tensor], feed_dict=feed)
        return layer;

    def get_embedding(self, load):
        """Return the embedding layer."""
        embedding = self.get_layer(load, 'fc_op')
        return embedding

    """LRP-related by Yang from here"""
    def get_embed_dim(self):
        return self.embed_dim;

    def get_clust_tensors(self):
        clust_weights = [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='.*kernel.*') if 'clust' in x.name];
        clust_biases = [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='.*bias.*') if 'clust' in x.name];
        clust_activations = tf.get_collection('LRPClust');
        print('clust_weights={}\n\n clust_biases={}\n\n clust_activations={}\n\n'.format(clust_weights, clust_biases, clust_activations));
        return clust_weights, clust_biases, clust_activations;

    def set_clust_tensors(self, clust_weight_tensor, clust_bias_tensor, new_weights, new_biases, debug_mode=True):
        if debug_mode:
            old_tensor_weights, old_tensor_biases = self.sess.run([clust_weight_tensor, clust_bias_tensor]);
            print('old_tensor_weights={}\told_tensor_biases={}'.format(old_tensor_weights, old_tensor_biases));
            print('new_weights={}\tnew_biases={}'.format(new_weights, new_biases));

        assign_weights = tf.assign(clust_weight_tensor, new_weights);
        assign_biases = tf.assign(clust_bias_tensor, new_biases);
        self.sess.run([assign_weights, assign_biases]);

        if debug_mode:
            new_tensor_weights, new_tensor_biases = self.sess.run([clust_weight_tensor, clust_bias_tensor]);
            print('new_tensor_weights={}\tnew_tensor_biases={}'.format(new_tensor_weights, new_tensor_biases));

    def get_embedder_tensors(self):
        embed_weights = [x for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='.*kernel.*') if 'output' not in x.name ];
        embed_biases = [x for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='.*bias.*') if 'output' not in x.name ];
        embed_activations = tf.get_collection('LRP');
        print('embed_weights={}\n\n embed_biases={}\n\n embed_activations={}\n\n'.format(embed_weights, embed_biases, embed_activations));
        return embed_weights, embed_biases, embed_activations;

    def get_output_tensor(self):
        output_tensor = tbn("output_op:0");
        print('output_tensor={}'.format(output_tensor));
        return output_tensor;

    def get_targeted_output_tensor(self, columns):
        output_tensor = tbn("output_op:0");
        targeted_output_tensor = tf.gather(output_tensor, columns, axis=1);
        print('targeted_output_tensor={}\tcolumns={}'.format(targeted_output_tensor, columns));
        return targeted_output_tensor;


    def calc_saliency(self, data_batch, clust_id, saliency_type=0 ):
        # output_tensor = tbn("output_op:0");
        output_tensor = self.get_targeted_output_tensor([clust_id]);
        input_tensor = tbn('x:0');
        grads_tensor = tf.gradients(output_tensor, input_tensor)[0];

        if saliency_type == 0:
            grads = self.sess.run(grads_tensor, feed_dict={input_tensor: data_batch});
        elif saliency_type == 1:
            """Smooth Gradient"""
            grads = np.zeros_like(data_batch);
            repeat = 20;
            stdev = np.std(data_batch) * 0.2;
            for iter in range(repeat):
                noise = np.random.normal(0, stdev, data_batch.shape);
                grads += self.sess.run(grads_tensor, feed_dict={input_tensor: data_batch+noise});
        elif saliency_type == 3:
            import shap;
            e = shap.GradientExplainer((input_tensor, grads_tensor), data_batch, session=self.sess, local_smoothing=0);
            grads = e.shap_values(data_batch);
        else: assert False;
        print('grads={}\tmin={}\tmax={}'.format(grads.shape, np.min(grads), np.max(grads)));
        return grads;
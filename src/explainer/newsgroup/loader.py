
import numpy as np

class Loader(object):

    def __init__(self, data, labels, shuffle=False ):
        self.start = 0
        self.epoch = 0
        self.data = data;
        self.labels = labels;
        assert self.data.shape[0] == len(self.labels);

        if shuffle:
            self.r = list(range(self.data.shape[0]))
            np.random.shuffle(self.r)
            self.data = self.data[self.r];
            self.labels = self.labels[self.r];

    def next_batch(self, batch_size=128):
        num_rows = self.data.shape[0];
        print('num_rows={}'.format(num_rows ));

        if self.start + batch_size < num_rows:
            data_batch = self.data[self.start:(self.start + batch_size)];
            label_batch = self.labels[self.start:(self.start + batch_size)];
            self.start += batch_size
        else:
            data_batch = self.data[self.start:]
            label_batch = self.labels[self.start:];
            self.start = 0
            self.epoch += 1

        print('data_batch={}\tlabel_batch={}'.format(data_batch.shape, label_batch.shape ));
        return data_batch.todense(), label_batch;

    def iter_batches(self, batch_size=128):
        num_rows = self.data.shape[0];
        print('num_rows={}'.format(num_rows ));
        end = 0

        for i in range(num_rows // batch_size):
            start = i * batch_size
            end = (i + 1) * batch_size

            data_batch = self.data[start:end];
            label_batch = self.labels[start:end];
            print('data_batch={}\tlabel_batch={}'.format(data_batch.shape, label_batch.shape ));
            yield data_batch.todense(), label_batch;

        if end != num_rows:
            data_batch = self.data[end:];
            label_batch = self.labels[end:];
            print('data_batch={}\tlabel_batch={}'.format(data_batch.shape, label_batch.shape ));
            yield data_batch.todense(), label_batch;


class LoaderCNN(object):

    def __init__(self, data, labels, embedding_matrix, shuffle=False ):
        self.start = 0
        self.epoch = 0
        self.data = data;
        self.labels = labels;
        self.embedding_matrix = embedding_matrix;
        assert self.data.shape[0] == len(self.labels);

        if shuffle:
            self.r = list(range(self.data.shape[0]))
            np.random.shuffle(self.r)
            self.data = self.data[self.r];
            self.labels = self.labels[self.r];

    def next_batch(self, batch_size=128):
        num_rows = self.data.shape[0];
        print('num_rows={}'.format(num_rows ));

        if self.start + batch_size < num_rows:
            data_batch = self.data[self.start:(self.start + batch_size)];
            label_batch = self.labels[self.start:(self.start + batch_size)];
            self.start += batch_size
        else:
            data_batch = self.data[self.start:]
            label_batch = self.labels[self.start:];
            self.start = 0
            self.epoch += 1

        embedding_batch = self.data2embedding(data_batch);
        print('data_batch={}\tlabel_batch={}'.format(data_batch.shape, label_batch.shape, ));
        return embedding_batch, label_batch;

    def iter_batches(self, batch_size=128):
        num_rows = self.data.shape[0];
        print('num_rows={}'.format(num_rows ));
        end = 0

        for i in range(num_rows // batch_size):
            start = i * batch_size
            end = (i + 1) * batch_size

            data_batch = self.data[start:end];
            label_batch = self.labels[start:end];
            embedding_batch = self.data2embedding(data_batch);
            print('data_batch={}\tlabel_batch={}\tembedding_batch={}'.format(data_batch.shape, label_batch.shape, embedding_batch.shape ));
            yield embedding_batch, label_batch;

        if end != num_rows:
            data_batch = self.data[end:];
            label_batch = self.labels[end:];
            embedding_batch = self.data2embedding(data_batch);
            print('data_batch={}\tlabel_batch={}\tembedding_batch={}'.format(data_batch.shape, label_batch.shape, embedding_batch.shape ));
            yield embedding_batch, label_batch;

    def data2embedding(self, data_batch):
        sample_num, sentence_len = data_batch.shape;
        embedding_dim = self.embedding_matrix.shape[1];

        embedding_batch = np.zeros((sample_num, sentence_len, embedding_dim));
        for sample_idx in range(sample_num):
            for word_idx in range(sentence_len):
                word = data_batch[sample_idx, word_idx];
                if word == 0: continue;
                embedding_batch[sample_idx, word_idx, :] = self.embedding_matrix[word];
        # print('embedding_batch={}'.format(embedding_batch.shape ));
        return embedding_batch;


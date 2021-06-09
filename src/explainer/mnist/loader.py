
import numpy as np

class Loader(object):
    """A loader class designed to help provide batches one at a time in random order during training, or in the same order all at once when evaluating results."""

    def __init__(self, data, labels, shuffle=False ):
        self.start = 0
        self.epoch = 0
        self.data = data;
        self.labels = labels;
        assert len(self.data) == len(self.labels);

        if shuffle:
            self.r = list(range(len(data)))
            np.random.shuffle(self.r)
            self.data = self.data[self.r];
            self.labels = self.labels[self.r];

    def full_batch(self):
        return self.data, self.labels;

    def next_batch(self, batch_size=512):
        num_rows = len(self.data);
        print('num_rows={}'.format(num_rows ));

        if self.start + batch_size < num_rows:
            data_batch = self.data[self.start:(self.start + batch_size)];
            label_batch = self.labels[self.start:(self.start + batch_size)];
            self.start += batch_size
        else:
            data_batch = self.data[self.start:];
            label_batch = self.labels[self.start:];
            self.start = 0
            self.epoch += 1

        print('data_batch={}\tlabel_batch={}'.format(data_batch.shape, label_batch.shape ));
        return data_batch, label_batch;

    def iter_batches(self, batch_size=512):
        num_rows = len(self.data);
        print('num_rows={}'.format(num_rows ));
        end = 0

        for i in range(num_rows // batch_size):
            start = i * batch_size
            end = (i + 1) * batch_size

            data_batch = self.data[start:end];
            label_batch = self.labels[start:end];
            print('data_batch={}\tlabel_batch={}'.format(data_batch.shape, label_batch.shape ));
            yield data_batch, label_batch

        if end != num_rows:
            data_batch = self.data[end:];
            label_batch = self.labels[end:];
            print('data_batch={}\tlabel_batch={}'.format(data_batch.shape, label_batch.shape ));
            yield data_batch, label_batch

    def restore_order(self, data):
        """
        Since the data is randomly shuffled at initialization, this helper can return it to its original order if necessary.

        :param data: array_like of size (N,D)
        """
        data_out = np.zeros_like(data)
        for i, j in enumerate(self.r):
            data_out[j] = data[i]
        return data_out


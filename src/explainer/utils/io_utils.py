import os, sys, numbers, random, json
import numpy as np


def load_mnist(DATA_PATH, raw=False):
    """Loads MNIST dataset from `DATA_PATH` or downloads it if necessary.

    :param raw: `True` if no preprocessing should be applied to the data. Otherwise, data is normalized to 1.
    :type raw: `bool`
    :return: `(x_train, y_train), (x_test, y_test), min, max`
    :rtype: `(np.ndarray, np.ndarray), (np.ndarray, np.ndarray), float, float`
    """

    path = get_file('mnist.npz', path=DATA_PATH, url='https://s3.amazonaws.com/img-datasets/mnist.npz')

    f = np.load(path)
    x_train = f['x_train']
    y_train = f['y_train']
    x_test = f['x_test']
    y_test = f['y_test']
    f.close()

    # Add channel axis
    min_, max_ = 0, 255
    if not raw:
        min_, max_ = 0., 1.
        x_train = np.expand_dims(x_train, axis=3)
        x_test = np.expand_dims(x_test, axis=3)
        x_train, y_train = preprocess(x_train, y_train)
        x_test, y_test = preprocess(x_test, y_test)

    return (x_train, y_train), (x_test, y_test), min_, max_


def _extract(full_path, path):
    import tarfile
    import zipfile
    import shutil

    if full_path.endswith('tar'):
        if tarfile.is_tarfile(full_path):
            archive = tarfile.open(full_path, "r:")
    elif full_path.endswith('tar.gz'):
        if tarfile.is_tarfile(full_path):
            archive = tarfile.open(full_path, "r:gz")
    elif full_path.endswith('zip'):
        if zipfile.is_zipfile(full_path):
            archive = zipfile.ZipFile(full_path)
        else:
            return False
    else:
        return False

    try:
        archive.extractall(path)
    except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
        if os.path.exists(path):
            if os.path.isfile(path):
                os.remove(path)
            else:
                shutil.rmtree(path)
        raise
    return True


def get_file(filename, url, path, extract=False):
    """
    Downloads a file from a URL if it not already in the cache. The file at indicated by `url` is downloaded to the
    path `path` (default is ~/.art/data). and given the name `filename`. Files in tar, tar.gz, tar.bz, and zip formats
    can also be extracted. This is a simplified version of the function with the same name in Keras.

    :param filename: Name of the file.
    :type filename: `str`
    :param url: Download URL.
    :type url: `str`
    :param path: Folder to store the download. If not specified, `~/.art/data` is used instead.
    :type: `str`
    :param extract: If true, tries to extract the archive.
    :type extract: `bool`
    :return: Path to the downloaded file.
    :rtype: `str`
    """

    if extract:
        extract_path = os.path.join(path, filename)
        full_path = extract_path + '.tar.gz'
    else:
        full_path = os.path.join(path, filename)

    # Determine if dataset needs downloading
    download = not os.path.exists(full_path)

    if download:
        print('Downloading data from %s', url)
        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                from six.moves.urllib.error import HTTPError, URLError
                from six.moves.urllib.request import urlretrieve

                urlretrieve(url, full_path)
            except HTTPError as e:
                raise Exception(error_msg.format(url, e.code, e.msg))
            except URLError as e:
                raise Exception(error_msg.format(url, e.errno, e.reason))
        except (Exception, KeyboardInterrupt):
            if os.path.exists(full_path):
                os.remove(full_path)
            raise

    if extract:
        if not os.path.exists(extract_path):
            _extract(full_path, path)
        return extract_path

    return full_path


def preprocess(x, y, nb_classes=10, max_value=255):
    """Scales `x` to [0, 1] and converts `y` to class categorical confidences.

    :param x: Data instances
    :type x: `np.ndarray`
    :param y: Labels
    :type y: `np.ndarray`
    :param nb_classes: Number of classes in dataset
    :type nb_classes: `int`
    :param max_value: Original maximum allowed value for features
    :type max_value: `int`
    :return: rescaled values of `x`, `y`
    :rtype: `tuple`
    """
    x = x.astype('float32') / max_value
    y = to_categorical(y, nb_classes)

    return x, y


def to_categorical(labels, nb_classes=None):
    """
    Convert an array of labels to binary class matrix.

    :param labels: An array of integer labels of shape `(nb_samples,)`
    :type labels: `np.ndarray`
    :param nb_classes: The number of classes (possible labels)
    :type nb_classes: `int`
    :return: A binary matrix representation of `y` in the shape `(nb_samples, nb_classes)`
    :rtype: `np.ndarray`
    """
    labels = np.array(labels, dtype=np.int32)
    if not nb_classes:
        nb_classes = np.max(labels) + 1
    categorical = np.zeros((labels.shape[0], nb_classes), dtype=np.float32)
    categorical[np.arange(labels.shape[0]), np.squeeze(labels)] = 1
    return categorical



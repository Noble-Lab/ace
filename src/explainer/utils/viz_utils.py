
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

COLORS = ['red', 'green', 'blue', 'orange', 'black', 'cyan', 'magenta', '0.7', 'yellow', 'gray'];
MARKS = ['o', 'x', 'd', 'v', '^', 's', 'p', 'P', '*', '.'];

LINESTYLES = ['-', '--', '-.', ':', (0, (1, 10)), (0, (5, 10)), (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5, 1, 5)), (0, (3, 1, 1, 1, 1, 1)) ];


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=["black", "white"], threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def create_image_sprite(images, normalize=True):
    """
    Creates a sprite of provided images.

    :param images: Images to construct the sprite.
    :type images: `np.array`
    :param normalize: whether to divided by the maximum value per image.
    :type normalize: `bool`
    :return: An image array containing the sprite.
    :rtype: `np.ndarray`
    """

    shape = np.shape(images);

    if len(shape) < 3 or len(shape) > 4:
        raise ValueError('Images provided for sprite have wrong dimensions ' + str(len(shape)));

    if len(shape) == 3:
        # Check to see if it's mnist type of images and add axis to show image is gray-scale
        images = np.expand_dims(images, axis=3);
        shape = np.shape(images);

    if normalize: images /= np.max(np.fabs(images));

    # Change black and white images to RGB
    if shape[3] == 1:
        images = convert_to_rgb(images)

    n = int(np.ceil(np.sqrt(images.shape[0])))
    padding = ((0, n ** 2 - images.shape[0]), (0, 0), (0, 0)) + ((0, 0),) * (images.ndim - 3)
    images = np.pad(images, padding, mode='constant', constant_values=0)

    # Tile the individual thumbnails into an image
    images = images.reshape((n, n) + images.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, images.ndim + 1)))
    images = images.reshape((n * images.shape[1], n * images.shape[3]) + images.shape[4:])
    sprite = (images * 255).astype(np.uint8)

    return sprite


def create_score_sprite(scores, normalize=True):
    shape = np.shape(scores);
    assert len(shape) == 4;
    if normalize: scores /= np.max(np.fabs(scores));

    n = int(np.ceil(np.sqrt(scores.shape[0])));
    padding = ((0, n ** 2 - scores.shape[0]), (0, 0), (0, 0)) + ((0, 0),) * (scores.ndim - 3);
    scores = np.pad(scores, padding, mode='constant', constant_values=0);

    scores = scores.reshape((n, n) + scores.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, scores.ndim + 1)))
    scores = scores.reshape((n * scores.shape[1], n * scores.shape[3]) + scores.shape[4:])
    return np.squeeze(scores);


def convert_to_rgb(images):
    """
    Converts grayscale images to RGB. It changes NxHxWx1 to a NxHxWx3 array, where N is the number of figures,
    H is the high and W the width.

    :param images: Grayscale images of shape (NxHxWx1).
    :type images: `np.ndarray`
    :return: Images in RGB format of shape (NxHxWx3).
    :rtype: `np.ndarray`
    """
    s = np.shape(images)
    if not ((len(s) == 4 and s[-1] == 1) or len(s) == 3):
        raise ValueError('Unexpected shape for grayscale images:' + str(s))

    if s[-1] == 1:
        # Squeeze channel axis if it exists
        rgb_images = np.squeeze(images, axis=-1)
    else:
        rgb_images = images
    rgb_images = np.stack((rgb_images,) * 3, axis=-1)

    return rgb_images


def save_image(image, full_path, cmap):
    """
    Saves image into a file inside `DATA_PATH` with the name `filename`.
    :param image: Image to be saved
    :type image: `np.ndarray`
    :param filename: File name containing extension e.g., my_img.jpg, my_img.png, my_images/my_img.png
    :type filename: `str`
    :return: `None`
    """

    # from PIL import Image
    # im = Image.fromarray(image);
    # im.save(full_path)
    # print('Image saved to %s.', full_path)

    from matplotlib import pyplot as plt
    print('image shape: {}'.format(image.shape));
    # assert len(image.shape) <= 3;

    # if cmap == '': plt.imsave(full_path, image);
    # else: plt.imsave(full_path, image, cmap=cmap);
    plt.imsave(full_path, image, cmap=cmap);

    # plt.imsave(full_path, image.reshape(image.shape[0], image.shape[1]), cmap='viridis');
    # plt.imsave(full_path, image.reshape(image.shape[0], image.shape[1]), cmap='plasma');
    # plt.imsave(full_path, image.reshape(image.shape[0], image.shape[1]), cmap='plasma');
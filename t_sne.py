print(__doc__)
from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold)
import sys

sys.dont_write_bytecode = True

viz_dir = './output/graph'

#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X_tsne,images, labels, iter,  title=None):
    '''

    :param X_tsne:
    :param images:
    :param labels:
    :param iter:
    :param title:
    :return:
    '''
    x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
    X_tsne = (X_tsne - x_min) / (x_max - x_min)

    # change main figure size
    fig = plt.figure(figsize=(25,25))

    # change sub plot size
    ax = fig.add_subplot(1,1,1)
    for i in range(X_tsne.shape[0]):
        plt.text(X_tsne[i, 0], X_tsne[i, 1], str(labels[i]),
                 color=plt.cm.Set1(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 30})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X_tsne.shape[0]):
            dist = np.sum((X_tsne[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X_tsne[i]]]
            # imagebox = offsetbox.AnnotationBbox(
            #     offsetbox.OffsetImage(images[i], cmap=plt.cm.gray_r),
            #     X_tsne[i])
            # ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        fig = plt.suptitle(title, fontsize=50)
    plt.savefig(viz_dir + str(iter) + 't_net.png')



def t_sne_fit(logits):
    # t-SNE embedding of the digits dataset
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()

    X_tsne = tsne.fit_transform(logits)

    print("time elapsed:{}".format(time() - t0))
    return(X_tsne)
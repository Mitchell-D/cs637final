import numpy as np
from scipy.io import loadmat

def preprocess(X:np.array, Y:np.array, seed=None, gain=500, offset=1000,
               cast_to_onehot=True):
    """
    """
    ## Collapse the spatial dimension
    X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[-1]))
    Y = np.reshape(Y, Y.shape[0]*Y.shape[1])
    ## Scale to W cm^-2 nm^-1 sr^-1
    X = (X-offset)/gain
    ## One-hot encode the true vectors
    if cast_to_onehot:
        Y = np.stack([np.where(Y==s, 1, 0) for s in np.unique(Y)], axis=-1)
    return X,Y

def random_permutation(size, seed=None):
    ## Shuffle the pixel dimension
    rng = np.random.default_rng(seed=seed)
    forward = np.arange(size)
    rng.shuffle(forward)
    backward = list(zip(*sorted(enumerate(forward), key=lambda s:s[1])))[0]
    return forward,np.asarray(backward)

def ratio_sample(label_ints:np.array, ratio, seed=None):
    """
    Given an array of integers identifying class labels, return a boolean
    mask identifying samples randomly chosen to fit the constraints.

    :@param label_ints: 1D numpy array of integer labels
    :@param ratio: Percentage of each class to return
    :@param seed: random seed to apply when selecting valid members

    :@return: Boolean mask identifying samples to include
    """
    unique_labels,counts = np.unique(label_ints, return_counts=True)
    forward,backward = random_permutation(label_ints.size, seed=seed)
    ## reversibly shuffle the labels
    labels_ints = label_ints[forward]
    ratio_counts = np.floor(ratio*counts)
    for i,l in enumerate(unique_labels):
        np.where(label_ints==l)[:i]


def uniform_sample(label_ints:np.array, nsamples=None, seed=None):
    """
    Given an array of integers identifying class labels, return

    :@param label_ints: Numpy array of integer labels
    :@param nsamples: Maximum number of samples to draw from each category.
        If any classes have fewer samples available than the provided amount,
        the included mask will include all of them.
    """
    pass

def gaussnorm(X, bulk_norm=False):
    """
    Normalize the array to a unit gaussian distribution, by default per feature
    on the final axis, but optionally normalized over the whole array

    :@param X: Array to normalize
    :@param bulk_norm: If True, normalizes using the full array's mean and
        standard deviation rather than per final axis element.
    """
    ax = [0,None][bulk_norm]
    means = np.average(X, axis=ax)
    stdevs = np.std(X, axis=ax)
    Xnorm = (X-means)/stdevs
    return Xnorm,means,stdevs

def ints_to_masks(class_ints):
    """
    Convert a (M,N) shaped array of L unique class identifiers to a
    (M,N,L) shaped boolean array and a list of the corresponding labels.
    """
    int_labels = np.unique(class_ints)
    masks = np.stack([class_ints==l for l in int_labels], axis=-1)
    return int_labels,masks

if __name__=="__main__":
    ## (M,N,F) array for F bands
    X = loadmat("./data/indian-pines.mat")["indian_pines"]
    ## (M,F) array of integer classes
    Y = loadmat("./data/indian-pines-truth.mat")["indian_pines_gt"]
    ## Construct a coordinate array of equally-spaced wavelengths
    wl_min,wl_max = (.4, 2.5) ## define wavelength range
    wls = np.linspace(.4,2.5,X.shape[-1])

    ## Reshape and rescale the feature data, and one-hot encode the labels
    X,Y = preprocess(X, Y, gain=500, offset=1000)
    X_norm,means,stdevs = gaussnorm(X)

    ## Get integer arrays encoding a seeded random permutation and its inverse
    forward,backward = random_permutation(X.shape[0])

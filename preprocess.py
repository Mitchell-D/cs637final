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

def gaussnorm(X):
    """ Normalize the array to a unit gaussian distribution """
    means = np.average(X, axis=0)
    stdevs = np.std(X, axis=0)
    Xnorm = (X-means)/stdevs
    return Xnorm,means,stdevs

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



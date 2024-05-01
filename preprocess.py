import numpy as np
from scipy.io import loadmat

def preprocess(X,Y, bulk_norm=False, sample_ratio=None, sample_max=None,
               cast_to_onehot=True, gain=500, offset=1000, seed=None):
    """ """
    unique_labels = np.unique(Y)
    ## conversion to spectral radiance (W * cm^-2 * nm^-1 * sr^-1) from docs
    X = (X-offset)/gain

    ## Normalize to mean 0 and stdev 1. if bulk_norm is True in the config,
    ## the whole array will be normalized by a single linear equation rather
    ## than each band being normalized independently. This is a hyperparameter,
    ## but some research suggests bulk normalization is a good idea in order
    ## to preserve "spectral angle" (see spectral angle mapping)
    IDX = np.arange(X.shape[0])
    X,means,stdevs = gaussnorm(X=(X_nonorm:=X), bulk_norm=bulk_norm)

    ## Get integer arrays encoding a seeded random permutation and its inverse
    forward,backward = random_permutation(
            X.shape[0], seed=seed)
    X,Y,IDX = X[forward],Y[forward],IDX[forward]

    ## Get a (L,N) shaped mask that, for each of the L classes, marks a subset
    ## of the N total samples to True, such that the provided ratio of
    ## total samples from each class is returned.
    if sample_ratio:
        train_masks,val_masks = ratio_sample(
                label_ints=Y,
                ratio=sample_ratio,
                sample_max=sample_max,
                seed=seed,
                )
    if sample_max:
        ## Saturate at sample_max if requested
        train_masks,val_masks = ratio_sample(
                label_ints=Y,
                ratio=1.,
                sample_max=sample_max,
                seed=seed,
                )
    ## If neither sample constraints provided, just return the whole dataset.
    if all(c is None for c in (sample_ratio,sample_max)):
        if cast_to_onehot:
            Y = np.stack([np.where(Y==s,1,0) for s in unique_labels], axis=-1)
        return (X[backward],Y[backward],IDX[backward]),(means,stdevs)

    TIDX = IDX[np.any(train_masks, axis=0)]
    VIDX = IDX[np.any(val_masks, axis=0)]

    ## Apply each mask and collect all the samples
    TY,TX = zip(*[
        (Y[train_masks[i]], X[train_masks[i]])
        for i in range(train_masks.shape[0])
        ])
    VY,VX = zip(*[
        (Y[val_masks[i]], X[val_masks[i]])
        for i in range(val_masks.shape[0])
        ])

    ## Re-join training and validation samples from each category
    TY = np.concatenate(TY, axis=0)
    TX = np.concatenate(TX, axis=0)
    VY = np.concatenate(VY, axis=0)
    VX = np.concatenate(VX, axis=0)

    ## Independently permute the training and validation samples again
    ## since they were separated by classes.
    tforward,_ = random_permutation(TY.shape[0],seed=seed)
    TY,TX,TIDX = TY[tforward],TX[tforward],TIDX[tforward]
    vforward,_ = random_permutation(VY.shape[0],seed=seed)
    VY,VX,VIDX = VY[vforward],VX[vforward],VIDX[vforward]

    ## One-hot encode the labels
    if cast_to_onehot:
        TY = np.stack([np.where(TY==s, 1, 0) for s in unique_labels], axis=-1)
        VY = np.stack([np.where(VY==s, 1, 0) for s in unique_labels], axis=-1)

    #print(Y.shape, X.shape, IDX.shape)
    #print(TY.shape, TX.shape, TIDX.shape)
    #print(VY.shape, VX.shape, VIDX.shape)
    return (TX,TY,TIDX),(VX,VY,VIDX),(means,stdevs)

def random_permutation(size, seed=None):
    ## Shuffle the pixel dimension
    rng = np.random.default_rng(seed=seed)
    forward = np.arange(size)
    rng.shuffle(forward)
    backward = list(zip(*sorted(enumerate(forward), key=lambda s:s[1])))[0]
    return forward,np.asarray(backward)

def ratio_sample(label_ints:np.array, ratio, sample_max=None, seed=None):
    """
    Given an array of integers identifying class labels, return a boolean
    mask identifying samples randomly chosen to fit the constraints.

    :@param label_ints: 1D numpy array of integer labels
    :@param ratio: Percentage of each class to return
    :@param sample_max: Integer maximum number of samples to allow
        each class to have (to modulate unbalanced class contribution)
    :@param seed: random seed to apply when selecting valid members

    :@return: Boolean mask identifying samples to include
    """
    assert len(label_ints.shape)==1, "label_ints must be a 1D array"
    unique_labels,counts = np.unique(label_ints, return_counts=True)
    forward,_ = random_permutation(label_ints.size, seed=seed)
    ## reversibly shuffle the labels
    labels_ints = label_ints[forward]
    ratio_counts = np.floor(ratio*counts).astype(int)
    if sample_max:
        ratio_counts[ratio_counts>sample_max] = sample_max
    in_masks = np.full((len(unique_labels), label_ints.size), False)
    out_masks = np.full((len(unique_labels), label_ints.size), False)
    for i,l in enumerate(unique_labels):
        idxs = np.where(label_ints==l)[0]
        in_masks[i,idxs[:ratio_counts[i]]] = True
        out_masks[i,idxs[ratio_counts[i]:]] = True
    return in_masks,out_masks

def uniform_sample(label_ints:np.array, nsamples=None, seed=None):
    """
    Given an array of integers identifying class labels, return

    :@param label_ints: Numpy array of integer labels
    :@param nsamples: Maximum number of samples to draw from each category.
        If any classes have fewer samples available than the provided amount,
        the included mask will include all of them.
    """
    rng = np.random.default_rng(seed=seed)
    print(label_ints.shape, nsamples)

def gaussnorm(X, bulk_norm=False, stdev_cutoff=10):
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
    np.clip(Xnorm, -stdev_cutoff, stdev_cutoff)
    return Xnorm,means,stdevs

def pca(X:np.ndarray, print_table:bool=False):
    """
    Perform principle component analysis on the provided array, and return
    the transformed array of principle components
    """
    flatX = np.copy(X).transpose(2,0,1).reshape(X.shape[2],-1)
    # Get a vector of the mean value of each band
    means = np.mean(flatX, axis=0)
    # Get a bxb covariance matrix for b bands
    covs = np.cov(flatX)
    # Calculate and sort eigenvalues and eigenvectors
    eigen = list(np.linalg.eig(covs))
    eigen[1] = list(map(list, eigen[1]))
    eigen = list(zip(*eigen))
    eigen.sort(key=lambda e: e[0])
    evals, evecs = zip(*eigen)
    # Get a diagonal matrix of eigenvalues
    transform = np.dstack(evecs).transpose().squeeze()
    Y = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i,j,:] = np.dot(transform, X[i,j,:])
    if print_table:
        cov_string = ""
        ev_string = ""
        for i in range(covs.shape[0]):
            cov_string+=" & ".join(
                    [f"{x:.4f}" for x in covs[i,:]])
            ev_string+=f"{evals[i]:.4f}"+" & "+" & ".join(
                    [f"{x:.4f}"for x in evecs[i]])
            ev_string += " \\\\ \n"
            cov_string += " \\\\ \n"
        print("Covariance matrix:")
        print(cov_string)
        print("Eigenvalue and Eigenvector table:")
        print(ev_string)

    return Y

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

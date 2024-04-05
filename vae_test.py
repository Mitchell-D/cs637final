import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat

#import tracktrain
from preprocess import preprocess,gaussnorm,random_permutation
from krttdkit.visualize import guitools as gt

plt.rcParams.update({'font.size':16, })

default_labels_colors = [
    ("Unknown", "white"),
    ("Alfalfa", "xkcd:greenish blue"),
    ("Corn-notill", "xkcd:light pink"),
    ("Corn-mintill", "xkcd:light purple"),
    ("Corn", "xkcd:violet"),
    ("Grass-pasture", "xkcd:grass green"),
    ("Grass-trees", "xkcd:leaf green"),
    ("Grass-pasture-mowed", "xkcd:light green"),
    ("Hay-windrowed", "xkcd:purple blue"),
    ("Oats", "xkcd:medium blue"),
    ("Soybean-notill", "xkcd:rose pink"),
    ("Soybean-mintill", "xkcd:orange red"),
    ("Soybean-clean", "xkcd:deep red"),
    ("Wheat", "xkcd:tangerine"),
    ("Woods", "xkcd:forest green"),
    ("Buildings-Grass-Trees-Drives", "gray"),
    ("Stone-Steel-Towers", "black"),
    ]

seed = 200007221752
rng = np.random.default_rng(seed=seed)

def plot_classes(wavelengths,class_intensities,class_err=None,show=False):
    """
    :@param wavelengths: array of wavelengths labeling the common x axis
    :@param class_intensities:list of arrays corresponding to each class'
        intensity values. Each array should be the same size as wavelengths
    :@param class_err: Optionally include an error metric like 1st stddev
        to shade a region surrounding the class trend.
    """
    fig,ax = plt.subplots()
    labels = default_labels_colors
    assert len(class_intensities) == len(labels)
    for i in range(class_intensities.shape[0]):
        ax.plot(wavelengths, class_intensities[i], label=labels[i][0],
                color=labels[i][1], linewidth=2)
        if not class_err is None:
            ax.fill_between(
                    wls,
                    class_intensities[i]-class_err[i]/2,
                    class_intensities[i]+class_err[i]/2,
                    color=labels[i][1],
                    alpha=.15
                    )
    ax.set_title("Gauss-normalized spectral intensity with 1-sigma error bars")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Intensity")
    ax.legend(prop={'size':14}, ncol=2)
    plt.yscale("linear")
    if show:
        plt.show()
    return fig,ax

def plot_sample(wavelengths, intensities, err=None,
                xlabel="Wavelength ($um$)", ylabel="Reflectance (%)",
                plot_spec:dict={}, fig_path:Path=None, show=False):
    """
    """
    ps = {"xlabel":xlabel, "ylabel":ylabel,
      "trend_color":"red", "trend_width":3, "line_width":2,
      "cmap":"nipy_spectral", "text_size":18, "title":"","fill_alpha":1.,
      "norm":"linear", "logx":False,"figsize":(12,12)}
    ps.update(plot_spec)

    #plt.clf()
    plt.rcParams.update({"font.size":ps["text_size"]})

    fig,ax = plt.subplots()
    ax.set_title(ps.get("title"))
    ax.set_xlabel(ps.get("xlabel"))
    ax.set_ylabel(ps.get("ylabel"))
    if not err is None:
        ax.fill_between(wavelengths, intensities-err, intensities+err,
                        alpha=ps.get("fill_alpha"))
    ax.plot(wavelengths, intensities, linewidth=ps.get("line_width"),
            color="red", zorder=100)
    if show:
        plt.show()
    if not fig_path is None:
        fig.set_size_inches(*ps.get("figsize"))
        fig.savefig(fig_path.as_posix(), bbox_inches="tight",dpi=80)
    return fig,ax

def ints_to_masks(class_ints):
    """  """
    int_labels = np.unique(class_ints)
    return int_labels,np.stack([class_ints==l for l in int_labels], axis=-1)

if __name__=="__main__":
    ## (M,N,F) array for F bands
    X = loadmat("./data/indian-pines.mat")["indian_pines"]
    ## (M,F) array of integer classes
    Y = loadmat("./data/indian-pines-truth.mat")["indian_pines_gt"]
    ## Construct a coordinate array of equally-spaced wavelengths
    wl_min,wl_max = (.4, 2.5) ## define wavelength range
    wls = np.linspace(.4,2.5,X.shape[-1])

    gt.quick_render(np.dstack([X[...,b] for b in (88,54,24)]))

    exit(0)

    X,Y = preprocess(X, Y, gain=500, offset=1000, cast_to_onehot=False)
    Xnorm,means,stdevs = gaussnorm(X)

    ## Plot
    fig,ax = plot_sample(wls, means, err=stdevs, show=False)

    int_labels,masks = ints_to_masks(Y)

    classes = [Xnorm[masks[...,i]] for i in range(masks.shape[-1])]
    class_means,class_stdevs = map(np.asarray, zip(*[
            (np.average(c, axis=0), np.std(c, axis=0))
            for c in classes
            ]))

    plot_classes(wls,class_means,class_stdevs,show=True)

    for i in range(10):
        plot_sample(
                wavelengths=wls,
                intensities=Xnorm[i],
                show=True,
                plot_spec={
                    "fill_alpha":.2,
                    },
                )

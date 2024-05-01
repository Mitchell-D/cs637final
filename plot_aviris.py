import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolo
from matplotlib.patches import Patch
from pathlib import Path
from scipy.io import loadmat

#import tracktrain
from preprocess import preprocess,gaussnorm,random_permutation
from krttdkit.visualize import guitools as gt
from metadata import default_labels_colors


#plt.rcParams.update({'font.size':16, })

seed = 200007221752
rng = np.random.default_rng(seed=seed)

def plot_classes(class_array:np.ndarray, class_labels:list, colors:list=None,
                 fig_path:Path=None, show:bool=False, plot_spec:dict={}):
    """
    Plots an integer array mapping pixels to a list of class labels

    :@param class_array: 2d integer array such that integer values are the
        indeces of the corresponding class label and color.
    :@param fig_path: Path to generated figure
    :@param class_labels: string labels indexed by class array values.
    :@param colors: List of 3-element [0,1] float arrays for RGB values.
    """
    old_ps = {"fig_size":(12,12), "fontsize":8}
    old_ps.update(plot_spec)
    plot_spec = old_ps
    if colors is None:
        colors = [[i/len(class_labels),
                   .5+(len(class_labels)-i)/(2*len(class_labels)),
                   .5+i/(2*len(class_labels))]
                  for i in range(len(class_labels))]
        colors = [ mcolo.hsv_to_rgb(c) for c in colors ]
    assert len(colors)==len(class_labels)
    cmap, norm = mcolo.from_levels_and_colors(
            list(range(len(colors)+1)), colors)
    im = plt.imshow(class_array, cmap=cmap, norm=norm, interpolation="none")
    handles = [ Patch(label=class_labels[i], color=colors[i])
               for i in range(len(class_labels)) ]
    plt.legend(handles=handles, fontsize=plot_spec.get("fontsize"))
    plt.tick_params(axis="both", which="both", labelbottom=False,
                    labelleft=False, bottom=False, left=False)
    fig = plt.gcf()
    fig.set_size_inches(*plot_spec.get("fig_size"))
    plt.title(plot_spec.get("title"))
    if fig_path:
        print(f"saving figure as {fig_path.as_posix()}")
        plt.savefig(fig_path, bbox_inches="tight", dpi=plot_spec.get("dpi"))
    if show:
        plt.show()

def plot_class_spectra(
        wavelengths, class_intensities, labels,
        colors=None, class_err=None, show=False,
        ):
    """
    :@param wavelengths: array of wavelengths labeling the common x axis
    :@param class_intensities:list of arrays corresponding to each class'
        intensity values. Each array should be the same size as wavelengths
    :@param class_err: Optionally include an error metric like 1st stddev
        to shade a region surrounding the class trend.
    """
    fig,ax = plt.subplots()
    assert len(class_intensities) == len(labels)
    if not colors:
        colors = [None for i in range(len(labels))]
    for i in range(class_intensities.shape[0]):
        ax.plot(wavelengths, class_intensities[i], label=labels[i],
                color=colors[i], linewidth=2)
        if not class_err is None:
            ax.fill_between(
                    wls,
                    class_intensities[i]-class_err[i]/2,
                    class_intensities[i]+class_err[i]/2,
                    color=colors[i],
                    alpha=.15
                    )
    ax.set_title("Gauss-normalized spectral intensity with 1-sigma error bars")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Intensity")
    ax.legend(prop={'size':8}, ncol=2)
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
    bulk_norm = False

    ## (M,N,F) array for F bands
    #X = loadmat("./data/indian-pines.mat")["indian_pines"]
    X = loadmat("./data/indian-pines-corrected.mat")["indian_pines_corrected"]
    ## (M,F) array of integer classes
    Y = loadmat("./data/indian-pines-truth.mat")["indian_pines_gt"]

    ## Construct a coordinate array of equally-spaced wavelengths
    wl_min,wl_max = (.4, 2.5) ## define wavelength range
    wls = np.linspace(.4,2.5,X.shape[-1])

    ## Print the per-class counts
    class_labels, colors = zip(*default_labels_colors)
    _,counts = np.unique(Y, return_counts=True)
    print("\n".join(f"{l:<32}  {c}" for l,c in zip(class_labels,counts)))

    ## Plot the spatial distribution of truth labels
    plot_classes(Y, class_labels, colors, show=True)

    ## Preprocess (unroll and scale) and gauss-normalize the data
    X,Y = preprocess(X, Y, gain=500, offset=1000, cast_to_onehot=False)

    ## Make boolean masks for each class and use them to reference pixel groups
    int_labels,masks = ints_to_masks(Y)

    classes = [X[masks[...,i]] for i in range(masks.shape[-1])]
    class_means,class_stdevs = map(np.asarray, zip(*[
            (np.average(c, axis=0), np.std(c, axis=0))
            for c in classes
            ]))
    ## Plot the spectral means/stdevs of all of the classes together
    plot_class_spectra(
            wavelengths=wls,
            class_intensities=class_means,
            labels=class_labels,
            colors=colors,
            class_err=class_stdevs,
            show=True,
            )

    X_norm,means,stdevs = gaussnorm(X, bulk_norm=bulk_norm)
    ## Plot feature-wise reflectance if bulk normalization wasn't used.
    if means.size>1:
        fig,ax = plot_sample(wls, means, err=stdevs, show=True)
    classes_normed = [X_norm[masks[...,i]] for i in range(masks.shape[-1])]
    class_means_normed,class_stdevs_normed = map(np.asarray, zip(*[
            (np.average(c, axis=0), np.std(c, axis=0))
            for c in classes_normed
            ]))

    ## Plot the spectral means/stdevs of all of the classes together
    plot_class_spectra(
            wavelengths=wls,
            class_intensities=class_means_normed,
            labels=class_labels,
            colors=colors,
            class_err=class_stdevs_normed,
            show=True,
            )

    exit(0)

    for i in range(10):
        plot_sample(
                wavelengths=wls,
                intensities=X_norm[i],
                show=True,
                plot_spec={
                    "fill_alpha":.2,
                    },
                )

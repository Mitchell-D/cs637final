import numpy as np
import os
import traceback
##os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import pickle as pkl
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from pathlib import Path
from scipy.io import loadmat
from pprint import pprint
from PIL import Image
import matplotlib.pyplot as plt

import tracktrain.model_methods as mm
from tracktrain.VariationalEncoderDecoder import VariationalEncoderDecoder
from tracktrain.ModelDir import ModelDir,ModelSet
from tracktrain.compile_and_train import train

import preprocess as pp # preprocess,gaussnorm,random_permutation,ints_to_masks
from metadata import indian_pines_labels,default_labels_colors
from plot_aviris import plot_classes

if __name__=="__main__":
    model_parent_dir = Path("data/models")
    fig_dir = Path("figures/predictions")

    MS = ModelSet.from_dir(model_parent_dir)
    plot_models = list(range(40))
    MS = ModelSet([
        md for md in MS.model_dirs
        if any(str(s) in md.dir.name for s in plot_models)
        ])
    '''
    MS.scatter_metrics(
            xmetric="loss",
            ymetric="val_loss",
            show=True,
            use_notes=False,
            plot_spec={
                "ylim":(0,2),
                "xlim":(0,2),
                "facecolor":"xkcd:dark grey",
                "legend_cols":1,
                "cmap":"nipy_spectral",
                "xlabel":"Test Loss",
                "ylabel":"Validation Loss",
                "title":"Indian Pines classifier training and validation loss",
                }
            )
    '''
    plot_models = ["ff"]
    MS.plot_metrics(
            metrics=("loss","val_loss"),
            show=True,
            use_notes=False,
            plot_spec={
                "ylim":(0,2),
                "facecolor":"xkcd:dark grey",
                "legend_cols":4,
                "cmap":"nipy_spectral",
                "xlabel":"Epoch",
                "ylabel":"Binary Crossentropy",
                "title":"Indian Pines feedforward classifier learning curves",
                },
            )
    #'''
    exit(0)


    ## (M,N,F) array for F bands
    X = loadmat("./data/indian-pines.mat")["indian_pines"]
    ## (M,F) array of integer classes
    Y = loadmat("./data/indian-pines-truth.mat")["indian_pines_gt"]

    str_labels = indian_pines_labels[1:]

    ## Reshape to:
    ## Y := (S,) for S labels for each sample (integers in [0,16] )
    ## X := (S,F) for S samples and F features (radiance bands)
    grid_shape = X.shape[:2]
    labels = np.reshape(Y, (Y.shape[0]*Y.shape[1],))
    features = np.reshape(X, (Y.size, X.shape[-1]))
    UNK = (labels == 0)

    check_substr = ("ff-0",)
    check_models = [m for m in MS.model_dirs
                    if any(s in m.dir.name for s in check_substr)]
    for md in check_models:
        print(f"Loading {md.dir.name}")
        model = md.load_weights()
        #tf.keras.utils.plot_model(
        #        model, to_file=md.dir.joinpath(f"{md.name}.png"),
        #        show_shapes=True, show_layer_names=True, expand_nested=True,
        #       show_layer_activations=True)
        (X,Y,IDX),(means,stdevs) = pp.preprocess(
                X=features, Y=labels, bulk_norm=md.config["bulk_norm"],
                cast_to_onehot=False)

        P = np.argmax(model(X), axis=-1)
        P_UNK = P[UNK]
        P += 1 ## index up for unknown dimension
        Y = np.reshape(Y, grid_shape)
        P = np.reshape(P, grid_shape)

        class_labels,colors = zip(*default_labels_colors)
        '''
        """ Plot the truth data """
        plot_classes(
                class_array=Y,
                class_labels=class_labels,
                colors=colors,
                show=False,
                plot_spec={
                    "title":f"Indian Pines Truth Labels"
                    },
                fig_path=fig_dir.joinpath("grid_truth.png"),
                )
        '''
        '''
        """ Plot the full predictions grid """
        #P[UNK] = 0
        #P[np.logical_not(UNK)] = 0
        plot_classes(
                class_array=P,
                class_labels=class_labels,
                colors=colors,
                show=False,
                plot_spec={
                    "title":f"{md.dir.name} Class Predictions"
                    },
                fig_path=fig_dir.joinpath(
                    f"grid_{md.dir.name}_nomask.png"),
                )
        '''


        #'''
        """ Look at the principle components of an intermediate layer """
        max_pca = 12
        layer_look = -2
        enc_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=model.layers[layer_look].output
                )
        enc_layer,_,_ = pp.gaussnorm(enc_model(X).numpy())
        enc_layer = (enc_layer-np.amin(enc_layer))/np.ptp(enc_layer)
        #from krttdkit.visualize import guitools as gt
        enc_layer = np.reshape(enc_layer, (*grid_shape,-1))
        rgb = np.round(pp.pca(enc_layer, print_table=False)*255).astype(np.uint8)
        for i in range(min(rgb.shape[-1]//3,max_pca)):
            #gt.quick_render()
            Image.fromarray(rgb[...,3*i:3*(i+1)]).save(
                    fig_dir.joinpath(f"{md.dir.name}_pca_{i}.png"))
        #'''

        try:
            (TX,TY,TIDX),(VX,VY,VIDX),(means,stdevs) = pkl.load(
                    md.dir.joinpath(f"{md.dir.name}_dataset.pkl").open("rb"))

            TP = model(TX)
            VP = model(VX)
            cm = confusion_matrix(
                    y_true=np.argmax(TY, axis=-1),
                    y_pred=np.argmax(TP, axis=-1),
                    )
            ConfusionMatrixDisplay(
                    confusion_matrix=cm,
                    display_labels=indian_pines_labels[1:],
                    ).plot()
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
            plt.xticks(rotation=30)
            plt.title("Training data confusion matrix")
            plt.savefig(md.dir.joinpath(f"{md.dir.name}_confusion_train.png"))

            cm = confusion_matrix(
                    y_true=np.argmax(VY, axis=-1),
                    y_pred=np.argmax(VP, axis=-1),
                    )
            ConfusionMatrixDisplay(
                    confusion_matrix=cm,
                    display_labels=indian_pines_labels[1:],
                    ).plot()
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
            plt.xticks(rotation=30)
            plt.title("Validation data confusion matrix")
            plt.savefig(md.dir.joinpath(f"{md.dir.name}_confusion_val.png"))
            #plt.show()
        except:
            continue

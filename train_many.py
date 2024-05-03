import numpy as np
import os
import traceback
##os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from pathlib import Path
from scipy.io import loadmat
from pprint import pprint
import pickle as pkl

import tracktrain.model_methods as mm
from tracktrain.VariationalEncoderDecoder import VariationalEncoderDecoder
from tracktrain.ModelDir import ModelDir
from tracktrain.compile_and_train import train

import preprocess as pp # preprocess,gaussnorm,random_permutation,ints_to_masks
from metadata import aviris_bands,indian_pines_labels

base_config = {
        ## Meta-info
        #"model_type":"ved",
        "model_type":"ff",
        "rand_seed":200007201752,

        ## sampling
        "drop_unknown":True,
        ## if True, values are normalized over all features to preserve
        ## spectral angles rather than independently.
        "bulk_norm":False,

        ## Exclusive to variational encoder-decoder
        #"num_latent":8,
        #"enc_node_list":[64,64,32,32,16],
        #"dec_node_list":[16,32,32,64],
        #"enc_dense_kwargs":{"activation":"relu"},
        #"dec_dense_kwargs":{"activation":"relu"},

        ## Common to models
        "batchnorm":True,
        "dropout_rate":0.0,

        ## Exclusive to compile_and_build_dir
        #"learning_rate":1e-5,
        "loss":"categorical_crossentropy",
        "metrics":["categorical_crossentropy"],
        "weighted_metrics":[],
        "softmax_out":True,

        ## Exclusive to train
        "early_stop_metric":"val_loss", ## metric evaluated for stagnation
        "early_stop_patience":64, ## number of epochs before stopping
        "save_weights_only":True,
        "batch_size":32,
        "batch_buffer":1,
        "max_epochs":512, ## maximum number of epochs to train
        "val_frequency":1, ## epochs between validation

        ## Exclusive to generator init
        #"notes":"fast learning rate, steep bottlenecks",
        }

variations = [
        {"dropout_rate":.2,
         "learning_rate":1e-4,
         "bulk_norm":False,
         "sample_ratio":.8,
         "sample_max":128,
         "batch_size":16,
         "dense_kwargs":{"activation":"sigmoid"},
         "node_list":(64,32),
         "notes":"Max 128 samples per class",
         },
        {"dropout_rate":.2,
         "learning_rate":1e-4,
         "bulk_norm":False,
         "sample_ratio":.8,
         "sample_max":512,
         "batch_size":16,
         "dense_kwargs":{"activation":"sigmoid"},
         "node_list":(64,32),
         "notes":"Max 512 samples per class",
         },
        {"dropout_rate":.2,
         "learning_rate":1e-4,
         "bulk_norm":False,
         "sample_ratio":.8,
         "sample_max":1024,
         "batch_size":16,
         "dense_kwargs":{"activation":"sigmoid"},
         "node_list":(64,32),
         "notes":"Max 1024 samples per class",
         },

        {"dropout_rate":.2,
         "learning_rate":1e-3,
         "bulk_norm":False,
         "sample_ratio":.8,
         "sample_max":512,
         "batch_size":16,
         "dense_kwargs":{"activation":"sigmoid"},
         "node_list":(64,32),
         "notes":"Fast learning rate",
         },
        {"dropout_rate":.2,
         "learning_rate":1e-4,
         "bulk_norm":False,
         "sample_ratio":.8,
         "sample_max":512,
         "batch_size":16,
         "dense_kwargs":{"activation":"sigmoid"},
         "node_list":(64,32),
         "notes":"Moderate learning rate",
         },
        {"dropout_rate":.2,
         "learning_rate":1e-5,
         "bulk_norm":False,
         "sample_ratio":.8,
         "sample_max":512,
         "batch_size":16,
         "dense_kwargs":{"activation":"sigmoid"},
         "node_list":(64,32),
         "notes":"Slow learning rate",
         },
        ]

if __name__=="__main__":
    model_parent_dir = Path("data/models")
    ## (M,N,F) array for F bands
    X = loadmat("./data/indian-pines.mat")["indian_pines"]
    ## (M,F) array of integer classes
    Y = loadmat("./data/indian-pines-truth.mat")["indian_pines_gt"]

    ## Reshape to:
    ## Y := (S,) for S labels for each sample (integers in [0,16] )
    ## X := (S,F) for S samples and F features (radiance bands)
    Y = np.reshape(Y, (Y.shape[0]*Y.shape[1],))
    X = np.reshape(X, (Y.size, X.shape[-1]))
    IDX = np.arange(Y.size)

    model_number_start = 17

    not_unknown = (Y != 0)
    U = X[np.logical_not(not_unknown)]
    ## unknown and known class indeces
    KIDX = IDX[not_unknown]
    UIDX = IDX[np.logical_not(not_unknown)]

    ## Remove 'unknown' (class 0) values if requested
    if base_config.get("drop_unknown"):
        str_labels = indian_pines_labels[1:]
        Y = Y[not_unknown]
        X = X[not_unknown]
        ## capture unknown values
    else:
        str_labels = indian_pines_labels

    base_config["num_inputs"] = X.shape[-1]
    base_config["num_outputs"] = np.unique(Y).size
    for j,v in enumerate(variations):
        ## Update the configuration for this model run
        cur_config = {**base_config, **v}
        model_gen = model_number_start+j
        cur_config["model_name"] = f"{cur_config['model_type']}-{model_gen:03}"

        ## preprocess the data into tensorflow training and validation datasets
        T,V,norm = pp.preprocess(
                X=X,
                Y=Y,
                bulk_norm=cur_config["bulk_norm"],
                sample_ratio=cur_config["sample_ratio"],
                sample_max=cur_config["sample_max"],
                )

        ## indeces refer to the pixel's location in the flattened spatial dims
        ##
        TX,TY,TIDX = T
        VX,VY,VIDX = V
        if base_config.get("drop_unknown"):
            TIDX = KIDX[TIDX]
            VIDX = KIDX[VIDX]
        means,stdevs = norm

        print(TIDX.shape, VIDX.shape, TX.shape, TY.shape, VX.shape, VY.shape)

        ## convert to tensorflow dataset
        data_train = tf.data.Dataset.from_tensor_slices((TX,TY))
        data_val = tf.data.Dataset.from_tensor_slices((VX,VY))

        ## Initialize the model and its directory
        model,md = ModelDir.build_from_config(
                config=cur_config,
                model_parent_dir=model_parent_dir,
                print_summary=True,
                )

        ## train the model and add the results to its directory
        best_model = train(
            model_dir_path=md.dir,
            train_config=cur_config,
            compiled_model=model,
            gen_training=data_train,
            gen_validation=data_val,
            )

        ## Dump the data from this preprocessing run, since the dataset
        ## used varies between sampling configurations.
        pkl.dump(((TX,TY,TIDX), (VX,VY,VIDX), norm),md.dir.joinpath(
            f"{md.dir.name}_dataset.pkl").open("wb"))

import numpy as np
import os
import traceback
##os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from pathlib import Path
from scipy.io import loadmat
from pprint import pprint

import tracktrain.model_methods as mm
from tracktrain.VariationalEncoderDecoder import VariationalEncoderDecoder
from tracktrain.ModelDir import ModelDir
from tracktrain.compile_and_train import train

import preprocess as pp # preprocess,gaussnorm,random_permutation,ints_to_masks

base_config = {
        ## Meta-info
        #"model_type":"ved",
        "model_type":"ff",
        "rand_seed":20000720,

        ## sampling
        "drop_unknown":True,

        ## Exclusive to feedforward
        #"node_list":[64,64,32,32,16],
        #"dense_kwargs":{"activation":"relu"},

        ## Exclusive to variational encoder-decoder
        #"num_latent":8,
        #"enc_node_list":[64,64,32,32,16],
        #"dec_node_list":[16,32,32,64],
        #"enc_dense_kwargs":{"activation":"relu"},
        #"dec_dense_kwargs":{"activation":"relu"},

        ## Common to models
        "batchnorm":True,
        #"dropout_rate":0.0,


        ## Exclusive to compile_and_build_dir
        #"learning_rate":1e-5,
        "loss":"categorical_crossentropy",
        "metrics":["categorical_crossentropy", "mae"],
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
        #"train_val_ratio":.9,
        #"mask_pct":0.0,
        #"mask_pct_stdev":0.0,
        "mask_val":999.,
        "mask_feat_probs":None,
        ## if True, values are normalized over all features to preserve
        ## spectral angles rather than independently.
        "bulk_norm":False,

        "notes":"",
        }

variations = {
        "dropout_rate":(0.0,0.1,0.2,0.4),
        "learning_rate":(1e-6,1e-4,1e-2),
        "train_val_ratio":(.6,.8,.9),
        "mask_pct":(0,0,0,.1,.2,.3),
        "mask_pct_stdev":(0,0,0,.1,.2),

        ## FF only
        "node_list":(
            (512,256,256,64,32),
            (512,256,64,32),
            (256,128,64,32),
            (256,64,32),
            (128,64,32),
            (128,64,32,16),
            (128,64,32,16),
            ),
        "dense_kwargs":(
            {"activation":"relu"},
            {"activation":"sigmoid"},
            #{"activation":"tanh"},
            ),

        ## VED only
        "num_latent":(3,4,8,12),
        "enc_node_list":(
            (256,64,32),
            (512,256,64,32),
            (256,256,256,64,64,64,32,32,32,16),
            (128,64,64,32,32,32,16),
            ),
        "dec_node_list":(
            (17,),
            ),
        "enc_dense_kwargs":(
            {"activation":"relu"},
            {"activation":"sigmoid"},
            #{"activation":"tanh"},
            ),
        "dec_dense_kwargs":(
            {"activation":"relu"},
            {"activation":"sigmoid"},
            #{"activation":"tanh"},
            ),
        }

num_samples = 16
offset = 64 ## Model numbering offset accounting for already-trained models
model_base_name = base_config.get("model_type")+"-nounk" #"ff"

if __name__=="__main__":
    model_parent_dir = Path("data/models")
    ## (M,N,F) array for F bands
    X = loadmat("./data/indian-pines.mat")["indian_pines"]
    ## (M,F) array of integer classes
    Y = loadmat("./data/indian-pines-truth.mat")["indian_pines_gt"]

    if base_config.get("drop_unknown"):
        not_unknown = (Y != 0)
        Y = Y[not_unknown]
        X = X[not_unknown]

    pp.ratio_sample(Y, .6)
    exit(0)

    ## Construct a coordinate array of equally-spaced wavelengths
    wl_min,wl_max = (.4, 2.5) ## define wavelength range
    wls = np.linspace(.4,2.5,X.shape[-1])

    ## Reshape and rescale the feature data, and one-hot encode the labels
    grid_shape = X.shape[:2]
    X,Y = pp.preprocess(X, Y, gain=500, offset=1000, cast_to_onehot=True)
    IDX = np.arange(X.shape[0])
    X,means,stdevs = pp.gaussnorm(
            X=(X_nonorm:=X),
            bulk_norm=base_config.get("bulk_norm"),
            )

    ## Get integer arrays encoding a seeded random permutation and its inverse
    forward,backward = pp.random_permutation(
            X.shape[0], seed=base_config.get("rand_seed"))
    X,Y,IDX = X[forward],Y[forward],IDX[forward]

    ## Exclude 35% of the data for testing from the front of the shuffled array
    train_split_idx = int(.35*X.shape[0])
    X_test,X_train_and_val = np.split(X, [train_split_idx], axis=0)
    Y_test,Y_train_and_val = np.split(Y, [train_split_idx], axis=0)
    IDX_test,IDX_train_and_val = np.split(IDX, [train_split_idx], axis=0)

    base_config["num_inputs"] = X_test.shape[-1]
    base_config["num_outputs"] = Y_test.shape[-1]

    ## Dispatch training loops
    comb_failed = []
    comb_trained = []
    vlabels,vdata = zip(*variations.items())
    vdata = tuple(map(tuple, vdata))
    comb_shape = tuple(len(v) for v in vdata)
    comb_count = np.prod(np.array(comb_shape))
    for i in range(num_samples):
        ## Get a random argument combination from the configuration
        cur_comb = tuple(np.random.randint(0,j) for j in comb_shape)
        cur_update = {
                vlabels[i]:vdata[i][cur_comb[i]]
                for i in range(len(vlabels))
                }
        cur_update["model_name"] = model_base_name+f"-{i+offset:03}"
        ## Construct the config for this model training run
        cur_config = {**base_config, **cur_update}
        pprint(cur_config)
        try:
            ## Initialize the masking data generators
            gen_train,gen_val = mm.array_to_noisy_tv_gen(
                    X=X_train_and_val.astype(np.float64),
                    Y=Y_train_and_val.astype(np.float64),
                    tv_ratio=cur_config.get("train_val_ratio"),
                    noise_pct=cur_config.get("mask_pct"),
                    noise_stdev=cur_config.get("mask_pct_stdev"),
                    mask_val=cur_config.get("mask_val"),
                    feat_probs=cur_config.get("mask_feat_probs"),
                    shuffle=True,
                    dtype=tf.float64,
                    rand_seed=cur_config.get("random_seed"),
                    )
            ## Initialize the model and its directory
            model,md = ModelDir.build_from_config(
                    config=cur_config,
                    model_parent_dir=model_parent_dir,
                    print_summary=False,
                    )
            ## train the model and add the results to its directory
            best_model = train(
                model_dir_path=md.dir,
                train_config=cur_config,
                compiled_model=model,
                gen_training=gen_train,
                gen_validation=gen_val,
                )
        except Exception as e:
            print(f"FAILED update combination {cur_update}")
            print(traceback.format_exc())
            comb_failed.append(cur_comb)
        comb_trained.append(cur_comb)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

from utils.resBlock import res_block, res_block_org
from utils.data_reader import read_hdf_data, read_hdf_data_psi, read_h5_data
from utils.writeANNProperties import writeANNProperties
from utils.customObjects import coeff_r2, SGDRScheduler
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

import ast
import time

##########################
# Parameters
n_neuron = 250
branches = 3
scale = 3
batch_size = 1024 * 8  # 512
this_epoch = 5000
vsplit = 0.1
batch_norm = False

# define the type of scaler: MinMax or Standard
scaler = "Standard"  # 'Standard' 'MinMax'

o_scaler = "cbrt_std"  # string: which output scaler function is used
##########################

# read in the species order
with open("GRI_species_order_lu13", "r") as f:
    # print(species)
    labels = f.read().splitlines() 

# append other fields: heatrelease,  T, PVs
# labels.append('heatRelease')
labels.append("T")
labels.append("PVs")

print("The labels are:")
print(labels)

# # tabulate psi, mu, alpha
# labels.append('psi')
# labels.append('mu')
# labels.append('alpha')

# DO NOT CHANGE THIS ORDER!!
input_features = ["f", "zeta", "pv"]

# read in the data
X, y, df, in_scaler, out_scaler = read_h5_data(
    "./data/tables_of_fgm.h5",
    input_features=input_features,
    labels=labels,
    i_scaler="std2",
    o_scaler=o_scaler,
)
# ('./data/tables_of_fgm.h5',key='of_tables',
# in_labels=input_features, labels = labels,scaler=scaler)

# split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# %%
print("set up ANN")

# ANN parameters
dim_input = X_train.shape[1]
dim_label = y_train.shape[1]

# This returns a tensor
inputs = Input(shape=(dim_input,), name="input_1")

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(n_neuron, activation="relu")(inputs)

x = res_block_org(x, n_neuron, stage=1, block="a", bn=batch_norm)
x = res_block_org(x, n_neuron, stage=1, block="b", bn=batch_norm)
# x = res_block_org(x, n_neuron, stage=1, block="c", bn=batch_norm)
# x = res_block_org(x,  n_neuron, stage=1,  block='d', bn=batch_norm)

predictions = Dense(dim_label, activation="linear", name="output_1")(x)

model = Model(inputs=inputs, outputs=predictions)

# WARM RESTART
batch_size_list = [batch_size]

epoch_factor = 1

t_start = time.time()

for this_batch in batch_size_list:

    model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
    # get the model summary
    model.summary()

    # checkpoint (save the best model based validate loss)
    filepath = "./tmp/weights.best.cntk.hdf5"

    # check if there are weights
    if os.path.isdir(filepath):
        model.load_weights(filepath)

    checkpoint = ModelCheckpoint(
        filepath,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
        period=10,
    )
    epoch_size = X_train.shape[0]
    a = 0
    base = 2
    clc = 2
    for i in range(9):
        a += base * clc ** (i)
    print(a)
    epochs, c_len = a, base

    schedule = SGDRScheduler(
        min_lr=1e-5,
        max_lr=1e-3,
        steps_per_epoch=np.ceil(epoch_size / batch_size),
        cycle_length=c_len,
        lr_decay=0.6,
        mult_factor=clc,
    )

    callbacks_list1 = [checkpoint]
    callbacks_list2 = [checkpoint, schedule]

    # fit the model
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,  # * epoch_factor,
        batch_size=this_batch,
        validation_split=vsplit,
        verbose=2,
        callbacks=callbacks_list1,
        shuffle=True,
    )
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,  # * epoch_factor,
        batch_size=this_batch,
        validation_split=vsplit,
        verbose=2,
        callbacks=callbacks_list2,
        shuffle=True,
    )

    epoch_factor += 1

t_end = time.time()
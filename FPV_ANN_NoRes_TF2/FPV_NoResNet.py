'''
This code version is written for TF 2.0.0

date: Nov 2019
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

from utils.resBlock import NoRes_block
from utils.data_reader import read_hdf_data, read_hdf_data_psi, read_h5_data
from utils.writeANNProperties import writeANNProperties
from utils.customObjects import coeff_r2, SGDRScheduler
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

import ast
import time

##########################
# Parameters
n_neurons = [100,250,500]
branches = 3
scale = 3
batch_size = 1024 * 8  # 512
this_epoch = 3000
vsplit = 0.1
batch_norm = False

blocks_name = '4_Block_Nets'

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
for n_neuron in n_neurons:

    print("set up ANN")

    # ANN parameters
    dim_input = X_train.shape[1]
    dim_label = y_train.shape[1]

    # This returns a tensor
    inputs = Input(shape=(dim_input,), name="input_1")

    # a layer instance is callable on a tensor, and returns a tensor
    x = Dense(n_neuron, activation="relu")(inputs)

    x = NoRes_block(x, n_neuron, stage=1, block="a", bn=batch_norm)
    x = NoRes_block(x, n_neuron, stage=1, block="b", bn=batch_norm)
    x = NoRes_block(x, n_neuron, stage=1, block="c", bn=batch_norm)
    x = NoRes_block(x,  n_neuron, stage=1,  block='d', bn=batch_norm)

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
        for i in range(10):
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

        # # fit the model
        # history = model.fit(
        #     X_train,
        #     y_train,
        #     epochs=epochs,  # * epoch_factor,
        #     batch_size=this_batch,
        #     validation_split=vsplit,
        #     verbose=2,
        #     callbacks=callbacks_list1,
        #     shuffle=True,
        # )
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

    # # %%
    # ref = df.loc[df['p'] == 40]
    # x_test = in_scaler.transform(ref[['p', 'he']])

    predict_val = model.predict(X_test)

    X_test_df = pd.DataFrame(in_scaler.inverse_transform(X_test), columns=input_features)
    y_test_df = pd.DataFrame(out_scaler.inverse_transform(y_test), columns=labels)

    sp = 'PVs'

    predict_df = pd.DataFrame(out_scaler.inverse_transform(predict_val), columns=labels)

    # for sp in labels:
    #     plt.figure()
    #     plt.scatter(predict_df[sp], y_test_df[sp], s=1)
    #     plt.title('R2 for ' + sp)
    #     plt.savefig('./exported/R2_%s_%s_%i.eps' % (sp, scaler, n_neuron), format='eps')
    #     plt.show(block=False)

    # loss
    fig = plt.figure()
    plt.semilogy(history.history['loss'])
    if vsplit:
        plt.semilogy(history.history['val_loss'])
    plt.title('mse')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig(join(blocks_name,'Loss_%s_%s_%i.eps' % (sp, scaler, n_neuron)), format='eps')
    #plt.show(block=False)

    # predict_df = pd.DataFrame(out_scaler.inverse_transform(predict_val), columns=labels)

    plt.figure()
    plt.title('Error of %s ' % sp)
    plt.plot((y_test_df[sp] - predict_df[sp]) / y_test_df[sp])
    plt.title(sp)
    plt.savefig(join(blocks_name,'Error_%s_%s_%i.eps' % (sp, scaler, n_neuron)), format='eps')
    #plt.show(block=False)

    plt.figure()
    plt.scatter(predict_df[sp], y_test_df[sp], s=1)
    plt.title('R2 for ' + sp)
    plt.savefig(join(blocks_name,'R2_%s_%s_%i.eps' % (sp, scaler, n_neuron)), format='eps')
    #plt.show(block=False)
    # %%
    a = (y_test_df[sp] - predict_df[sp]) / y_test_df[sp]
    test_data = pd.concat([X_test_df, y_test_df], axis=1)
    pred_data = pd.concat([X_test_df, predict_df], axis=1)

    test_data.to_hdf('sim_check.H5', key='test')
    pred_data.to_hdf('sim_check.H5', key='pred')

    # Save model
    # sess = K.get_session()
    # saver = tf.train.Saver(tf.global_variables())
    # saver.save(sess, './exported/my_model')
    model.save(join(blocks_name,'FPV_ANN_tabulated_%s_%i.H5' % (scaler, n_neuron)))

    # write the OpenFOAM ANNProperties file
    writeANNProperties(in_scaler, out_scaler, scaler, o_scaler)

    # save train time
    train_time = (t_end - t_start)
    print('Training took %i sec.' % train_time)
    np.savetxt(join(blocks_name,'training_time_%i.csv' % (n_neuron)),[train_time])


    # save the loss history
    losses_df = pd.DataFrame(np.array(
        [history.history['accuracy'], history.history['val_accuracy'], history.history['loss'], history.history['val_loss']]).T,
                             columns=['accuracy', 'val_accuracy', 'loss', 'val_loss'])

    losses_df.to_csv(join(blocks_name,'losses_%i_%iepochs.csv' % (n_neuron, this_epoch)))

# end loop over n_neurons

# Convert the model to
#run -i k2tf.py --input_model='FPV_ANN_tabulated_Standard.H5' --output_model='exported/FPV_ANN_tabulated_Standard.pb'

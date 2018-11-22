import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input
from keras.callbacks import ModelCheckpoint

from resBlock import res_block
from data_reader import read_csv_data
#from writeANNProperties import writeANNProperties


# define the labels
labels = ['T','CH4','O2','CO2','CO','H2O','H2','OH','PVs']

X, y, df, in_scaler, out_scaler = read_csv_data('premix_data',labels = labels)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1)

######################
print('set up ANN')
# ANN parameters
dim_input = 2
dim_label = y_train.shape[1]
n_neuron = 600
batch_size = 1024
epochs = 500
vsplit = 0.1
batch_norm = False

# This returns a tensor
inputs = Input(shape=(dim_input,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(n_neuron, activation='relu')(inputs)

# less then 2 res_block, there will be variance
x = res_block(x, n_neuron, stage=1, block='a', bn=batch_norm)
x = res_block(x, n_neuron, stage=1, block='b', bn=batch_norm)
x = res_block(x, n_neuron, stage=1, block='c', bn=batch_norm)
x = res_block(x, n_neuron, stage=1, block='d', bn=batch_norm)
x = res_block(x, n_neuron, stage=1, block='e', bn=batch_norm)
x = res_block(x, n_neuron, stage=1, block='f', bn=batch_norm)
x = res_block(x, n_neuron, stage=1, block='g', bn=batch_norm)

predictions = Dense(dim_label, activation='linear')(x)

model = Model(inputs=inputs, outputs=predictions)
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# checkpoint (save the best model based validate loss)
filepath = "./tmp/weights.best.cntk.hdf5"

checkpoint = ModelCheckpoint(filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min',
                             period=10)

callbacks_list = [checkpoint]

# fit the model
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=vsplit,
    verbose=2,
    callbacks=callbacks_list,
    shuffle=True)

# loss
fig = plt.figure()
plt.semilogy(history.history['loss'])
if vsplit:
    plt.semilogy(history.history['val_loss'])
plt.title('mse')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

#########################################
model.load_weights("./tmp/weights.best.cntk.hdf5")
# cntk.combine(model.outputs).save('mayerTest.dnn')

# # %%
# ref = df.loc[df['p'] == 40]
# x_test = in_scaler.transform(ref[['p', 'he']])

X_test = pd.DataFrame(X_test,columns=['f','PV'])
y_test = pd.DataFrame(y_test,columns=labels)

predict_val = model.predict(X_test.values)

sp='CH4'

predict = pd.DataFrame(out_scaler.inverse_transform(predict_val), columns=labels)

plt.figure()
plt.plot(X_test['f'], y_test[sp], 'r:')
plt.plot(X_test['f'], predict[sp], 'b-')
plt.show()
plt.figure()
plt.title('Error of %s ' % sp)
plt.plot((y_test[sp] - predict[sp]) / y_test[sp])
plt.show()

# %%
from keras import backend as K

sess = K.get_session()
saver = tf.train.Saver(tf.global_variables())
saver.save(sess, './exported/my_model')
tf.train.write_graph(sess.graph, '.', "./exported/graph.pb", as_text=False)
np.savetxt('x_test.csv',x_test)
np.savetxt('prediction.csv',predict_val)
model.save('FPB.H5')

# write the OpenFOAM ANNProperties file
# writeANNProperties(in_scaler,out_scaler)




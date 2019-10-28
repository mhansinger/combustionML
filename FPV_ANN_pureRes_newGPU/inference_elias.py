import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.callbacks import ModelCheckpoint

from utils.data_reader import read_hdf_data, read_hdf_data_psi, read_h5_data

from keras import backend as K
from keras.models import load_model
import os

import ast
import time

##########################

# muss das gleiche model sein, welches du als tensorflow graph verwendest!
path_to_model = '4_Block_Nets/FPV_ANN_tabulated_Standard_500.H5'

path_to_data = './data/tables_of_fgm.h5'

# disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

data_points=1000000

##########################



# define the type of scaler: MinMax or Standard
scaler = 'Standard' # 'Standard' 'MinMax'

o_scaler = 'cbrt_std'   #string: which output scaler function is used


# read in the species order
with open('GRI_species_order_lu13', 'r') as f:
    # print(species)
    labels = f.read().splitlines()

labels.append('T')
labels.append('PVs')

print('The labels are:')
print(labels)

# DO NOT CHANGE THIS ORDER!!
input_features=['f','zeta','pv']

# read in the data
X, y, df, in_scaler, out_scaler = read_h5_data(path_to_data,
                                               input_features=input_features,
                                               labels=labels,
                                               i_scaler='std2',
                                               o_scaler=o_scaler)

# split into train and test data
test_size=data_points/len(X)
print('Test size is %f of entire data set\n' % test_size)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size)

# load the model
model = load_model(path_to_model)

# #############################
# inference part
t_start = time.time()
predict_val = model.predict(X_test)

t_end = time.time()
# #############################

print('Inference time was: %.3f' % float(t_end- t_start))

X_test_df = pd.DataFrame(in_scaler.inverse_transform(X_test),columns=input_features)
y_test_df = pd.DataFrame(out_scaler.inverse_transform(y_test),columns=labels)

sp='PVs'

predict_df = pd.DataFrame(out_scaler.inverse_transform(predict_val), columns=labels)

plt.figure()
plt.scatter(predict_df[sp],y_test_df[sp],s=1)
plt.title('R2 for '+sp)
#plt.savefig('./exported/R2_%s_%s_%i.eps' % (sp,scaler,n_neuron),format='eps')
plt.show(block=False)





import tensorflow as tf
import keras
import keras.models
from Model_comparisons import *
from keras import Model

#Liam training and validation datasets,
# TODO: need to check if these contain same data as Jad's tfrecords. If not then use Jad's TFrecords
# TODO: So jad may not have used TFrecords at all.  May have to redo from scratch.
dataset_val = tf.data.TFRecordDataset('D:/Comparison_1/data_val.tfrecord')
dataset_train = tf.data.TFRecordDataset('D:/Comparison_1/data_train.tfrecord')

batch_size = 32

dataset_train = dataset_train.map(map_fn)
dataset_train = dataset_train.shuffle(2048, reshuffle_each_iteration=True)
dataset_train = dataset_train.prefetch(buffer_size=tf.data.AUTOTUNE)
dataset_train = dataset_train.batch(batch_size)

dataset_val = dataset_val.map(map_fn)
dataset_val = dataset_val.prefetch(buffer_size=tf.data.AUTOTUNE)
dataset_val = dataset_val.batch(batch_size)

# conv_model = ranking_model(224)
# # conv_model = load_model("D:/acc_checkpoint") #Or load in from a checkpoint
#
# # Add early stopping callback
# # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)
#
# history = conv_model.fit(dataset_train, validation_data=dataset_val, epochs=55)  # ,callbacks=[callback])


# Jads optimizied comparison model for feature extractor
filename = r"C:\Users\laggi\University of Ottawa\LAGGISS - General\Jad\Comp_Q1_87_acc\fitted_model.h5"
jad_w = keras.models.load_model(filename)
vgg=jad_w.get_layer("vgg19")

#vgg = Model(comp_model.layers[5].input, comp_model.layers[5].output)
rank_model = ranking_model(224, vgg_feature_extractor = vgg)

history_rank = rank_model.fit(dataset_train, validation_data = dataset_val, epochs = 50)


import matplotlib.pyplot as plt

plt.plot(history_rank.history['val_accuracy'])
plt.plot(history_rank.history['accuracy'])
plt.plot(history_rank.history['val_loss'])
plt.plot(history_rank.history['loss'])

import h5py
def print_attrs(name, obj):
    with open("c:/temp/jmod.txt",'w') as file:
        #f.write(name+'\n')
        for key, val in obj.attrs.items():
            print("{}: {}".format(key, val))
            file.write("    {}: {}".format(key, val))


filename = r"C:\Users\laggi\University of Ottawa\LAGGISS - General\Jad\Comp_Q1_87_acc\fitted_model.h5"
f = h5py.File(filename, 'r')
f.visititems(print_attrs)

import tensorflow as tf
jad=tf.keras.models.load_model(filename)



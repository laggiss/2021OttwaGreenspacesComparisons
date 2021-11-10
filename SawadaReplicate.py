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

import tensorflow
#Visualize TFRecord image
import tensorflow as tf
from tensorflow.io import *
import matplotlib.pyplot as plt

def parse_fn(data_record):
    features = {
        'data_label' : tf.train.Feature(int64_list=tf.train.Int64List(value=data_label[j])),
        'labels_score': tf.train.Feature(int64_list=tf.train.Int64List(value=np.array([labels_score[j]]))),
        'data_left': tf.train.Feature(int64_list=tf.train.Int64List(value=data_left[j].astype(int).flatten())),
        'data_right': tf.train.Feature(int64_list=tf.train.Int64List(value=data_right[j].astype(int).flatten())),

        }
    feature = {'image/encoded': tf.io.FixedLenFeature([], tf.string),
               'image/object/class/label': tf.io.FixedLenFeature([], tf.int64)}

    sample = tf.io.parse_single_example(data_record, feature)
    print('iiisdfasdf')
    return sample

file_path = 'D:/Comparison_1/data_val.tfrecord'
dataset = tf.data.TFRecordDataset([file_path])
iterator=iter(dataset)
record_iterator = next(iterator)

with tensorflow.compat.v1.Session() as sess:
    # Read and parse record
    parsed_example = parse_fn(record_iterator)

    # Decode image and get numpy array
    encoded_image = parsed_example['image/encoded']
    decoded_image = tf.image.decode_jpeg(encoded_image, channels=3)
    image_np = sess.run(decoded_image)

    # Display image
    plt.imshow(image_np)
    plt.show()




def parse_tfrecord_fn(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "path": tf.io.FixedLenFeature([], tf.string),
        "area": tf.io.FixedLenFeature([], tf.float32),
        "bbox": tf.io.VarLenFeature(tf.float32),
        "category_id": tf.io.FixedLenFeature([], tf.int64),
        "id": tf.io.FixedLenFeature([], tf.int64),
        "image_id": tf.io.FixedLenFeature([], tf.int64),
        }
    example = tf.io.parse_single_example(example, feature_description)
    print("ikskdfasdf")
    example["image"] = tf.io.decode_png(example["image"], channels=3)
    example["bbox"] = tf.sparse.to_dense(example["bbox"])
    return example

raw_dataset = tf.data.TFRecordDataset('D:/Comparison_1/data_val.tfrecord')
parsed_dataset = raw_dataset.map(parse_tfrecord_fn(1))

for features in parsed_dataset.take(1):
    for key in features.keys():
        if key != "image":
            print(f"{key}: {features[key]}")

    print(f"Image shape: {features['image'].shape}")
    plt.figure(figsize=(7, 7))
    plt.imshow(features["image"].numpy())
    plt.show()
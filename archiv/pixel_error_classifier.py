from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
K.clear_session()

import os, random
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
!pwd
# CNN model setup
# VERSION 2 zachariah047_384_960
'''
This script builds a image classification models to detect pixes-errors
in video footage.
steps to do first:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created clean/ and error/ subfolders inside train/ and validation/
- put the clean pictures index 0-999 in data/train/clean
- put the clean pictures index 1000-1400 in data/validation/clean

- put the error pictures index 12500-13499 in data/train/error
- put the error pictures index 13500-13900 in data/validation/error
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:

data/
    train/
        error/
            error001.jpg
            error002.jpg
            ...
        clean/
            clean001.jpg
            clean002.jpg
            ...
    validation/
        error/
            error001.jpg
            error002.jpg
            ...
        clean/
            clean001.jpg
            clean002.jpg
            ...
'''

# dimensions of our images.
img_width, img_height = 64, 64

train_data_dir = 'train/'
validation_data_dir = 'validation/'
test_dir = 'test/'
nb_train_samples = 375000
nb_validation_samples = 160000
epochs = 60
batch_size = 16
location = !pwd

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

n_images = nb_train_samples + nb_validation_samples
print(f"[INFO] loading {n_images} images from '{location[-1].split('/')[-1]}' ...")

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape), padding='same')
# padding anschalten um die ecken zu checken
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# checkpoint
checkpointer = ModelCheckpoint(filepath='bestmodel_weights.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
earlystopper = EarlyStopping(monitor='val_loss', patience=4, verbose=1)
callbacks_list = [checkpoint, earlystopper]
model.summary()

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(horizontal_flip=True)
# this is the augmentation configuration we will use for testing:
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=callbacks_list)


# serialize weights to HDF5
model.save_weights('cnn-model_03.h5')
print("Saved model to disk")


test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        color_mode="rgb",
        shuffle = False,
        class_mode='binary',
        batch_size=1)


# serialize model to JSON
with open("model.json", "w") as json_file:
   json_file.write(model.to_json())

filenames = test_generator.filenames
nb_samples = len(filenames)

predict = model.predict_generator(test_generator,steps = nb_samples)
result = dict(zip([each[0] for each in predict], [filename.split('.')[-2].split('/')[-1] for filename in filenames]))
df = pd.DataFrame(result, index=range(1))
df.to_csv('result.csv', index=False)

result

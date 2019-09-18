from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import regularizers
from keras import backend as K
from sklearn.metrics import classification_report
from keras.layers.normalization import BatchNormalization

from time import gmtime, strftime
import os, random
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# dimensions of input images
img_width, img_height = 64, 64

train_data_dir = 'keras_cnn/train/'
validation_data_dir = 'keras_cnn/validation/'
test_dir = 'keras_cnn/test/'
nb_train_samples = 375000
nb_validation_samples = 160000
epochs = 5
batch_size = 300
location = os.getcwd()
!pwd
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

n_images = nb_train_samples + nb_validation_samples
print(f"[INFO] loading {n_images} images from '{location.split('/')[-1]}' ...")


# Model reconstruction from JSON file
with open('keras_cnn/model/model_strides_11-24-50.json', 'r') as f:
    model = model_from_json(f.read())
# Load weights into the new model
model.load_weights('keras_cnn/model/cnn-model_strides_11-24-50.h5')
model.summary()
# evaluate the network
print("[INFO] predicting pixel errors...")


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(horizontal_flip=True)
# this is the augmentation configuration we will use for testing:
test_datagen = ImageDataGenerator(horizontal_flip=True)

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

hist = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[checkpoint])

ti = strftime("%d_%H-%M-%S", gmtime())

# save eights to HDF5
model.save_weights(f'keras_cnn/model/cnn-model_strides_{ti}.h5')
print("Saved model to disk")
# save model to JSON
with open(f"keras_cnn/model/model_strides_{ti}.json", "w") as json_file:
    json_file.write(model.to_json())

# evaluate the network
print("[INFO] evaluating network...")
# plot the training loss and accuracy
epochs = len(list(hist.history.values())[0])
fig = plt.figure()
plt.plot(np.arange(0, epochs), hist.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), hist.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), hist.history["acc"], label="train_acc")
plt.plot(np.arange(0, epochs), hist.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel(f"Epoch (max. {epochs})")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(f"keras_cnn/plots/plot_strides_{ti}.png")
plt.show()

test_generator = test_datagen.flow_from_directory(
test_dir,
target_size=(img_width, img_height),
color_mode="rgb",
shuffle = False,
class_mode='binary',
batch_size=1)

filenames = test_generator.filenames
nb_samples = len(filenames)
predict = model.predict_generator(test_generator,steps = nb_samples)
y_pred = [i[0].round() for i in predict]

# set y_true for test data
images_clean = os.listdir('keras_cnn/test/clean')
images_error = os.listdir('keras_cnn/test/error')
y_clean = np.zeros(len(images_clean))
y_error =  np.ones(len(images_error))
y_true = list(np.concatenate((y_clean, y_error), axis=0))
filenames = list(np.concatenate((images_clean, images_error), axis=0))
loc_folder = ['/clean/']*len(images_clean)+['/error/']*len(images_error)
loc_images = list(np.concatenate((images_clean, images_error), axis=0))
src_img=[]
for i in range(len(loc_folder)):
    src_img.append(loc_folder[i] + loc_images[i])
y_clean = np.zeros(len(images_clean))
y_error = np.ones(len(images_error))
y_true = list(np.concatenate((y_clean, y_error), axis=0))

print(classification_report(y_true, y_pred))

result = dict(zip([each[0] for each in predict], src_img))
df = pd.DataFrame(result, index=range(1))
df = df.T.reset_index()
df.columns = ['y_pred', 'file']
df.to_csv(f'keras_cnn/result_csv/result_strides_{ti}.csv', index=False)

from PIL import Image
for _, row in df[150:165].iterrows():
    print(row["y_pred"])
    pil = Image.open(f"keras_cnn/test/{row['file']}", "r")
    df['img'] = plt.imshow(np.asarray(pil))
    plt.figure()

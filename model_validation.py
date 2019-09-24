from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from time import gmtime, strftime
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import pydot
import cv2

epochs = 15
ti = strftime("%d_%H-%M-%S", gmtime())
img_width, img_height = 64, 64
test_dir = 'keras_cnn/test/'
!pwd
# Model reconstruction from JSON file
with open('keras_cnn/model/model_strides_11-24-50.json', 'r') as f:
    model = model_from_json(f.read())
# Load weights into the new model
model.load_weights('keras_cnn/model/cnn-model_strides_11-24-50.h5')

# save a list of np.arrays with the weights
w = model.get_weights()

# see the underlying TensorFlow variables
model.weights

# extract the names of the TF variables
[v.name for v in model.weights]

# plot weights of one layer (for MNIST)

SVG(model_to_dot(model, show_shapes=True, show_layer_names=True, rankdir='HB').create(prog='dot', format='svg'))

# prediction
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
test_dir,
target_size=(img_width, img_height),
color_mode="rgb",
shuffle = False,
class_mode='binary',
batch_size=1)

nb_samples = len(test_generator.filenames)
predict = model.predict_generator(test_generator,steps = nb_samples)
y_pred = [i[0].round() for i in predict]

# set y_true for test data
images_clean = os.listdir('keras_cnn/test/clean')
images_error = os.listdir('keras_cnn/test/error')
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

#result = dict(zip([each[0] for each in predict], [filename.split('.')[-2].split('/')[-1] for filename in filenames]))
result = dict(zip([each[0] for each in predict], src_img))
df = pd.DataFrame(result, index=range(1))
df = df.T.reset_index()
df.columns = ['y_pred', 'file']
df.head()
df.to_csv(f'keras_cnn/result_csv/result_strides_{ti}.csv', index=False)
plt.plot(y_true, y_pred)

from PIL import Image
for _, row in df.head().iterrows():
    print(row["y_pred"])
    pil = Image.open(f"keras_cnn/test/{row['file']}", "r")
    df['img'] = plt.imshow(np.asarray(pil))
    plt.figure()

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from time import gmtime, strftime
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os

epochs = 15
ti = strftime("%d_%H-%M-%S", gmtime())
img_width, img_height = 64, 64
test_dir = 'test/'

# Model reconstruction from JSON file
with open('model/model_strides_11-24-50.json', 'r') as f:
    model = model_from_json(f.read())
# Load weights into the new model
model.load_weights('model/cnn-model_strides_11-24-50.h5')

# evaluate the network
print("[INFO] predicting pixel errors...")

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
images_clean = os.listdir('test/clean')
images_error = os.listdir('test/error')
filenames = list(np.concatenate((images_clean, images_error), axis=0))
y_clean = np.zeros(len(images_clean))
y_error =  np.ones(len(images_error))
y_true = list(np.concatenate((y_clean, y_error), axis=0))

print(classification_report(y_true, y_pred))

result = dict(zip([each[0] for each in predict], [filename.split('.')[-2].split('/')[-1] for filename in filenames]))
df = pd.DataFrame(result, index=range(1))
df = df.T.reset_index()
df.columns = ['y_pred', 'filename']
df.head()
df.to_csv(f'result_csv/result_strides_{ti}.csv', index=False)
plt.plot(y_true, y_pred)

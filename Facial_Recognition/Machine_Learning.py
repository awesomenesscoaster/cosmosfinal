import cv2
import os
import numpy as np

from tensorflow import keras
from sklearn.model_selection import train_test_split
    
from facial_recognition import picture_files
from facial_recognition import dir_files_cropped
from facial_recognition import img_numpy_array_list

x_data = img_numpy_array_list
y_data = np.array(dir_files_cropped)
print(x_data.shape)
x_data = img_numpy_array_list.reshape(-1, 200, 200, 1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.20, random_state=0)

model_cnn = keras.Sequential()

num_filters = 8
filter_size = 3
pool_size = 2

model_cnn.add( keras.layers.Conv2D(num_filters, filter_size, input_shape=(8,8,3)) )
model_cnn.add( keras.layers.MaxPooling2D(pool_size=pool_size) )
model_cnn.add( keras.layers.Flatten() )
model_cnn.add( keras.layers.Dense(2, activation='softmax') )
print('Complete!')
    
model_cnn.compile(
'adam',
loss='categorical_crossentropy',
metrics=['accuracy'],
)
    
model_cnn.fit(
x = x_train,
y=keras.utils.to_categorical(y_train),
epochs=2,
batch_size=10
)
    
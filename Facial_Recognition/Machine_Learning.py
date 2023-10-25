import cv2
import os
import numpy as np

from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array

from facial_recognition import integer_img_conversion
from facial_recognition import img_numpy_array_list

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x_data = img_numpy_array_list
y_data = np.array(integer_img_conversion)
x_data = img_numpy_array_list.reshape(-1, 200, 200, 1)
print(x_data.shape)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.20, random_state=0)

model_cnn = keras.Sequential()

num_filters = 8
filter_size = 3
pool_size = 2

model_cnn.add( keras.layers.Conv2D(num_filters, filter_size, input_shape=(200,200,1)) )
model_cnn.add( keras.layers.MaxPooling2D(pool_size=pool_size) )
model_cnn.add( keras.layers.Flatten() )
model_cnn.add( keras.layers.Dense(2, activation='softmax') )
print('Complete!')
    
model_cnn.compile(
'adam',
loss='categorical_crossentropy',
metrics=['accuracy'],
)
    
# model_cnn.fit(
# x = x_train,
# y=keras.utils.to_categorical(y_train),
# epochs=2,
# batch_size=10
# )

# score = model_cnn.evaluate(x_test, y_test)
# print(score)
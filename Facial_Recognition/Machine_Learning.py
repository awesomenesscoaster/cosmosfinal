import cv2
import os
import numpy as np

from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array

from facial_recognition import integer_img_conversion
from facial_recognition import img_numpy_array_list

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

userCountL = []
for i in integer_img_conversion:
    if i not in userCountL:
        userCountL.append(i)

x_data = img_numpy_array_list
y_data = np.array(integer_img_conversion)
x_data = img_numpy_array_list.reshape(-1, 200, 200, 1)
print(x_data.shape)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.20, random_state=0)

model_cnn = keras.Sequential()

num_filters = 150
filter_size = 5
pool_size = 4

model_cnn.add( keras.layers.Conv2D(num_filters, filter_size, input_shape=(200,200,1)) )
model_cnn.add( keras.layers.MaxPooling2D(pool_size=pool_size) )
model_cnn.add( keras.layers.Flatten() )
model_cnn.add( keras.layers.Dense(8, activation='softmax') )
model_cnn.add( keras.layers.Dense(1)) #len(userCountL)
print('Complete!')
    
model_cnn.compile(
optimizer = 'adam',
loss='categorical_crossentropy',
metrics=['accuracy'],
)

#model_cnn.summary()

model_cnn.fit(x_train,y_train,epochs=2,validation_data=(x_test,y_test))

# model_cnn.fit(
# x = x_train,
# y=keras.utils.to_categorical(y_train),
# epochs=10,
# batch_size=len(y_train)
# )
print('Complete 2.0')

test_lost, test_acc = model_cnn.evaluate(x_test, y_test)
print(test_acc)
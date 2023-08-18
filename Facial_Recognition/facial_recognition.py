import cv2
import os
import numpy as np

import time

from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array

global picture_files
global dir_files_cropped
global img_numpy_array_list


scanning_process = False
guessing_process = False

names = []

user_name = input("What is your name? ")

if user_name not in names:
    scanning_process = True
    names.append(user_name)
    
    folder_path = os.path.join('C:/Users/Rhyan Shah/Documents/GitHub/cosmosfinal/Facial_Recognition/Datasets', user_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
else:
    guessing_process = True
    print('Welcome back:', user_name)

if scanning_process == True:
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Facial Recognition")
    
    img_counter = 0
    
    while True:
        ret, frame = cam.read()
        
        if not ret:
            print("Failed.")
            
        cv2.imshow("Facial Recognition", frame)
        
        k = cv2.waitKey(1)
        
        if k%256 == 27:
            print("Escape")
            break   
        elif k%256 == 32:
            fo = os.chdir(folder_path)
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name,frame)
            print("screenshot taken")            
            img_counter += 1
            
    cam.release()
    cv2.destroyAllWindows()
    
    dir_files = []
    
    dir_index = 'C:/Users/Rhyan Shah/Documents/GitHub/cosmosfinal/Facial_Recognition/Datasets'
    
    for dir in os.scandir(dir_index):
        if dir.is_dir():
            dir_files.append(dir)

    picture_files = []
    dir_files_cropped = []
    img_numpy_array_list = []
    
    
    for dir in os.listdir(dir_index):
        folder_name = os.path.join(dir_index, dir)
        for file in os.listdir(folder_name):
            if file.endswith(".png"):
                picture_files.append(file)
                dir_files_cropped.append(dir)
                img = cv2.imread(os.path.join(folder_name, file), cv2.IMREAD_GRAYSCALE)
                img.resize((200,200))
                img_numpy_array = img_to_array(img)
                img_numpy_array_list.append(img_numpy_array)
    
    img_numpy_array_list = np.array(img_numpy_array_list)
    print(picture_files)
    print(dir_files_cropped)
    

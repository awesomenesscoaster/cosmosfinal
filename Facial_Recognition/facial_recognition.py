import cv2
import os
import numpy as np

import time

from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array

global picture_files
global dir_files_cropped
global integer_img_conversion
global img_numpy_array_list

scanning_process = False
guessing_process = False

user_name = input("What is your name? ")

folder_path =  folder_path = os.path.join('C:/Users/Rhyan Shah/Documents/GitHub/cosmosfinal/Facial_Recognition/Datasets', user_name)
# if user_name not in names:
#     scanning_process = True
#     names.append(user_name)
    
#     folder_path = os.path.join('C:/Users/Rhyan Shah/Documents/GitHub/cosmosfinal/Facial_Recognition/Datasets', user_name)
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
# else:
#     guessing_process = True
#     print('Welcome back:', user_name)

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    scanning_process = True
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
            
        def face_screenshot(num_of_pics):
            fo = os.chdir(folder_path)
            ss_counter = 0
            img_counter = 0
            while ss_counter < num_of_pics:
                img_name = "opencv_frame_{}.png".format(img_counter)
                cv2.imwrite(img_name,frame)
                print("screenshot taken")            
                img_counter += 1
                ss_counter += 1
                
        cv2.imshow("Facial Recognition", frame)
        
        k = cv2.waitKey(1)  
        
        #Escape Key
        if k%256 == 27:
            print("Escape")
            break   
        
        #Space Bar
        elif k%256 == 32:
            face_screenshot(15)
            
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
    integer_img_conversion = []
    
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
    
    name_counter = 0
    
    #FIgure this SHIT out so that there is actually x number of data outputted; rn there is only x-1 data shown
    #Turning corresponding "name" labels into integers for the CNN to actually work
    for i in range(0,len(dir_files_cropped)-1):
        if dir_files_cropped[i] == dir_files_cropped[i+1]:
            integer_img_conversion.append(name_counter)
        else:
            integer_img_conversion.append(name_counter)
            name_counter += 1
    integer_img_conversion.append(name_counter)
    
    print(integer_img_conversion)
    

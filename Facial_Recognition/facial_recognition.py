import cv2
import os
import numpy as np

from tensorflow import keras
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split

# global picture_files
# global dir_files_cropped
# global img_numpy_array_list
# global integer_img_conversion
# global model_cnn

scanning_process = False
guessing_process = False

user_name = input("What is your name? ")

folder_path = os.path.join('C:/Users/Rhyan Shah/Documents/GitHub/cosmosfinal/Facial_Recognition/Datasets', user_name)
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

model_cnn = keras.Sequential()

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
        fo = os.chdir(folder_path)
        #Escape Key
        if k%256 == 27:
            print("Escape")
            break   
        
        #Space Bar
        elif k%256 == 32:
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name,frame)
            print('Screenshot Taken')
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
    
    userCountL = []
    for i in integer_img_conversion:
        if i not in userCountL:
            userCountL.append(i)
    
    x_data = img_numpy_array_list
    y_data = np.array(integer_img_conversion)
    x_data = img_numpy_array_list.reshape(-1, 200, 200, 1)
    print(x_data.shape)
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.20, random_state=0)
    
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
    
    model_cnn.fit(x_train,y_train,epochs=2,validation_data=(x_test,y_test))
    
    print('Complete 2.0')

    test_lost, test_acc = model_cnn.evaluate(x_test, y_test)
    print(test_acc)

if guessing_process == True:
    cam = cv2.VideoCapture(0)
    cv2.namedWindow('Facial Recognition')
    while True:
        ret, frame = cam.read()
        if not ret:
            print('Failed')
        k = cv2.waitKey(1)
        if k%256 == 32:
            fo = os.chdir('C:/Users/Rhyan Shah/Documents/GitHub/cosmosfinal/Facial_Recognition/TestingData')
            img_name = 'opencv_frame.png'
            cv2.imwrite(img_name,frame)
            print('Screenshot taken')
        if k%256 == 27:
            print('Escape')
            break
        cv2.imshow("Facial Recognition", frame)
    cam.release()
    cv2.destroyAllWindows()
    
    test_path = 'C:/Users/Rhyan Shah/Documents/GitHub/cosmosfinal/Facial_Recognition/TestingData'
    img_name = 'opencv_frame.png'
    
    pic = cv2.imread(os.path.join(test_path, img_name), cv2.IMREAD_GRAYSCALE)
    pic.resize((200,200))
    pic = img_to_array(pic)
    pic = np.array(pic)
    pic = pic.reshape(-1,200,200,1)
    print(pic.shape)
    
    classes = []
    datasetFile = 'C:/Users/Rhyan Shah/Documents/GitHub/cosmosfinal/Facial_Recognition/Datasets'
    for dir in os.scandir(datasetFile):
        if dir.is_dir():
            classes.append(dir)
    
    classes = np.array(classes)
    pic_guess = model_cnn.predict(pic)
    pic_single = classes[np.argmax(pic_guess, axis = -1)]
    print(pic_single)
    
    os.remove(os.path.join(test_path,img_name))
    
    with open('C:/Users/Rhyan Shah//Documents/GitHub/cosmosfinal/Facial_Recognition/accuracy.txt', 'a') as fhand:
        check = input('Was this recognition correct? (y/n) ')
        if check == 'y':
            fhand.write('Right\n')
        elif check == 'n':
            fhand.write('Wrong\n')
        fhand.close()
    
    with open('C:/Users/Rhyan Shah//Documents/GitHub/cosmosfinal/Facial_Recognition/accuracy.txt', 'r') as fhand:
        num = 0
        den = 0
        for line in fhand:
            line.split()
            if 'Right' in line:
                num += 1
                den += 1
            elif 'Wrong' in line:
                den +=1 
                
        perc_acc = (num/den)*100
        print('Your feedback is valued, thank you for contributing to the data')
        print('Currently, the algorithm is', perc_acc, 'percent accurate.')
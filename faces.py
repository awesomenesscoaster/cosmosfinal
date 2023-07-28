import pandas as pd
import csv
from zipfile import ZipFile
from PIL import Image
import os

with ZipFile("/Users/ethanc/Documents/faces/triplets.csv", 'r') as zObject:
    zObject.extractall()

folder = os.listdir('CelebA FR Triplets')
for image in folder:
    img = Image.open('CelebA FR Triplets', str(image))
    img.thumbnail((200,200))
    display(img)





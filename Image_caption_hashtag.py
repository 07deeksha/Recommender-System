# Import all the required libraries

# System Libraries
import os, glob
from glob import glob
import pickle
from sys import getsizeof

# Date and Time
import datetime, time

# Data manipulation
import numpy as np
import pandas as pd
import collections, random, re
from collections import Counter

# Model building
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Read/Display  images
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import warnings

warnings.filterwarnings("ignore")

# tensorflow Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import plot_model

# NLP Libraries
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Import the dataset and read the image into a seperate variable

INPUT_PATH = r'C:\Users\Deeksha\Desktop\TrendPulse\Tp'
IMAGE_PATH = INPUT_PATH + 'images/'
CAPTIONS_FILE = INPUT_PATH + 'imgcap.txt'
OUTPUT_IMAGE_PATH = "../working/Img/"

#1.Import the dataset and read image & captions into two seperate variables
#2.Visualise both the images & text present in the dataset

all_imgs = glob(IMAGE_PATH + '*.jpg')
print("The total images present in the dataset: {}".format(len(all_imgs)))
print(all_imgs[0])


def plot_image(images, captions=None, cmap=None):
    f, axes = plt.subplots(1, len(images), sharey=True)
    f.set_figwidth(15)

    for ax, image in zip(axes, images):
        ax.imshow(io.imread(image), cmap)

# Plotting last 10 images
plot_image(all_imgs[8081:])



def extract_keywords(content):
    # Use NLTK to extract keywords from content
    tokens = nltk.word_tokenize(content)
    tags = nltk.pos_tag(tokens)
    keywords = [word for (word, tag) in tags if tag.startswith('NN') or tag == 'JJ' or tag == 'NNP']
    return keywords


def generate_hashtags(keywords):
    # Generate hashtags from keywords
    hashtags = ['#' + keyword for keyword in keywords]
    return hashtags


# Prompt the user to enter content
# content = input("Enter the content you want to generate hashtags for: ")
#
# # Extract keywords and generate hashtags
# keywords = extract_keywords(content)
# hashtags = generate_hashtags(keywords)
#
# # Print the resulting hashtags
# print("Generated hashtags:")
# for hashtag in hashtags:
#     print(hashtag)

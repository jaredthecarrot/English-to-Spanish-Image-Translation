# Main File Until Otherwise
# Importing Necessary Libraries and
# Modules from Keras and TensorFlow API
# Alongside Python Libraries
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import pathlib
import random
import string
import re
import numpy as np

import tensorflow.data as tf_data
import tensorflow.strings as tf_strings

import keras
from keras import layers
from keras import ops
from keras.layers import TextVectorization

file = keras.utils.get_file(
    fname="spa-eng.zip",
    origin="http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip",
    extract=True,
)
file = pathlib.Path(file).parent / "spa-eng" / "spa.txt"

# Separate the source sequence and target sequence
# Source = English, Target = Spanish

with open(file, encoding="utf8") as f:
    lines = f.read().split("\n")[:-1]
pairs = []
for line in lines:
    # Source and Target are Tab Delimited
    eng, spa = line.split("\t")
    # Tokenizing start and end of sequence
    spa = "[start] " + spa + " [end]"
    pairs.append((eng,spa))

print(pairs)
# Dataset should be partially tokenized, and made into a list of English
    # and Spanish translations
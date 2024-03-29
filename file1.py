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
# Dataset should be partially tokenized, and made into a list of English
    # and Spanish translations

random.shuffle(pairs)
# Split the pairs of sentences into training, validation, and test sets

number_validation_samples = int(0.20 * len(pairs))
number_training_samples = len(pairs) - 2 * number_validation_samples
training = pairs[:number_training_samples]
validation = pairs[number_training_samples: number_training_samples + number_validation_samples]
testing = pairs[number_training_samples + number_validation_samples :]

# Now we have training, validation, and testing pairs

# Vectorization will be performed using TextVectorization
# First, need to strip punctuation

strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

def standard(i_string):
    lc = tf_strings.lower(i_string)
    return tf_strings.regex_replace(lc, "[%s]" % re.escape(strip_chars), "")

eng_vec = TextVectorization(
    max_tokens = 15000,
    output_mode = "int",
    output_sequence_length = 20,
)

spa_vec = TextVectorization(
    max_tokens = 15000,
    output_mode = "int",
    output_sequence_length = 21, # Additional integer for the Spanish punc
    standardize = standard,
)

training_eng_text = [pair[0] for pair in training]
training_spa_text = [pair[1] for pair in training]
eng_vec.adapt(training_eng_text)
spa_vec.adapt(training_spa_text)

# After each set is vectorized, formatting is necessary

def format(eng, spa):
    eng = eng_vec(eng)
    spa = spa_vec(spa)
    return (
        {
            "encoder_inputs": eng,
            "decoder_inputs":spa[:, :-1],
        },
        spa[:, 1:],
    )

def create(pairs):
    eng_text, spa_text = zip(*pairs)
    eng_text = list(eng_text)
    spa_text = list(spa_text)
    dset = tf_data.Dataset.from_tensor_slices((eng_text, spa_text))
    dset = dset.batch(64)
    dset = dset.map(format)
    return dset.cache().shuffle(2048).prefetch(16)

training_dset = create(training)
validation_dset = create(validation)

# Now our training and validation datasets are complete
# Our Transformer model will have an encoder, decoder, and positional embedding
 
class TransformerEncoder(layers.Layer):
    def __init__(self, embed, dense, num_heads, **kwargs):
        super().__init__(**kwargs)
        
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

vocab = 15000
seq_len = 20
batsize = 64
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

import keras.ops as ops

class TransformerEncoder(layers.Layer):
    def __init__(self, embed, dense, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed = embed
        self.dense = dense
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads = num_heads, key_dim = embed
        )
        self.projection = keras.Sequential(
            [
                layers.Dense(dense, activation = "relu"),
                layers.Dense(embed),
            ]
        )
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.supports_masking =  True

        def call(self, inputs, mask = None):
            if mask is not None:
                padmask = ops.cast(mask[:, None, :], dtype = "int32")
            else:
                padmask = None
            
            attention_output = self.attention(
                query = inputs,
                value = inputs,
                key = inputs,
                attention_mask = padmask
            )
            input_projection = self.layernorm1(inputs + attention_output)
            output_projection = self.projection(input_projection)
            return self.layernorm2(input_projection + output_projection)

        def get_config(self):
            config = super().get_config()
            config.update(
                {
                    "embed": self.embed,
                    "dense": self.dense,
                    "num_heads": self.num_heads,
                }
            )
            return config

# Positional Embedding

class PositionalEmbedding(layers.Layer):
    def __init__(self, seq_len, vocab, embed, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim = vocab,
            output_dim = embed,
        )
        self.position_embeddings = layers.Embedding(
            input_dim = seq_len,
            output_dim = embed,
        )
        self.seq_len = seq_len
        self.vocab = vocab
        self.embed = embed

    def call(self, inputs):
        len = ops.shape(inputs)[-1]
        positions = ops.arange(0, len, 1)
        embedtoks = self.token_embeddings(inputs)
        embedpos = self.position_embeddings(positions)
        return embedtoks + embedpos

    def comp_mask(self, inputs, mask = None):
        if mask is None:
            return None
        else:
            return ops.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "seq_len": self.seq_len,
                "vocab": self.vocab,
                "embed": self.embed,
            }
        )
        return config
    
class TransformerDecoder(layers.Layer):
    def __init__(self, embed, latent, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed = embed
        self.latent = latent
        self.num_heads = num_heads
        self.attention1 = layers.MultiHeadAttention(
            num_heads = num_heads,
            key_dim = embed,
        )
        self.attention2 = layers.MultiHeadAttention(
            num_heads = num_heads,
            key_dim = embed
        )
        self.projection = keras.Sequential(
            [
                layers.Dense(latent, activation = "relu"),
                layers.Dense(embed),
            ]
        )
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.layernorm3 = layers.LayerNormalization()
        self.supp_mask = True

    def call(self, inputs, encode_out, mask = None):
        casmask = self.get_cam(inputs)
        if mask is not None:
            padmask = ops.cast(mask[:, None, :], dtype = "int32")
            padmask = ops.minimum(padmask, casmask)
        else:
            padmask = None

        attention_output1 = self.attention1(
            query = inputs,
            value = inputs,
            key = inputs,
            attention_mask = casmask,
        )
        out1 = self.layernorm1(inputs + attention_output1)

        attention_output2 = self.attention2(
            query = out1,
            value = encode_out,
            key = encode_out,
            attention_mask = padmask,
        )
        out2 = self.layernorm2(out1 + attention_output2)

        projection = self.projection(out2)
        return self.layernorm3(out2 + projection)

    def get_cam(self, inputs):
        input_shape = ops.shape(inputs)
        batsize, seq_len = input_shape[0], input_shape[1]
        i = ops.arange(seq_len)[:, None]
        j = ops.arange(seq_len)
        mask = ops.cast(i >= j, dtype = "int32")
        mask = ops.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = ops.concatenate(
            [ops.expand_dims(batsize, -1), ops.convert_to_tensor([1,1])],
            axis = 0,
        )
        return ops.tile(mask, mult)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed": self.embed,
                "latent": self.latent,
                "num_heads": self.num_heads,
            }
        )
        return config

embed_dim = 256
latent_dim = 2048
num_heads = 8

encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
x = PositionalEmbedding(seq_len, vocab, embed_dim)(encoder_inputs)
encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads)(x)
encoder = keras.Model(encoder_inputs, encoder_outputs)

decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
encoded_seq_inputs = keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")
x = PositionalEmbedding(seq_len, vocab, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, encoded_seq_inputs)
x = layers.Dropout(0.5)(x)
decoder_outputs = layers.Dense(vocab, activation="softmax")(x)
decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

decoder_outputs = decoder([decoder_inputs, encoder_outputs])
transformer = keras.Model(
    [encoder_inputs, decoder_inputs], decoder_outputs, name="transformer"
)

epochs = 1  # This should be at least 30 for convergence

transformer.summary()
transformer.compile(
    "rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
transformer.fit(training, epochs=epochs, validation_data=validation)

spa_vocab = spa_vec.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = 20


def decode_sequence(input_sentence):
    tokenized_input_sentence = eng_vec([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = spa_vec([decoded_sentence])[:, :-1]
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])

        # ops.argmax(predictions[0, i, :]) is not a concrete value for jax here
        sampled_token_index = ops.convert_to_numpy(
            ops.argmax(predictions[0, i, :])
        ).item(0)
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token

        if sampled_token == "[end]":
            break
    return decoded_sentence


test_eng_texts = [pair[0] for pair in test_pairs]
for _ in range(30):
    input_sentence = random.choice(test_eng_texts)
    translated = decode_sequence(input_sentence)

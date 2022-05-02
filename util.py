from __future__ import print_function
from cProfile import label
from msilib import sequence

import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from matplotlib.colors import LogNorm
from tqdm import tqdm              # to track progress of loops
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import random 
def name_to_seq(name,tokenizer):
    return [tokenizer.texts_to_sequences(c)[0][0] for c in name]
def get_sequences(names,tokenizer):
    sequences=[]
    for name in names:
        seq = name_to_seq(name,tokenizer)
        if len(seq) >= 2:
            sequences += [seq[:i] for i in range(2, len(seq)+1)]
    return sequences
def create_tokenizer(data):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='~!@#$%^&*()-_=+{[}]:;\\|\?/.,123456789', split='\n')
    tokenizer.fit_on_texts(data)
    return tokenizer
def generate_char_index_maps(tokenizer):
    char_to_index = tokenizer.word_index
    index_to_char = dict((v,k) for k, v in char_to_index.items())
    return char_to_index,index_to_char
def generate_datasets(sequences,max_len):
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, padding='pre',
        maxlen=max_len
    )
    x, y = padded_sequences[:, :-1], padded_sequences[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    return  x_train, x_test, y_train, y_test
def fuzz_string(string):
    new_string = ''.join(chr(ord(letter)+random.randrange(3)-1) for letter in string)
    print('Fuzzed Seed: ' +new_string)

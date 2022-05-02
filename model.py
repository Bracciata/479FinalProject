from __future__ import print_function
from cProfile import label
from ctypes import util
from tensorflow.keras.optimizers import RMSprop

import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from matplotlib.colors import LogNorm
from tqdm import tqdm              # to track progress of loops
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from util import *


class Model:
    def __init__(self, num_char, max_len):
        self.model = keras.models.Sequential([
            keras.layers.Embedding(num_char, 8, input_length=max_len-1),
            keras.layers.Conv1D(64, 5, strides=1, activation='tanh',
                                padding='causal', input_shape=[None, 1]),
            keras.layers.MaxPool1D(2),
            keras.layers.LSTM(32, return_sequences=True),
            keras.layers.GRU(32),
            keras.layers.Dense(num_char, activation='softmax')
        ])

        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=RMSprop(lr=0.01)
        )

        self.model.summary()
        # code from tensorflow to save model
        self.checkpoint_path = "training/checkpoint.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

        # Create a callback that saves the model's weights
        self.checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                                      save_weights_only=True,
                                                                      verbose=1)
        if os.path.exists(self.checkpoint_path[:-5]):
            self.model.load_weights(self.checkpoint_path)
        else:
            print("Couldn't find the last checkpoint")

    def train(self, x_train, y_train, x_test, y_test, epochs):
        self.history = self.model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test), epochs=epochs,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy', patience=5), self.checkpoint_callback
            ]
        )

    def generate_names(self, seed, max_len, index_to_char, tokenizer):
        for _ in range(30):
            seq = name_to_seq(seed, tokenizer)
            padded = tf.keras.preprocessing.sequence.pad_sequences(
                [seq], padding='pre', maxlen=max_len-1, truncating='pre')
            pred = self.model.predict(padded)[0]
            pred_char = index_to_char[tf.argmax(pred).numpy()]
            seed += pred_char

            if pred_char == '\t':
                break
        return seed

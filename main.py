from __future__ import print_function
from cProfile import label

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
from model import Model
from util import *
with open('s2.txt','r') as f:
    data = f.read()


tokenizer = create_tokenizer(data)
char_to_index,index_to_char=generate_char_index_maps(tokenizer)
names = data.splitlines()
sequences =get_sequences(names,tokenizer)
max_len = max([len(x) for x in sequences])
x_train, x_test, y_train, y_test = generate_datasets(sequences,max_len)
num_char = len(char_to_index.keys()) + 1

model = Model(num_char,max_len)
model.train(x_train,y_train,x_test,y_test,10)
looping = True
seen_phrases = set()
while looping:
    seed = input("Enter letter/letters/word: ")
    print("Generated name is: ")
    phrase = model.generate_names(seed,max_len,index_to_char,tokenizer)
    direction_to_choose = 0
    while  phrase in seen_phrases:
        if direction_to_choose == 0:
            direction_to_choose = input('To generate a random name would you like to fuzz(1) or train the model(2)')

            while direction_to_choose !='1' or direction_to_choose!='2':
                direction_to_choose = input('To generate a random name would you like to fuzz(1) or train the model(2)')
        if direction_to_choose == '1':
            seed = fuzz_string(phrase)
        else:
            model.train(x_train,y_train,x_test,y_test,1)
            phrase = model.generate_names(seed,max_len,index_to_char,tokenizer)
    seen_phrases.add(phrase)
    print(phrase)
    again = input("Would you like to generate another name(yes or no): ")
    if 'n' in again.lower():
        looping=False
    

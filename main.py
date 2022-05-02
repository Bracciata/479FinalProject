from __future__ import print_function
from cProfile import label

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
from model import Model
from util import *
with open('s2.txt', 'r') as f:
    data = f.read()


tokenizer = create_tokenizer(data)
char_to_index, index_to_char = generate_char_index_maps(tokenizer)
names = data.splitlines()
sequences = get_sequences(names, tokenizer)
max_len = max([len(x) for x in sequences])
x_train, x_test, y_train, y_test = generate_datasets(sequences, max_len)
num_char = len(char_to_index.keys()) + 1

model = Model(num_char, max_len)
#model.train(x_train, y_train, x_test, y_test, 10)
looping = True
seen_phrases = set()

index = 0
names = ['Khalid','Josh','Sarat','Shubham','Sanat','Thomas','Alex','Mitchell','Ujwal','Bidhan','Husein','Logan','Uri','Andrew','Ray','Didarul','Pascal','Jack','Houman','Mohammad','Richard','Daniel','Joey','Jeffrey','Karnav','Shiva','Charlie','Towfiqur','Amit','David','Steven','Puranjit','Jayden','Tessa','Prashant','Raoul','Nate','Kantilata','Kendric','Serigne','Aime','Connor','Eric','Thomas','Anh','Thomas','Andrew','Junxiao']
output = []
for name in names:
    seen_phrases.add(name.lower().strip())
while index< len(names):
    seed = names[index]
    print("Generated name is: ")
    phrase = model.generate_names(seed, max_len, index_to_char, tokenizer).strip().lower()
    direction_to_choose = 0
    tries = 0
    while  phrase in seen_phrases or (len(phrase)<len(names[index])+3 and tries<3):
        model.train(x_train, y_train, x_test, y_test, 1)
        phrase = model.generate_names(
            seed, max_len, index_to_char, tokenizer).strip().lower()
        tries+=1
    seen_phrases.add(phrase)
    print(phrase)
    #again = input("Would you like to generate another name(yes or no): ")
    #if 'n' in again.lower():
    #    looping = False
    index+=1
    output+=[phrase]

print(output)
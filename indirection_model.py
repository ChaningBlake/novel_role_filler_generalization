#!/usr/bin/env python3

'''
November 2019
Blake Mullinax
Original code from Mike Jovanovich 2017
https://github.com/mpjovanovich/indirection

TODO: [Description]

---Usage---
arguments: 
seed, ntasks, nstripes, nfillers, nroles,
state_cd: Conjunction or Disjunction
sid_cd: Conjunction or Disjunction
interstripe_cd: Conjuction or Disjunction
use_sids_input:
use_sids_output:
  
./indirection_model.py 1 100000 3 10 3 C C D F F
'''
import keras
import sys
import numpy as np
from hrr import *

# Create hrr vectors for each of the inputs to the neural net
N = 1024
hrri = hrri(N) # Identity vector for general use
args = LTM(N, normalized=True)
args.lookup("agent*verb*patient*open*close*store*query")

# TODO: Get outputs from encoder and store them here
# encodings = np.zeros((3,6))
encodings = ["cat", "ate", "toy"]

# -- Input Gates --
# Set up the input gate neural networks (one for each wm slot)
ig = np.empty(3, dtype=object)
for i in range(3):
    ig[i] = keras.Sequential()
    # The net will be choosing the highest output for 
    # reinforcement learning, so output is just 1
    output_size = 1
    input_size = args.N
    ig[i].add(keras.layers.Dense(output_size,
                                 input_shape=[input_size]))
    ig[i].compile(loss=keras.losses.mse,
                  optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])

# -- Output Gates --
# Set up the output gate neural network (one for each wm slot)
og = np.empty(3, dtype=object)
for i in range(3):
    og[i] = keras.Sequential()
    # See input size and output size from the Input Gate
    og[i].add(keras.layers.Dense(output_size,
                              input_shape=[input_size]))
    og[i].compile(loss=keras.losses.mse,
                  optimizer=keras.optimizers.Adam(lr=.001),
                  metrics=['accuracy'])

# -- Train --
# Set the parameters
batch_size = 1
epochs = 10
validation_split = 0.5

# Store HRRs for the role*open/close*store combo
input_combo = np.array([hrri, hrri])
input_combo[0] = args.lookup("agent*open*store")
input_combo[1] = args.lookup("agent*close*store")

# Size: 3 -> # of ig og gates,
#       2 -> hrr convolved with open or close, 
ig_vals = np.empty((3,2), dtype=float)
og_vals = np.empty((3,2), dtype=float)

# Iterate through each of the words
for i in range(3):
    # Iterate through each of the IGs and OGs
    for j in range(2):
        ig_vals[i,j] = ig.predict(input_combo[k])
        og_vals[i,j] = og.predict(input_combo[k])
        
# max of result trains the model from the previous
# time step
if i-1 > 0:
    ig[i-1].fit(max(ig_vals[i,:]))
    og[i-1].fit(max(og_vals[i,:]))

# -- Test --
test = args.lookup("agent*open*store")
test = np.expand_dims(test,axis=0)
print(ig[0].predict(test)[0,0])
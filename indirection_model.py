#!/usr/bin/env python3.6

'''
November 2019
Blake Mullinax
Original code from Mike Jovanovich 2017
https://github.com/mpjovanovich/indirection
/outputgate_no_ac_nn.r

TODO: [Description]

---Usage---
arguments: 
seed, ntasks, nstripes, nfillers, nroles,
state_cd: Conjunction or Disjunction
sid_cd: Conjunction or Disjunction
interstripe_cd: Conjuction or Disjunction
use_sids_input:
use_sids_output:
  
./indirection_model.py 
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
max_tasks = 100000
wm = np.empty(3, dtype=str)

# TODO: Get outputs from encoder and store them here
# encodings = np.zeros((3,6))
encodings = ["cat", "ate", "toy"]

#-- Input Gates --
# Set up the input gate neural networks (one for each wm slot)
ig = np.empty(3, dtype=object)
for i in range(3):
    ig[i] = keras.models.Sequential()
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
    og[i] = keras.models.Sequential()
    # See input size and output size from the Input Gate
    og[i].add(keras.layers.Dense(output_size,
                              input_shape=[input_size]))
    og[i].compile(loss=keras.losses.mse,
                  optimizer=keras.optimizers.Adam(lr=.001),
                  metrics=['accuracy'])

# -- Train --
# Set the parameters
success_reward = 1
default_reward = 0
block_size = 200
batch_size = 1
epochs = 10
validation_split = 0.5
accuracy = 0
cur_task = 0
cur_block_task = 0
# Store HRRs for the role*(open/close)*store combo
input_combo = np.empty((3,2), dtype=object)
roles = ["agent", "verb", "patient"]
action = ["*open*", "*close*"]
for i in range(3):
    for j in range(2):
        input_combo[i,j] = roles[i] + action[j] + "store"
# training loop
while accuracy < 95 and cur_task < max_tasks:

    if cur_task % 200 == 0:
        block_tasks_correct = 0


    # Size: 4 -> # of time steps (3 words plus a query)
    #       3 -> # of ig og gates,
    #       2 -> hrr convolved with open or close, 
    ig_vals = np.empty((4,3,2), dtype=float)
    og_vals = np.empty((4,3,2), dtype=float)
    max_val = np.empty((4,3,2), dtype=int)

    # i -> The current word
    # j -> The current working memory slot
    # k -> open or close value
    for i in range(3):
        for j in range(3):
            for k in range(2):
                ig_vals[i,j,k] = ig[j].predict(np.expand_dims(args.lookup(input_combo[i,k]),axis=0))
                og_vals[i,j,k] = og[j].predict(np.expand_dims(args.lookup(input_combo[i,k]),axis=0))
            max_val[i,j,:] = [np.argmax(ig_vals[i,j,:]), np.argmax(og_vals[i,j,:])]
            # if the input gate is open, store encoding in wm
            if max_val[i,j,0] == 0:
                wm[i] = encodings[i]
                
            # max of result trains the model from the previous
        # time step
        if i != 0:
            # Given the input from the last time step, train the model with the
            # highest output from the current timestep
            for wm in range(3):
                ig[wm].fit(np.expand_dims(args.lookup(input_combo[i-1,max_val[i-1,wm,0]]),axis=0),
                           max(ig_vals[i,wm,:]))
                og[wm].fit(np.expand_dims(args.lookup(input_combo[i-1,max_val[i-1,wm,1]]),axis=0), 
                           max(og_vals[i,wm,:]))

    # -- Query --
    query_role = np.random.choice(roles)
    query_hrr = [args.lookup(query_role + "*query*open"),
                 args.lookup(query_role + "*query*close")]
    role_dict = {"agent":0, "verb":1, "patient":2}
    # for each og predict query*open and query*close
    # i -> current working memory slot
    # j -> open or close value
    for i in range(3):
        for j in range(2):
            og_vals[3,i,j] = og[i].predict(np.expand_dims(query_hrr[j],axis=0))
        # train net
        # if open and storing the correct thing in wm
        if (og_vals[3,i,0] > og_vals[3,i,1]) and (wm[i] ==
        encodings[role_dict[query_role]]):
                ig[i].fit(np.expand_dims(args.lookup(input_combo[max_val[2,i,0]]),axis=0),
                           success_reward)
                og[i].fit(np.expand_dims(args.lookup(input_combo[max_val[2,i,1]]),axis=0), 
                           rsuccess_reward)
        else:
            ig[i].fit(np.expand_dims(args.lookup(input_combo[max_val[2,i,0]]),axis=0),
                       default_reward)
            og[i].fit(np.expand_dims(args.lookup(input_combo[max_val[2,i,1]]),axis=0), 
                       default_reward)
    cur_task += 1
    if cur_task % 200 == 0:
        print("Tasks Complete:", cur_task)
        print("Block Accuracy: %.2f"%((block_tasks_correct/200)*100))
    print(cur_task)

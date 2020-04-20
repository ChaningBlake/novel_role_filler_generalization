#!/usr/bin/env python3

'''
November 2019
Blake Mullinax
Original code from Mike Jovanovich 2017
https://github.com/mpjovanovich/indirection
Rscript outputgate_no_ac_nn.r 1 100000 3 10 3 C C D F F SG

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
from os import system
from hrr import *

# Create hrr vectors for each of the inputs to the neural net
N = 1024
hrri = hrri(N) # Identity vector for general use
args = LTM(N, normalized=True)
args.lookup("agent*verb*patient*open*close*store*query")
max_tasks = 100000
# These are the working memory slots
wm = np.empty(3, dtype=object)

# TODO: Get outputs from encoder and store them here
# encodings = np.zeros((3,6))
# encodings = ["cat", "ate", "toy"]
encodings = np.loadtxt("SG-10-train.txt", dtype=object)
for i in range(encodings.shape[0]):
    encodings[i] = np.array(list(encodings[i]))
    

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
                                 input_shape=[input_size],
                                 use_bias=False))
    ig[i].compile(loss=keras.losses.huber_loss,
                  optimizer=keras.optimizers.SGD(lr=0.1),
                  metrics=['accuracy'])

# -- Output Gates --
# Set up the output gate neural network (one for each wm slot)
og = np.empty(3, dtype=object)
for i in range(3):
    og[i] = keras.models.Sequential()
    # See input size and output size from the Input Gate
    og[i].add(keras.layers.Dense(output_size,
                                 input_shape=[input_size],
                                 use_bias=False))
    og[i].compile(loss=keras.losses.huber_loss,
                  optimizer=keras.optimizers.SGD(lr=.1),
                  metrics=['accuracy'])

# -- Train --
# Set the parameters
success_reward = 1
default_reward = 0
block_size = 200
batch_size = 1
epochs = 10
accuracy = 0
cur_task = 0
cur_block_task = 0
block_tasks_correct = 0
epsilon = .025
lmda = .9
bias = 1.0

# Store HRRs for the role*(open/close)*store combo
input_combo = np.empty((4,2), dtype=object)
roles = ["agent", "verb", "patient"]
action = ["*open*", "*close*"]
for i in range(3):
    for j in range(2):
        input_combo[i,j] = roles[i] + action[j] + "store"

# -----------------------        
#     training loop
# -----------------------
system('clear')
print("Beginning training....\n")
while accuracy < 95.0 and cur_task < max_tasks:
    # This is to choose a random sentence from the training
    sample = np.random.randint(0,200)

    # Size: 4 -> # of time steps (3 words plus a query)
    #       3 -> # of ig og gates,
    #       2 -> hrr convolved with open or close,
    ig_vals = np.empty((4,3,2), dtype=float)
    og_vals = np.empty((4,3,2), dtype=float)
    # In this case, 2 -> the index of the max value of ig_vals (index 0) or og_vals
    max_val = np.empty((4,3,2), dtype=int)

    # --- First Three Time Steps ---
    # i -> The current word (time step)
    # j -> The current working memory slot
    # k -> open or close value
    for i in range(3):
        for j in range(3):
            for k in range(2):
                ig_vals[i,j,k] = ig[j].predict(np.array([args.lookup(input_combo[i,k])])) + bias
                og_vals[i,j,k] = og[j].predict(np.array([args.lookup(input_combo[i,k])])) + bias
            max_val[i,j,:] = [np.argmax(ig_vals[i,j,:]), np.argmax(og_vals[i,j,:])]
            # Episilon soft policy
            # for ig
            if np.random.random() < epsilon:
                # flip 0 to 1 and 1 to 0
                max_val[i,j,0] = abs(max_val[i,j,0] - 1)
            # for og
            if np.random.random() < epsilon:
                max_val[i,j,1] = abs(max_val[i,j,1] - 1)
            # if the input gate is open, store encoding in wm
            if max_val[i,j,0] == 0:
                wm[i] = encodings[sample][i]
                
        # max of result trains the model from the previous
        # time step
        if i != 0:
            # Given the input from the last time step, train the model with the
            # highest output from the current timestep
            for wmslot in range(3):
                ig[wmslot].fit(np.array([args.lookup(input_combo[i-1,max_val[i-1,wmslot,0]])]),
                           np.array([max(ig_vals[i,wmslot,:]) - bias]),
                           verbose=0)
                og[wmslot].fit(np.array([args.lookup(input_combo[i-1,max_val[i-1,wmslot,1]])]),
                           np.array([max(og_vals[i,wmslot,:]) - bias]),
                           verbose=0)

    # -- Query --
    query_role = np.random.choice(roles)
    query_hrr = [args.lookup(query_role + "*query*open"),
                 args.lookup(query_role + "*query*close")]
    role_dict = {"agent":0, "verb":1, "patient":2}
    # tracks stats for the reward. Only one gate should be open and the role should be matched. 
    gates_open = 0
    role_is_matched = False
    # for each og predict query*open and query*close
    # i -> current working memory slot
    # j -> open or close value
    for i in range(3):
        for j in range(2):
            ig_vals[3,i,j] = ig[i].predict(np.array([query_hrr[j]])) + bias
            og_vals[3,i,j] = og[i].predict(np.array([query_hrr[j]])) + bias
        max_val[3,i,:] = [np.argmax(ig_vals[3,i,:]), np.argmax(og_vals[3,i,:])]
        # Epsilon soft policy
        # for ig 
        if np.random.random() < epsilon:
            # flip 0 to 1 and 1 to 0
            max_val[3,i,0] = abs(max_val[i,j,0] - 1)
        # for og
        if np.random.random() < epsilon:
            max_val[3,i,1] = abs(max_val[i,j,1] - 1)
        # train net
        # train the third time step with the max output from the query step
        ig[i].fit(np.array([args.lookup(input_combo[2, max_val[2,i,0]])]),
                  np.array([ig_vals[3,i,max_val[3,i,0]] - bias]),
                  verbose=0)
        og[i].fit(np.array([args.lookup(input_combo[2, max_val[2,i,1]])]),
                  np.array([og_vals[3,i,max_val[3,i,1]] - bias]),
                  verbose=0)
        
        
        # if open 
        # AND storing the correct thing in wm
        # AND it hasn't been stored before
        if (og_vals[3,i,0] > og_vals[3,i,1]):
            # args.lookup(input_combo[role_dict[query_role],max_val[2,i,0]])
                gates_open += 1
                if (wm[i] == encodings[sample][role_dict[query_role]]):
                    role_is_matched = True
                
                
                
    if (gates_open ==1) and (role_is_matched == True):
        block_tasks_correct += 1
        reward = success_reward
    else:
        reward = default_reward
        
    for i in range(3):
        ig[i].fit(np.array([query_hrr[max_val[3,i,0]]]),
                   np.array([reward]),
                   verbose=0)
        og[i].fit(np.array([query_hrr[max_val[3,i,1]]]),
                   np.array([reward]),
                   verbose=0)     
            
    cur_task += 1
    if cur_task % 200 == 0:
        print()
        print("Tasks Complete:", cur_task)
        print("Block Tasks Correct:", block_tasks_correct)
        accuracy = (block_tasks_correct/200)*100
        print("Block Accuracy: %.2f"%accuracy)
        print()
        block_tasks_correct = 0


# -----------------
#   Testing loop
# -----------------
block_tasks_correct = 0
encodings = np.loadtxt("SG-10-test.txt", dtype=object)
for sentence in encodings:
    # This is to choose a random sentence from the training
    sample = np.random.randint(0,100)

    # Size: 4 -> # of time steps (3 words plus a query)
    #       3 -> # of ig og gates,
    #       2 -> hrr convolved with open or close,
    ig_vals = np.empty((4,3,2), dtype=float)
    og_vals = np.empty((4,3,2), dtype=float)
    # In this case, 2 -> the index of the max value of ig_vals (index 0) or og_vals
    max_val = np.empty((4,3,2), dtype=int)

    # --- First Three Time Steps ---
    # i -> The current word (time step)
    # j -> The current working memory slot
    # k -> open or close value
    for i in range(3):
        for j in range(3):
            for k in range(2):
                ig_vals[i,j,k] = ig[j].predict(np.array([args.lookup(input_combo[i,k])])) + bias
                og_vals[i,j,k] = og[j].predict(np.array([args.lookup(input_combo[i,k])])) + bias
            max_val[i,j,:] = [np.argmax(ig_vals[i,j,:]), np.argmax(og_vals[i,j,:])]
            # if the input gate is open, store encoding in wm
            if max_val[i,j,0] == 0:
                wm[i] = sentence[i]

    # -- Query --
    query_role = np.random.choice(roles)
    query_hrr = [args.lookup(query_role + "*query*open"),
                 args.lookup(query_role + "*query*close")]
    # tracks stats for success. Only one gate should be open and the role should be matched. 
    gates_open = 0
    role_is_matched = False
    # for each og predict query*open and query*close
    # i -> current working memory slot
    # j -> open or close value
    for i in range(3):
        for j in range(2):
            ig_vals[3,i,j] = ig[i].predict(np.array([query_hrr[j]])) + bias
            og_vals[3,i,j] = og[i].predict(np.array([query_hrr[j]])) + bias
        max_val[3,i,:] = [np.argmax(ig_vals[3,i,:]), np.argmax(og_vals[3,i,:])]
        
        if (og_vals[3,i,0] > og_vals[3,i,1]):
            # args.lookup(input_combo[role_dict[query_role],max_val[2,i,0]])
            gates_open += 1
            if (wm[i] == sentence[role_dict[query_role]]):
                role_is_matched = True
        # if open and storing the correct thing in wm
    if (gates_open == 1) and (role_is_matched == True):
        # args.lookup(input_combo[role_dict[query_role],max_val[2,i,0]])
        block_tasks_correct += 1
                
print("Generalization Accuracy:", block_tasks_correct)   

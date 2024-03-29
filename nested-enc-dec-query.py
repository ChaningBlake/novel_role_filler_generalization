#!/usr/bin/env python3

import sys
if len(sys.argv) != 5:
    print()
    print("Usage: %s <Corpus> <TrainingSet> <TestingSet> <#querySteps>"%(sys.argv[0]))
    print()
    sys.exit(1)
    
import keras
import numpy as np
from os import system
import encode # My module
import pickle, random, string

def my_mse(vec1, vec2):
    print(vec1[0])
    print()
    print(vec2[0])
    print()
    print("Token_1 Loss: ", ((vec1[0]-vec2[0])**2).mean(axis=None))
    print("Token_2 Loss: ", ((vec1[1]-vec2[1])**2).mean(axis=None))
    input("Press Enter to Conitnue...")
    return


# Import outer decoder
with open('models/encoder_len5.json', 'r') as encoder_file, open('models/decoder_len5.json', 'r') as decoder_file:
    encoder_json = encoder_file.read()
    decoder_json = decoder_file.read()
outer_encoder = keras.models.model_from_json(encoder_json)
outer_decoder = keras.models.model_from_json(decoder_json)
outer_encoder.load_weights("models/encoder_len5.h5")
outer_decoder.load_weights("models/decoder_len5.h5")
keras.utils.plot_model(outer_encoder, to_file="word_enc.png", show_shapes=True)
keras.utils.plot_model(outer_decoder, to_file="word_dec.png", show_shapes=True)

# Prepare input
corpus = np.loadtxt(sys.argv[1], dtype=object)

# Each letter that represents a role will be mapped to the encoding for a
# random word from the corpus.
encoded_mapping = {}
selected_words = {}
for letter in string.ascii_lowercase[:10]:
    # Store the letter with the word for use in testing
    onehot = encode.onehot(random.choice(corpus))
    selected_words[letter] = np.concatenate((onehot.copy(), encode.onehot("stop").reshape(1,28)))
    encoded_mapping[letter] = outer_encoder.predict(np.array([onehot]))
encoded_mapping["blank"] = np.zeros((1,28))

x_train = []
roles = np.loadtxt(sys.argv[2], dtype=object)
query_tokens = []
role_encoding = keras.utils.to_categorical([i for i in range(3)])
y_train = []
x_role_input = []
nquery_steps = int(sys.argv[4])
for sentence in roles:
    role_index = np.random.randint(0,high=3) # pick a random role to query
    x_role_input.append(np.vstack((np.zeros((3,3)), 
                                     np.tile(role_encoding[role_index], (nquery_steps,1)))))
    x_train.append(np.vstack(([encoded_mapping[letter] for letter in sentence], np.zeros((nquery_steps,2,1,50)))))
    y_train.append([encoded_mapping[sentence[role_index]]])
    
x_role_input = np.array(x_role_input)
x_train = np.array(x_train) # shape (n, 3 + nquery_steps, 2, 1, 50)

t1 = x_train[:,:,0,0,:] # new shape (n, 3 + nquery_steps,50)
t2 = x_train[:,:,1,0,:] # " '' "
y_train = np.array(y_train)[:,:,:,0,:] #(n, 4, 2, 50)

# 4 time steps. pre
pre_t1 = np.concatenate((np.zeros((x_train.shape[0],1,50)), y_train[:,:,0,:]), axis = 1)
pre_t2 = np.concatenate((np.zeros((x_train.shape[0],1,50)), y_train[:,:,1,:]), axis = 1)
post_t1 = np.concatenate((y_train[:,:,0,:], np.zeros((x_train.shape[0],1,50))), axis = 1)
post_t2 = np.concatenate((y_train[:,:,1,:], np.zeros((x_train.shape[0],1,50))), axis = 1)



# Start or stop tokens
s_s = {"start": [0,1], "stop": [1,0], "none": [0,0]}
pre_t3 = np.zeros((x_train.shape[0], 2, 2))
post_t3 = np.copy(pre_t3)
pre_t3[:,0,:] = s_s["start"]
post_t3[:,-1,:] = s_s["stop"]




# Construct Inner Encoder Decoder
# ----------------------------------
# Encoder Construction
hidden_size = 300
encoder_input_t1 = keras.layers.Input(shape=(None, t1.shape[2]), name="enc_token_1")
encoder_input_t2 = keras.layers.Input(shape=(None, t1.shape[2]), name="enc_token_2")
encoder_query_input = keras.layers.Input(shape=(None, x_role_input.shape[2]), name="query_role_input")
encoder_input = keras.layers.Concatenate()([encoder_input_t1, encoder_input_t2, encoder_query_input])

encoder_hidden = keras.layers.LSTM(hidden_size, return_state=True, name="encoder")
# Tie them together
encoder_output, enc_state_h, enc_state_c = encoder_hidden(encoder_input)
# Don't need the encoder outputs, just need the states.
encoder_states = [enc_state_h, enc_state_c]

## Decoder Construction
decoder_input_t1 = keras.layers.Input(shape=(None, t1.shape[2]), name="dec_token_1")
decoder_input_t2 = keras.layers.Input(shape=(None, t1.shape[2]), name="dec_token_2")
decoder_input_t3 = keras.layers.Input(shape=(None, 2), name="dec_start/stop")
decoder_input = keras.layers.Concatenate()([decoder_input_t1, decoder_input_t2, decoder_input_t3])

decoder_hidden = keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True, name="decoder")
# Tie it together
decoder_hidden_output, decoder_state_h, decoder_state_c = decoder_hidden(decoder_input,
                                                                         initial_state=encoder_states)
decoder_dense_t1 = keras.layers.Dense(t1.shape[2], activation='linear', name="token_1")
decoder_dense_t2 = keras.layers.Dense(t1.shape[2], activation='linear', name="token_2")
decoder_dense_t3 = keras.layers.Dense(2, activation='sigmoid', name="start/stop")
# Connect output to hidden
decoder_output = [decoder_dense_t1(decoder_hidden_output), decoder_dense_t2(decoder_hidden_output), decoder_dense_t3(decoder_hidden_output)]


# Finally, tie everything into a model
model = keras.Model([encoder_input_t1, encoder_input_t2, encoder_query_input, decoder_input_t1, decoder_input_t2, decoder_input_t3], decoder_output)
keras.utils.plot_model(model, to_file="new_model.png", show_shapes=True)


# Compile it...
model.compile(loss = [keras.losses.MSE, keras.losses.MSE, keras.losses.binary_crossentropy],
               optimizer=keras.optimizers.Adam(),
               metrics=['accuracy'])

model_input = {"enc_token_1": t1, "enc_token_2": t2, "query_role_input": x_role_input, "dec_token_1": pre_t1, "dec_token_2": pre_t2, "dec_start/stop": pre_t3}
model_target = {"token_1": post_t1, "token_2": post_t2, "start/stop": post_t3}

# Train it
batch_size = 100
epochs = 1600
history = model.fit(model_input, model_target,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=0)
accuracy = model.evaluate(model_input, model_target)
# use `model.metrics_names` to get indices for accuracy:
print('T1 Accuracy:', accuracy[4]*100.0, '%')
print('T2 Accuracy:', accuracy[5]*100.0, '%')
print('T3 Accuracy:', accuracy[6]*100.0, '%')



# Now remove the Teacher Forcing

# Encoder
encoder_model = keras.Model([encoder_input_t1, encoder_input_t2, encoder_query_input], encoder_states)

# Decoder
decoder_state_input_h = keras.layers.Input(shape=(hidden_size,), name="states_input_h")
decoder_state_input_c = keras.layers.Input(shape=(hidden_size,), name="states_input_c")
# inputs to hidden
decoder_states_input = [decoder_state_input_h, decoder_state_input_c]
decoder_hidden_output, decoder_state_h, decoder_state_c = decoder_hidden(decoder_input,
                                                                         initial_state=decoder_states_input)
decoder_states = [decoder_state_h, decoder_state_c]
# hidden to outputs

decoder_output = [decoder_dense_t1(decoder_hidden_output), decoder_dense_t2(decoder_hidden_output), decoder_dense_t3(decoder_hidden_output)]
decoder_model = keras.Model(
    [decoder_input_t1, decoder_input_t2, decoder_input_t3] + decoder_states_input,
    decoder_output + decoder_states)

keras.utils.plot_model(encoder_model, to_file="new_encoder.png", show_shapes=True)
keras.utils.plot_model(decoder_model, to_file="new_decoder.png", show_shapes=True)


# Prepare testing data
# Prepare input

# Each letter that represents a role will be mapped to the encoding for a
# random word from the corpus.
debug_encoding = []
x_test = []
x_role_input = []
correct_result = [] # used to get accuracy at end
roles = np.loadtxt(sys.argv[3], dtype=object)
for sentence in roles:
    #np.vstack(([encoded_mapping[letter] for letter in sentence], np.zeros((nquery_steps,2,1,50))))
    role_index = np.random.randint(0,high=3)
    x_role_input.append(np.vstack((np.zeros((3,3)), 
                                   np.tile(role_encoding[role_index], (nquery_steps,1)))))
    x_test.append(np.vstack(([encoded_mapping[letter] for letter in sentence],np.zeros((nquery_steps,2,1,50)))))
    correct_result.append([selected_words[sentence[role_index]]])
    #debug_encoding.append(encoded_mapping[sentence[role_index]])
#debug_encoding = np.array(debug_encoding)
x_test = np.array(x_test) # shape (n, 3, 2, 1, 50)
correct_result = np.array(correct_result)
x_role_input = np.array(x_role_input)
t1 = x_test[:,:,0,0,:] # new shape (n,3,50)
t2 = x_test[:,:,1,0,:] # " '' "


outer_result = np.empty((len(x_test),6,28)) # (samples, letters in target word, size of encoding)
for i, sentence in enumerate(x_test):
    context = encoder_model.predict({"enc_token_1": t1[i:i+1], 
                                     "enc_token_2": t2[i:i+1], 
                                     "query_role_input":x_role_input[i:i+1]})
    dec_t1 = np.zeros((1,1,50))
    dec_t2 = np.zeros((1,1,50))
    dec_s_s = pre_t3[0:1,0:1,:]
    inner_result = np.zeros([2,50])
    
    # obtain the result from the inner decoder (one word)
    out1, out2, out3, h, c = decoder_model.predict({"states_input_h": context[0], 
                                     "states_input_c": context[1],
                                     "dec_token_1": dec_t1,
                                     "dec_token_2": dec_t2,
                                     "dec_start/stop": dec_s_s})
    
    # Debugging
    #my_mse(debug_encoding[i], [out1,out2])
    
    inner_result[0,:] = out1
    inner_result[1,:] = out2
    
    # obtain the result from the outer decoder
    context = []
    context.append(inner_result[0:1,:])
    context.append(inner_result[1:2,:])
    token = np.array(encode.onehot("start"))
    token = token.reshape([1, 1, token.shape[0]])
    for letter in range(5+1):
        out, h, c = outer_decoder.predict([token] + context)
        token = np.round(out)
        context = [h,c]
        outer_result[i, letter, :] = token            

# Obtain Accuracy
word_accuracy = 0
letter_accuracy = 0
correct_result = correct_result[:,0,:,:] # Remove extra dimension. Now (nsamples, word_len, encoding_len)
for answer, response in zip(correct_result, outer_result):
    # check target word
    if np.array_equal(answer[:,:], response[:,:]):
        word_accuracy += 1
        letter_accuracy += 6
    #check each letter
    else:
        for letter in range(6):
            if np.array_equal(answer[letter,:], response[letter,:]):
                letter_accuracy += 1
                    
word_accuracy /= float(correct_result.shape[0])
letter_accuracy /= float(correct_result.shape[0] * 6)


print('''   Generalization Accuracy
-----------------------------
word_accuracy: %f
letter_accuracy: %f
'''%((word_accuracy*100), (letter_accuracy*100)))

# Debugging
for i in range(0):
    for letter in range(len(correct_result[0])):
        print(correct_result[i,letter])
        print(outer_result[i,letter])
    print("----------------------")
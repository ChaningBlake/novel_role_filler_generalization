#!/usr/bin/env python3

import keras
import numpy as np
import encode
import sys

# ./test_enc.py word

with open('encoder_len5.json', 'r') as encoder_file, open('decoder_len5.json', 'r') as decoder_file:
    encoder_json = encoder_file.read()
    decoder_json = decoder_file.read()
encoder_model = keras.models.model_from_json(encoder_json)
decoder_model = keras.models.model_from_json(decoder_json)
encoder_model.load_weights("encoder_len5.h5")
decoder_model.load_weights("decoder_len5.h5")

word = np.array([encode.onehot(sys.argv[1])])
context = encoder_model.predict(word)
token = np.array(encode.onehot("start"))
token = token.reshape([1, 1, token.shape[0]])
result = np.zeros([1,6,28])

for x in range(6):
        out,h,c = decoder_model.predict([token]+context)
        token = np.round(out)
        context = [h,c]
        result[0,x,:] = token
print(encode.check(result[0]))

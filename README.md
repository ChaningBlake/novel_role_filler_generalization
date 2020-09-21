# Novel Role Filler Generalization for Recurrent Neural Networks Using Working Memory-Based Indirection
* `/data` - Contains all of the sets of words and outputs from runnin the scripts and models
* `/scripts` - Contains all of the scripts used to generate the output.
### Workflow
There are several bash scripts in the `/scripts` folder that will help you work with the python programs to generate and test data. Here is a summary on what each one does and in what order they should be used.
1. script1
2. script2
3. script3
4. script4

## There are a couple different models:
* `e2e_enc_dec.py` - As a contrast to the indirection model. This model is an enc/dec for the words as well as the roles. It expects the whole sentence
* `e2e_enc_dec_query.py` -
* `indirection_model.py` - What I worked on for a majority of the time. This is the plain indirection model with nothing to encode the words for it.
* `indirection_model_w_enc-dec.py` - This is the normal model plus the word encoder-decoder. This is the complete indirection model. 
* `nested-enc-dec-query.py` -
* `nested-enc-dec.py` -


# PAV Speaker Identifier with Deep Neural Networks

Speaker recognition baseline for PAV subject in ETSETB UPC (Telecom BCN)

This program creates a Multilayer perceptron to classify speaker.

## Installation
You need python3 and pytorch (version > 0.4)

## Training
Execute `python train.py --help` to get the help message.

An example:
```
python train.py --save_path model_h20 --hsize 20 --in_frames=1 \
                --db_path work/mcp \
                --tr_list_file cfg/all.train \
		--va_list_file cfg/all.test 
```


## Test
Execute `python test.py --help` to get the help message.

An example:
```
python test.py --weights_ckpt model_h20/bestval_e19_weights.ckpt \
               --train_cfg model_h20/train.opts --log_file results.txt \
               --db_path work/mcp \
	       --te_list_file cfg/all.test
```


From the result file you can easily compute the number of errors.

## Architecture

- The architecture (number of layers, activation) is hard-coded in `train.py` and `test.py`. It is straight forward to change.
- The input files are binary files (fmatrix) which represent arrays of num_of_frames x num_features_per_frame. The format is two integers (of 4 bytes) with nframes and nfeatures and then num_of_framex * num_features_per_frame float values of 4 bytes.
- The network predict the speaker for each input vector. An input vector is the result of stacking `in_frame` frames. This value can be changed as an option. The classification for the complete test file (utterance) is based on the sum of logprob of each input vector.


## Speaker ID.

The script `make_spk2idx.py` was used to create the _dictionary_
`cfg/spk2idx.json`. This _dictionary_ associates each speaker name (as
`SES0001`) with an integer from `0` to `num_speakers-1`.


You don't need to execute it again, if the same database is used.


## Train - Validation - Test


You should use
- the database named `train` for estimate the weigths (`--train_file` in `train.py`).
- the database named `test` for validation (`--val_file` in `train.py`)
- the database named `final test` for test (`--test_file` in `test.py`)

However, as you don't know the ground truth for the `final test` you need to submit the results to us.
If you want to get a first idea of how the system performs, use the validation data as test.


## Experiments

Some parameters you may want to experiment with:
- Influence of number of unit of hidden layers (performance, training and test time, size of model)
- Influence of number of context frames
- You can edit `train.py` and/or `test.py` and change number of layers,
activation function, optimizer, learning rate, add dropout, etc.


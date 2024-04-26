#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --cuda --ortho --train-data=10000000 --dev-data=100000 --val-every=20000 --log-every=1000 --max-epoch=100
elif [ "$1" = "train-rnn" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --cuda --train-data=10000000 --dev-data=100000 --val-every=20000 --log-every=1000 --max-epoch=100
else
	echo "Invalid Option Selected"
fi
This folder was used to test the uRNN/uGRU on the Penn Treebank dataset.
===

# Generate the data set

Firstly, run:

```bash
sh getdata.sh
```

to obtain the Penn Treebank dataset.

# Word-level language modeling RNN

This example trains a RNN (RNN_TANH, RNN_RELU, GRU, or LSTM) or uRNN (URNN, UGRU) on a language modelling task on the Penn Treebank dataset.
The trained model can then be used by the generate script to generate new text.

```bash
python main.py --cuda --epochs 6        # Train a LSTM with CUDA
python main.py --model UGRU             # Train a UGRU on CPU
python generate.py                      # Generate samples from the trained model.
```

The model uses the `nn.RNN` module (and its sister modules `nn.GRU` and `nn.LSTM`)
which will automatically use the cuDNN backend if run on CUDA with cuDNN installed.
If a uRNN is used, a custom update step will be used to optimize the recurrent weight matrix
over SO(n).

During training, if a keyboard interrupt (Ctrl-C) is received,
training is stopped and the current model is evaluated against the test dataset.

The `main.py` script accepts the following arguments:

```bash
optional arguments:
  -h, --help         show this help message and exit
  --data DATA        location of the data corpus
  --model MODEL      type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, URNN, UGRU)
  --emsize EMSIZE    size of word embeddings
  --nhid NHID        number of hidden units per layer
  --nlayers NLAYERS  number of layers
  --lr LR            initial learning rate
  --clip CLIP        gradient clipping
  --epochs EPOCHS    upper epoch limit
  --batch_size N     batch size
  --bptt BPTT        sequence length
  --dropout DROPOUT  dropout applied to layers (0 = no dropout)
  --seed SEED        random seed
  --cuda             use CUDA
  --log-interval N   report interval
  --save SAVE        path to save the final model
```

# Credits

Heavily based off [PyTorch example language model](https://github.com/pytorch/examples/tree/master/word_language_model), we introduce recurrent architectures with recurrent weight matrices constrained to the space of special orthogonal matrices SO(n), that have theoretical gradient bounds which preclude the vanishing/exploding gradients problem.
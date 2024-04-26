"""
Run file for OrthoRNN Toy Example.
Usage:
    run.py train [options]

Options:
    -h --help               Show this screen.
    --cuda                  If using gpu
    --ortho                 Orthogonal RNN (vs normal RNN)
    --seed=<int>            Seed values for randomness (for reproducibility) [Default: 0]
    --save-path=<str>       Relative path to save model to [Default: /models/saved_model.pt]
    --optimizer=<str>       Optimizer [Default: adam]
    --data-dir=<str>        Folder for data [Default: data]
    --reduced               Load reduced data set

    --batch-size=<int>      Batch-size [Default: 100]
    --train-data=<int>      Number of training samples [Default: 100000]
    --dev-data=<int>        Number of validation samples [Default: 10000]
    --long-val              Long validation (test for generalization) (INVALID)
    --print-matrices        Prints matrices during validation

    --n-char=<int>          Number of characters [Default: 100]
    --embed-size=<in>       Embedding dimensions [Default: 30]
    --hidden-size=<int>     Hidden dimensions [Default: 20]

    --lr=<float>            Learning rate [Default: 0.00005]
    --lr-decay=<float>      lr decay factor [Default: 0.5]

    --max-epoch=<int>       max number of epochs [Default: 200]
    --log-every=<int>       logs every N iterations [Default: 100]
    --val-every=<int>       validates every N iterations against test set [Default: 1000]
    --patience=<int>        lr decay if no best in P validations [Default: 5]
    --decay-limit=<int>     number of times to decay [Default: 5]

    --plot-graphs           plot graphs?

    --weightdrop=<float>    weight drop probability [Default: 0.0]
"""
from utils import *
from charRNN import CharRNN
import torch
import torch.nn as nn
from train import *
from dataset_generator import *
from docopt import docopt

def main():
    args = docopt(__doc__)

    if args['train']:
        # Set random seed for reproducibility.
        seed = int(args['--seed'])
        np.random.seed(seed)
        torch.manual_seed(seed)

        train_set, dev_set, test_set = load_data(reduced=args['--reduced'],datapath=args['--data-dir'])

        train_data= [[],[]]
        train_n = int(args['--train-data'])
        train_data[0] = train_set[0][:train_n]
        train_data[1] = train_set[1][:train_n]

        dev_data = [[], []]
        dev_n = int(args['--dev-data'])
        dev_data[0] = dev_set[0][:dev_n]
        dev_data[1] = dev_set[1][:dev_n]

        test_data = [[], []]
        test_n = int(args['--dev-data'])
        test_data[0] = test_set[0][:test_n]
        test_data[1] = test_set[1][:test_n]

        long_dev_data = None
        longer_dev_data = None

        device=torch.device("cuda:0" if args['--cuda'] else "cpu")
        
        model = CharRNN(int(args['--n-char']), int(args['--embed-size']), int(args['--hidden-size']), device, weightdrop=float(args['--weightdrop']))

        if args['--cuda']:
            model.cuda()

        if args['--optimizer'] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), float(args['--lr']), amsgrad=True)
        if args['--optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), float(args['--lr']))
        
        
        loss_func = nn.CrossEntropyLoss()

        train(model, train_data, dev_data, test_data, optimizer, loss_func, args)

if __name__ == '__main__':
    main()

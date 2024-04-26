import time

import torch
import torch.nn as nn
import os
from charRNN import CharRNN
from utils import weight_step, get_minibatches, exp_antisym, plot_acc, plot_losses
    


def train(model, train_data, dev_data, test_data, optimizer, loss_func, args):
    """ Train the model for single epoch.

    Note: In PyTorch we can signify train versus test and automatically have
    the Dropout Layer applied and removed, accordingly, by specifying
    whether we are training, `model.train()`, or evaluating, `model.eval()`

    @param model (TestModel): initialized TestModel
    @param train_data (): question-answer pairs. Shape: (2, xs + ys), where xs = (line, max_line_length), ys = (line)
    @param dev_data (): question-answer pairs. Shape: (2, xs + ys), where xs = (line, max_line_length), ys = (line)
    @param optimizer (torch.Optimizer): SGD Optimizer
    @param loss_func (nn.CrossEntropyLoss): Cross Entropy Loss Function
    @param args (Dict): options passed into run.py
    """
    

    def model_save(fn):
        with open(fn, 'wb') as f:
            torch.save(model, f)

    save_path = os.getcwd() + args['--save-path']
    epoch = 0
    train_iter = 0
    decays = 0
    current_patience = 0
    val_loss_history = [[],[]]
    train_loss_history = [[],[]]
    acc_history = [[], []]
    
    # Training options
    cuda = args['--cuda']
    max_epoch = int(args['--max-epoch'])
    log_every = int(args['--log-every'])
    val_every = int(args['--val-every'])
    ortho = args['--ortho']
    batch_size = int(args['--batch-size'])
    lr = float(args['--lr'])
    long_val = args['--long-val']
    lr_decay = float(args['--lr-decay'])
    decay_limit = int(args['--decay-limit'])
    patience = int(args['--patience'])

    stored_loss = val_loss = 100000000
    
    device=torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('use device: %s' % device)

    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training%s' % (' with Orthogonality' if ortho else ''))
    while True:
        epoch += 1
        # print(train_iter)
        # print("Size :", len(train_data[0]))
        for train_x, train_y in get_minibatches(train_data, batch_size, shuffle=True):
            train_iter +=1

            optimizer.zero_grad()
            xs = torch.tensor(train_x, device=device)
            ys = torch.tensor(train_y, device=device)

            result = model(xs) 
            loss = loss_func(result, ys)
            loss.backward()
            nn.utils.clip_grad_norm_([model.orthoRNN.rnn.weight_hh_l0],1)
            model.orthoRNN.update_b_grad()
            optimizer.step()
            model.orthoRNN.update_weight_from_b()
            
            ## logging 
            if train_iter % log_every == 0:
                print('epoch %d, iter %d, loss %.2f, speed %.2f iter/sec, time elapsed %.2f sec, lr %.6f' % (epoch, train_iter,loss.item(), log_every / (time.time()-train_time), time.time() - begin_time, optimizer.param_groups[0]['lr']))
                train_loss_history[0].append(train_iter)
                train_loss_history[1].append(loss.item())
                train_time = time.time()


            ## validation
            if train_iter % val_every == 0:
                
                if args['--print-matrices']:
                    print("Hidden grad:")
                    print(model.orthoRNN.rnn.weight_hh_l0.grad.data)
                    print("Hidden weights:")
                    print(model.orthoRNN.rnn.weight_hh_l0.data)
                    if ortho:
                        print("B grad:")
                        print(model.orthoRNN.B.grad.data)
                        print("B weights:")
                        print(model.orthoRNN.B.data)
                    print("W^T W:")
                    print(model.orthoRNN.rnn.weight_hh_l0.data.matmul(model.orthoRNN.rnn.weight_hh_l0.data.t()))
                    print("max entry in hidden grad: ",model.orthoRNN.rnn.weight_hh_l0.grad.data.max())
                print('begin validation')
                was_training = model.training
                model.eval()

                # Default eval
                with torch.no_grad():
                    cum_loss = 0
                    batches = 0
                    correct = 0
                    dev_size = len(dev_data[1])
                    val_iter = 0
                    for train_x, train_y in get_minibatches(dev_data, batch_size):
                        xs = torch.tensor(train_x, device=device)
                        ys = torch.tensor(train_y, device=device)
                        result = model(xs)
                        loss = loss_func(result, ys)
                        cum_loss += loss.item() * len(train_x)
                        predictions = result.argmax(dim=1)
                        a = (predictions == ys).sum()
                        correct += (predictions == ys).sum().data
                        val_iter += 1
                        batches += 1
                    print("validation loss: %.2f, accuracy: %.4f, correct: %d" % (cum_loss / dev_size, float(correct) / dev_size, correct))
                    val_loss = cum_loss / dev_size
                    current_patience += 1
                    val_loss_history[0].append(train_iter)
                    val_loss_history[1].append(float(val_loss))
                    acc_history[0].append(train_iter)
                    acc_history[1].append(float(correct) / dev_size)

                # save best model 
                if val_loss < stored_loss:
                    model_save(save_path)
                    print('Saving model (new best validation)')
                    stored_loss = val_loss
                    current_patience = 0

                # Extended eval
                if long_val:
                    with torch.no_grad():
                        cum_loss = 0
                        batches = 0
                        correct = 0
                        long_dev_size = len(long_dev_data[1])
                        val_iter = 0
                        val_iter = 0
                        for train_x, train_y in get_minibatches(long_dev_data, batch_size):
                            xs = torch.tensor(train_x, device=device)
                            ys = torch.tensor(train_y, device=device)
                            result = model(xs)
                            loss = loss_func(result, ys)
                            cum_loss += loss.item()
                            predictions = result.argmax(dim=1)
                            a = (predictions == ys).sum()
                            correct += (predictions == ys).sum().data
                            val_iter += 1
                            batches += 1
                        print("long validation loss: %.2f, accuracy: %.4f, correct: %d" % (cum_loss / batches, float(correct) / long_dev_size, correct))

                    with torch.no_grad():
                        cum_loss = 0
                        batches = 0
                        correct = 0
                        longer_dev_size = len(longer_dev_data[1])
                        val_iter = 0
                        val_iter = 0
                        for train_x, train_y in get_minibatches(longer_dev_data, batch_size):
                            xs = torch.tensor(train_x, device=device)
                            ys = torch.tensor(train_y, device=device)
                            result = model(xs)
                            loss = loss_func(result, ys)
                            cum_loss += loss.item()
                            predictions = result.argmax(dim=1)
                            a = (predictions == ys).sum()
                            correct += (predictions == ys).sum().data
                            val_iter += 1
                            batches += 1
                        print("longer validation loss: %.2f, accuracy: %.4f, correct: %d" % (cum_loss / batches, float(correct) / longer_dev_size, correct))

                if was_training:
                    model.train()
            
                # lr adapt
                if current_patience > patience:
                    lr = lr * lr_decay
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    decays += 1
                    current_patience = 0

                print("Patience: %d/%d, Decays: %d/%d" % (current_patience, patience, decays, decay_limit))

        if (epoch == max_epoch) or (decays > decay_limit) or (loss.item() < 0.01):
            # run test
            cum_loss = 0

            model.eval()
            dev_size = len(test_data[0])
            xs = torch.tensor(test_data[0], device=device)
            ys = torch.tensor(test_data[1], device=device)
            result = model(xs)
            loss = loss_func(result, ys)
            cum_loss += loss.item() * len(train_x)
            predictions = result.argmax(dim=1)
            a = (predictions == ys).sum()
            correct = (predictions == ys).sum().data
            print("Data file: %s" % (args['--data-dir']))
            print("Test loss: %.2f, accuracy: %.4f, correct: %d" % (cum_loss / dev_size, float(correct) / dev_size, correct))
            if (epoch != max_epoch):
                print("Early exit! Epochs: %d, Iterations: %d" % (epoch, train_iter))
            else:
                print("Max epoch reached")
            print("Total training time: %.2f" % (time.time() - begin_time))
            if args['--plot-graphs']:
                name = 'ortho' if args['--ortho'] else 'rnn'
                plot_losses(val_loss_history, train_loss_history, name + "-" + args['--data-dir'] + "-loss-plot")
                # plot_acc(acc_history, name + "-" + args['--data-dir'] + "-acc-plot")
            return
    
    
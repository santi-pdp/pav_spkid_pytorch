from utils import *
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import timeit


# This is for pytorch 0.4
# For version 0.3, change .item() => .data[0]

# In particular:
#  0.3   pred.eq(y.view_as(pred)).sum().data[0]
#        loss.data[0]

# >0.4   pred.eq(y.view_as(pred)).sum().item()
#        loss.item()



def compute_accuracy(y_, y):
    pred = y_.max(1, keepdim=True)[1] 
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / y_.size(0)

def train_spkid_epoch(dloader, model, opt, 
                      epoch, log_freq):
    # setup train mode
    model.train()
    timings = []
    losses = []
    accs = []
    beg_t = timeit.default_timer()
    for bidx, batch in enumerate(dloader, start=1):
        X, Y = batch
        X = Variable(X)
        Y = Variable(Y)
        # reset any previous gradients in optimizer
        opt.zero_grad()
        # (1) Forward data through neural network
        Y_ = model(X)
        # (2) Compute loss (quantify mistake to correct towards giving good Y)
        # Loss is Negative Log-Likelihood, to reduce probability mismatch
        # between network output distribution and true distribution Y
        loss = F.nll_loss(Y_, Y)
        # (3) Backprop gradients
        loss.backward()
        # (4) Apply update to model parameters with optimizer
        opt.step()
        # Compute accuracy to check its increment during training
        acc = compute_accuracy(Y_, Y)
        end_t = timeit.default_timer()
        timings.append(end_t - beg_t)
        beg_t = timeit.default_timer()

        if bidx % log_freq == 0 or bidx >= len(dloader):
            print('TRAINING: {}/{} (Epoch {}) loss: {:.4f} acc:{:.2f} '
                  'mean_btime: {:.3f} s'.format(bidx, len(dloader), 
                                                epoch, loss.item(),
                                                acc,
                                                np.mean(timings)))
            losses.append(loss.item())
            accs.append(acc)
    return losses, accs

def eval_spkid_epoch(dloader, model, epoch, log_freq):
    # setup eval mode
    model.eval()
    va_losses = []
    va_accs = []
    timings = []
    beg_t = timeit.default_timer()
    for bidx, batch in enumerate(dloader, start=1):
        X, Y = batch
        X = Variable(X, volatile=True, requires_grad=False)
        Y = Variable(Y, volatile=True, requires_grad=False)
        Y_ = model(X)
        loss = F.nll_loss(Y_, Y)
        acc = compute_accuracy(Y_, Y)
        va_losses.append(loss.item())
        va_accs.append(acc)
        end_t = timeit.default_timer()
        timings.append(end_t - beg_t)
        beg_t = timeit.default_timer()
        if bidx % log_freq == 0 or bidx >= len(dloader):
            print('EVAL: {}/{} (Epoch {}) m_loss(so_far): {:.4f} mean_btime: {:.3f}'
                  's'.format(bidx, len(dloader), epoch, np.mean(va_losses),
                             np.mean(timings)))
    m_va_loss = np.mean(va_losses)
    m_va_acc = np.mean(va_accs)
    print('EVAL RESULT Epoch {} >> m_loss: {:.3f} m_acc: {:.2f}'
          ''.format(epoch, m_va_loss, m_va_acc))
    return [m_va_loss], [m_va_acc]

def main(opts):
    dset = SpkDataset(opts.db_path, opts.tr_list_file,
                      opts.ext, opts.spk2idx,
                      in_frames=opts.in_frames)
    dloader = DataLoader(dset, batch_size=opts.batch_size,
                         num_workers=1, shuffle=True, 
                         pin_memory=False)

    va_dset = SpkDataset(opts.db_path, opts.va_list_file,
                         opts.ext, opts.spk2idx,
                         in_frames=opts.in_frames)
    va_dloader = DataLoader(va_dset, batch_size=opts.batch_size,
                            num_workers=1, shuffle=True, 
                            pin_memory=False)
    opts.input_dim = dset.input_dim
    opts.num_spks = dset.num_spks
    # save training config
    with open(os.path.join(opts.save_path, 'train.opts'), 'w') as opts_f:
        opts_f.write(json.dumps(vars(opts), indent=2))
    # Feed Forward Neural Network
    model = nn.Sequential(nn.Linear(dset.input_dim * dset.in_frames, opts.hsize),
                          nn.ReLU(),
                          nn.Linear(opts.hsize, opts.hsize),
                          nn.ReLU(),
                          nn.Linear(opts.hsize, opts.hsize),
                          nn.ReLU(),
                          nn.Linear(opts.hsize, dset.num_spks),
                          nn.LogSoftmax(dim=1))
    print('Created model:')
    print(model)
    print('-')

    #opt = optim.SGD(model.parameters(), lr=opts.lr, momentum=opts.momentum)
    opt = optim.Adam(model.parameters(), lr=opts.lr)
    tr_loss = []
    tr_acc = []
    va_loss = []
    va_acc = []
    best_val = np.inf
    # patience factor to validate data and get out of train earlier
    # of things do not improve in the held out dataset
    patience = opts.patience
    for epoch in range(opts.epoch):
        tr_loss_, tr_acc_ = train_spkid_epoch(dloader, model, 
                                              opt, epoch,
                                              opts.log_freq)
        va_loss_, va_acc_ = eval_spkid_epoch(va_dloader, model, 
                                             epoch, opts.log_freq)
        if best_val <= va_loss_[0]:
            patience -= 1
            print('Val loss did not improve. Patience '
                  '{}/{}.'.format(patience, opts.patience))
            if patience <= 0:
                print('Breaking train loop: Out of patience')
                break
            mname = os.path.join(opts.save_path,
                                 'e{}_weights.ckpt'.format(epoch))
        else:
            # reset patience
            print('Val loss improved {:.3f} -> {:.3f}'.format(best_val,
                                                      va_loss_[0]))
            best_val = va_loss_[0]
            patience = opts.patience
            mname = os.path.join(opts.save_path,
                                 'bestval_e{}_weights.ckpt'.format(epoch))
        # save model
        torch.save(model.state_dict(), mname)
        tr_loss += tr_loss_
        tr_acc += tr_acc_
        va_loss += va_loss_
        va_acc += va_acc_
        stats = {'tr_loss':tr_loss,
                 'tr_acc':tr_acc,
                 'va_loss':va_loss,
                 'va_acc':va_acc}

        with open(os.path.join(opts.save_path,
                               'train_stats.json'), 'w') as stats_f:
            stats_f.write(json.dumps(stats, indent=2))

        # plot training loss/acc and eval loss/acc
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.plot(tr_loss)
        plt.xlabel('Global step')
        plt.ylabel('Train NLL Loss')
        plt.subplot(2, 2, 2)
        plt.plot(tr_acc)
        plt.xlabel('Global step')
        plt.ylabel('Train Accuracy')
        plt.subplot(2, 2, 3)
        plt.plot(va_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Eval NLL Loss')
        plt.subplot(2, 2, 4)
        plt.plot(va_acc)
        plt.xlabel('Epoch')
        plt.ylabel('Eval Accuracy')
        plt.savefig(os.path.join(opts.save_path, 
                                 'log_plots.png'),
                   dpi=200)
        plt.close()
        # save curves for future purpose


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MLP for Speaker Classification')
    parser.add_argument('--db_path', type=str, default='mcp',
                        help='path to feature files (default: ./mcp)')
    parser.add_argument('--tr_list_file', type=str, default='cfg/all.train',
                        help='File list of train files (default: cfg/all.train)')
    parser.add_argument('--va_list_file', type=str, default='cfg/all.test',
                        help='File list of eval files (default: cfg/all.test)')
    parser.add_argument('--ext', type=str, default='mcp', 
                        help='Extension of feature files (default mcp)')
    parser.add_argument('--spk2idx', type=str, default='cfg/spk2idx.json',
                        help='File to map spk code to spkID: 0,1, .... (def. cfg/spk2idx.json)')
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size (default: 1000)')
    parser.add_argument('--hsize', type=int, default=100,
                        help='Num. of units in hidden layers (default=100)')
    parser.add_argument('--in_frames', type=int, default=21,
                        help='num of frames stacked to create the input features (default: 21)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Num of epochs to wait if val loss improves '
                             '(default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (def. 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, help='Momentum (def. 0.5)')
    parser.add_argument('--epoch', type=int, default=20, 
                        help='Number of epochs to train (default: 20)')
    parser.add_argument('--log_freq', type=int, default=20, 
                        help='Every <log_freq> batches, log stuff (default: 20)')
    parser.add_argument('--save_path', type=str, default='ckpt', help='path for the model (def. ckpt)')


    opts = parser.parse_args()
    if not os.path.exists(opts.save_path):
        os.makedirs(opts.save_path)
    print('Parsed options:')
    print(json.dumps(vars(opts), indent=2))
    print('-' * 30)
    main(opts)



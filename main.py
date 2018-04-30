#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Activation
from utils import *
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import timeit


def train_spkid_epoch(dloader, model, opt, 
                epoch, log_freq, writer):
    timings = []
    global_step = epoch * len(dloader)
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
        end_t = timeit.default_timer()
        timings.append(end_t - beg_t)
        beg_t = timeit.default_timer()
        if bidx % log_freq == 0 or bidx >= len(dloader):
            print('TRAINING: {}/{} (Epoch {}) loss: {:.4f} mean_btime: {:.3f}'
                  's'.format(bidx, len(dloader), epoch, loss.data[0],
                             np.mean(timings)))
            writer.add_scalar('train/loss', loss.data[0], global_step)
        global_step += 1


def main(opts):
    dset = SpkDataset(opts.db_path, opts.list_file,
                      opts.ext, opts.spk2idx)
    dloader = DataLoader(dset, batch_size=opts.batch_size,
                         num_workers=1, shuffle=True, 
                         pin_memory=False)

    # Feed Forward Neural Network
    model = nn.Sequential(nn.Linear(dset.input_dim * dset.in_frames, 256),
                          nn.ReLU(),
                          nn.Linear(256, 256),
                          nn.ReLU(),
                          nn.Linear(256, 256),
                          nn.ReLU(),
                          nn.Linear(256, dset.num_spks),
                          nn.LogSoftmax(dim=1))
    print('Created model:')
    print(model)
    print('-')

    #opt = optim.SGD(model.parameters(), lr=opts.lr, momentum=opts.momentum)
    opt = optim.Adam(model.parameters(), lr=opts.lr)
    writer = SummaryWriter(os.path.join(opts.save_path, 'train'))
    for epoch in range(opts.epoch):
        train_spkid_epoch(dloader, model, opt, epoch,
                         opts.log_freq, writer)
    #model = Sequential([
    #    Dense(
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str, default='mcp')
    parser.add_argument('--list_file', type=str, default='cfg/all.train',
                        help='File list of files (Def: cfg/all.train).')
    parser.add_argument('--ext', type=str, default='mcp', 
                        help='Default mcp')
    parser.add_argument('--spk2idx', type=str, default='cfg/spk2idx.json')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--epoch', type=int, default=20, 
                        help='Number of epochs to train (DeF: 20)')
    parser.add_argument('--log_freq', type=int, default=20, 
                       help='Every <log_freq> batches, log stuff')
    parser.add_argument('--save_path', type=str, default='ckpt')


    opts = parser.parse_args()
    if not os.path.exists(opts.save_path):
        os.makedirs(opts.save_path)
    print('Parsed options:')
    print(json.dumps(vars(opts), indent=2))
    print('-' * 30)
    with open(os.path.join(opts.save_path, 'train.opts'), 'w') as opts_f:
        opts_f.write(json.dumps(vars(opts), indent=2))
    main(opts)



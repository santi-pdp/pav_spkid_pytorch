from utils import *
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import os
from torch.autograd import Variable
import timeit


def classify(model, fmatrix, in_frames):
    # fmatrix of size [T, feat_dim]
    # (1) build frames with context first
    frames = build_frames(fmatrix, in_frames)
    # (2) build input tensor to network
    x = Variable(torch.FloatTensor(np.array(frames)))
    # (3) Infer the prediction through network
    y_ = model(x)
    # (4) Sum up the logprobs to obtain final decission
    pred = y_.sum(dim=0)
    class_ = pred.max(dim=0)[1]
    return class_

def main(opts):
    with open(opts.train_cfg, 'r') as cfg_f:
        cfg = json.load(cfg_f)
    with open(cfg['spk2idx'], 'r') as spk2idx_f:
        spk2idx = json.load(spk2idx_f)
        idx2spk = dict((v, k) for k, v in spk2idx.items())
    # Feed Forward Neural Network
    model = nn.Sequential(nn.Linear(cfg['input_dim'] * cfg['in_frames'],
                                    cfg['hsize']),
                          nn.ReLU(),
                          nn.Linear(cfg['hsize'], cfg['hsize']),
                          nn.ReLU(),
                          nn.Linear(cfg['hsize'], cfg['hsize']),
                          nn.ReLU(),
                          nn.Linear(cfg['hsize'], cfg['num_spks']),
                          nn.LogSoftmax(dim=1))
    print('Created model:')
    print(model)
    print('-')
    # load weights
    model.load_state_dict(torch.load(opts.weights_ckpt))
    print('Loaded weights')
    out_log = open(opts.log_file, 'w')
    with open(opts.test_list, 'r') as test_f:
        test_list = [l.rstrip() for l in test_f]
        timings = []
        beg_t = timeit.default_timer()
        for test_i, test_file in enumerate(test_list, start=1):
            test_file = os.path.join(opts.test_db, test_file + '.' + opts.ext)
            fmatrix = read_fmatrix(test_file)
            class_ = classify(model, fmatrix, cfg['in_frames']).data[0]
            end_t = timeit.default_timer()
            timings.append(end_t - beg_t)
            beg_t = timeit.default_timer()
            print('Processing test file {} ({}/{}) with shape: {}'
                  ', mtime: {:.3f} s'.format(test_file, test_i, len(test_list),
                                             fmatrix.shape,
                                             np.mean(timings)))
            out_log.write('{}\t{}\n'.format(test_file, idx2spk[class_]))
    out_log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_ckpt', type=str, default=None)
    parser.add_argument('--log_file', type=str, default='g000_recognition.log')
    parser.add_argument('--train_cfg', type=str, default='ckpt/train.opts')
    parser.add_argument('--test_list', type=str, default='spk_rec.test')
    parser.add_argument('--test_db', type=str, default='sr_test')
    parser.add_argument('--spk2idx', type=str, default='cfg/spk2idx.json')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--ext', type=str, default='mcp')


    opts = parser.parse_args()
    if opts.weights_ckpt is None:
        raise ValueError('Weights ckpt not specified!')
    main(opts)



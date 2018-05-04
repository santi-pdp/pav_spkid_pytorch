import numpy as np
import struct
import json
from torch.utils.data import Dataset
import os


def read_fmatrix(fmatrix_path):
    with open(fmatrix_path, 'rb') as ff:
        rows = struct.unpack('i', ff.read(4))
        cols = struct.unpack('i', ff.read(4))
        # read remaining data in floats
        bstream = ff.read()
        blen = len(bstream)
        data = struct.unpack('{}f'.format(blen // 4),
                             bstream)
        # build np array
        data = np.array(data, dtype=np.float32)

        # as rows and cols are tuples, row+cols is tuple (nrows, ncols)
        data = data.reshape(rows + cols) 
        return data

def build_frames(x, num_frames):
    # x is [T, feat_dim] dimensional, where T depends on wav file length
    # and it is wav_dur / window_shift num of frames T
    frames = []
    # pad the first and last positions of T such that first sample is
    # centered in first and last window: [0,0,..,0,x1, x2,...,xT,0,0,...,0]
    pad_size = num_frames // 2
    pad_T = np.zeros((pad_size, x.shape[1]))
    x_p = np.concatenate((pad_T, x, pad_T), axis=0)
    for beg_i in range(x.shape[0] - num_frames):
        frames.append(x[beg_i:beg_i + num_frames, :].reshape(-1))
    return frames


class SpkDataset(Dataset):

    def __init__(self, db_path, list_file, ext, spk2idx,
                 in_frames=21):
        super().__init__()
        self.in_frames = in_frames
        if list_file is None:
            raise ValueError('List file must be specified to Dataset!')
        with open(spk2idx, 'r') as spk2idx_f:
            spk2idx = json.load(spk2idx_f)
            self.spk2idx = spk2idx
            self.num_spks = len(spk2idx)
            self.X = []
            self.Y = []
            with open(list_file, 'r') as list_f:
                files_list = [l.rstrip() for l in list_f]
                for i, fname in enumerate(files_list, start=1):
                    spkname = fname.split('/')[1]
                    spkidx = spk2idx[spkname]
                    fpath = os.path.join(db_path, 
                                         fname + '.' + ext)
                    print('{}/{} Loading {} file {}'.format(i, len(files_list), 
                                                            ext, fpath))
                    x = read_fmatrix(fpath)
                    if not hasattr(self, 'input_dim'):
                        self.input_dim = x.shape[1]
                    # re-arrange data to construct neighboring frames
                    frames = build_frames(x, in_frames)
                    self.X.append(frames)
                    self.Y.append([spkidx] * len(frames))
                self.X = np.concatenate(self.X, axis=0)
                self.Y = np.concatenate(self.Y, axis=0)
                print('X size: ', self.X.shape)
                print('Y size: ', self.Y.shape[0])

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index, :], self.Y[index]


if __name__ == '__main__':
    print(read_fmatrix('ona.mcp').shape)

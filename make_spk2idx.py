import json
import os


# explore cfg path, excluding 'all' train and test lists
spk_tr_files = [fi.split('.')[0] for fi in os.listdir('cfg') if 'all' not in fi]
spk_tr_files = set(spk_tr_files)

print('Num of spk available: ', len(spk_tr_files))
with open('cfg/spk2idx.json', 'w') as spk_f:
    spk2idx = dict((spk.split('.')[0], k) for k, spk in enumerate(spk_tr_files))
    spk_f.write(json.dumps(spk2idx, indent=2))
    print('Written spk2idx dict in cfg/spk2idx.json')

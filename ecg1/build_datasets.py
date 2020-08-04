#!/usr/bin/env python
# coding: utf-8

# In[14]:


from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import json
import numpy as np
import os
import random
import scipy.io as sio
import tqdm
import pickle 
import argparse
import json
import keras
import time
import sys             #chjoo added on Jul 28-2020
import scipy.io as sio
import network
import load
import util

MAX_EPOCHS = 100

STEP = 256


# In[15]:


def load_ecg_mat(ecg_file):
    return sio.loadmat(ecg_file)['val'].squeeze()


# In[16]:


def load_all(data_path):
    label_file = os.path.join(label_path, "REFERENCE-v3.csv")                #C.H.Joo ---- change data_path to label_path and ../REFERENCE-v3.csv to REFERENCE-v3.csv
#    print("label_file == {}".format(label_file))       #C.H.Joo --- 2020-07-29
    with open(label_file, 'r') as fid:
        records = [l.strip().split(",") for l in fid]

    dataset = []
#    print("records == {}".format(records))              #C.H Joo --- 2020-07-29
    for record, label in tqdm.tqdm(records):
        
        ecg_file = os.path.join(data_path, record + ".mat")
        ecg_file = os.path.abspath(ecg_file)
        ecg = load_ecg_mat(ecg_file)
        num_labels = ecg.shape[0] / STEP
#        print("num_labels: {}, labes: {}".format(num_labels, label))
        dataset.append((ecg_file, [label]*int(num_labels)))          #C.H. Joo --- 2020-07-30 change wrap the num_labels with int due to float error in num_labels
    return dataset


# In[17]:


def split(dataset, dev_frac):
    dev_cut = int(dev_frac * len(dataset))
    random.shuffle(dataset)
    dev = dataset[:dev_cut]
    train = dataset[dev_cut:]
    return train, dev


# In[18]:


def make_json(save_path, dataset):
    with open(save_path, 'w') as fid:
        for d in dataset:
            datum = {'ecg' : d[0],
                     'labels' : d[1]}
            json.dump(datum, fid)
            fid.write('\n')


# In[19]:


if __name__ == "__main__":
    random.seed(2018)

    dev_frac = 0.1
    data_path = "/home/jovyan/ecg/examples/cinc17/data/training2017/"  #c.H.Joo --- 2020-07-30 changed from data/training2017 to /home/~/training2017
    label_path = "/home/jovyan/ecg/examples/cinc17/"      #C.H.Joo --- 2020-07-29
    dataset = load_all(data_path)
    train, dev = split(dataset, dev_frac)
    make_json("train.json", train)
    make_json("dev.json", dev)


# ## util

# In[5]:


def load(dirname):
    preproc_f = os.path.join(dirname, "preproc.bin")
    with open(preproc_f, 'r') as fid:
        preproc = pickle.load(fid)
    return preproc


# In[6]:


def save(preproc, dirname):
    preproc_f = os.path.join(dirname, "preproc.bin")
    with open(preproc_f, 'w') as fid:
        print("preproc_f = {}, fid = {}".format(preproc_f, fid))  #C.H. Joo -- 2020-07-30
        pickle.dump(preproc, fid)


# ## load

# In[7]:


def data_generator(batch_size, preproc, x, y):
    num_examples = len(x)
    examples = zip(x, y)
    examples = sorted(examples, key = lambda x: x[0].shape[0])
    end = num_examples - batch_size + 1
    batches = [examples[i:i+batch_size]
                for i in range(0, end, batch_size)]
    random.shuffle(batches)
    while True:
        for batch in batches:
            x, y = zip(*batch)
            yield preproc.process(x, y)


# In[8]:


class Preproc:

    def __init__(self, ecg, labels):
        self.mean, self.std = compute_mean_std(ecg)
        self.classes = sorted(set(l for label in labels for l in label))
        self.int_to_class = dict( zip(range(len(self.classes)), self.classes))
        self.class_to_int = {c : i for i, c in self.int_to_class.items()}

    def process(self, x, y):
        return self.process_x(x), self.process_y(y)

    def process_x(self, x):
        x = pad(x)
        x = (x - self.mean) / self.std
        x = x[:, :, None]
        return x

    def process_y(self, y):
        # TODO, awni, fix hack pad with noise for cinc
        y = pad([[self.class_to_int[c] for c in s] for s in y], val=3, dtype=np.int32) 
        y = keras.utils.np_utils.to_categorical(
                y, num_classes=len(self.classes))
        return y


# In[9]:


def pad(x, val=0, dtype=np.float32):
    max_len = max(len(i) for i in x)
    padded = np.full((len(x), max_len), val, dtype=dtype)
    for e, i in enumerate(x):
        padded[e, :len(i)] = i
    return padded


# In[10]:


def compute_mean_std(x):
    x = np.hstack(x)
    return (np.mean(x).astype(np.float32),
           np.std(x).astype(np.float32))


# In[11]:


def load_dataset(data_json):
    with open(data_json, 'r') as fid:
        data = [json.loads(l) for l in fid]
    labels = []; ecgs = []
    for d in tqdm.tqdm(data):
        labels.append(d['labels'])
        ecgs.append(load_ecg(d['ecg']))
    return ecgs, labels


# In[12]:


def load_ecg(record):
    if os.path.splitext(record)[1] == ".npy":
        ecg = np.load(record)
    elif os.path.splitext(record)[1] == ".mat":
        ecg = sio.loadmat(record)['val'].squeeze()
    else: # Assumes binary 16 bit integers
        with open(record, 'r') as fid:
            ecg = np.fromfile(fid, dtype=np.int16)

    trunc_samp = STEP * int(len(ecg) / STEP)
    return ecg[:trunc_samp]


# In[13]:


if __name__ == "__main__":
    data_json = "/home/jovyan/ecg/examples/cinc17/train.json"        #C.H. Joo --- 2020-07-30 change: data_json path from examples/cinc17/train.json to /home/~cinc17
    train = load_dataset(data_json)
    preproc = Preproc(*train)
    gen = data_generator(32, preproc, *train)
    for x, y in gen:
        print(x.shape, y.shape)
        break


# In[ ]:





# In[ ]:





# In[102]:


def make_save_dir(dirname, experiment_name):
    start_time = str(int(time.time())) + '-' + str(random.randrange(1000))
    save_dir = os.path.join(dirname, experiment_name, start_time)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


# In[103]:


def get_filename_for_saving(save_dir):
    return os.path.join(save_dir,
            "{val_loss:.3f}-{val_acc:.3f}-{epoch:03d}-{loss:.3f}-{acc:.3f}.hdf5")


# In[104]:


def train(args, params):
    print(sys.argv[1:])
    print("Loading training set...")
    train = load.load_dataset(params['train'])
    print("Loading dev set...")
    dev = load.load_dataset(params['dev'])
    print("Building preprocessor...")
    preproc = load.Preproc(*train)
    print("Training size: " + str(len(train[0])) + " examples.")
    print("Dev size: " + str(len(dev[0])) + " examples.")


    save_dir = make_save_dir(params['save_dir'], args.experiment)

    util.save(preproc, save_dir)

    params.update({
        "input_shape": [None, 1],
        "num_categories": len(preproc.classes)
    })
    
    model = network.build_network(**params)

    stopping = keras.callbacks.EarlyStopping(patience=8)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        factor=0.1,
        patience=2,
        min_lr=params["learning_rate"] * 0.001)

    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath=get_filename_for_saving(save_dir),
        save_best_only=False)

    batch_size = params.get("batch_size", 32)

    if params.get("generator", False):
        train_gen = load.data_generator(batch_size, preproc, *train)
        dev_gen = load.data_generator(batch_size, preproc, *dev)
        model.fit_generator(
            train_gen,
            steps_per_epoch=int(len(train[0]) / batch_size),
            epochs=MAX_EPOCHS,
            validation_data=dev_gen,
            validation_steps=int(len(dev[0]) / batch_size),
            callbacks=[checkpointer, reduce_lr, stopping])
    else:
        train_x, train_y = preproc.process(*train)
        dev_x, dev_y = preproc.process(*dev)
        model.fit(
            train_x, train_y,
            batch_size=batch_size,
            epochs=MAX_EPOCHS,
            validation_data=(dev_x, dev_y),
            callbacks=[checkpointer, reduce_lr, stopping])


# In[105]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    parser.add_argument("--experiment", "-e", help="tag with experiment name",
                        default="default")
    args = parser.parse_args()
    print("args.config_file = {}".format(args.config_file))
    params = json.load(open(args.config_file, 'r'))
    train(args, params)


# In[ ]:





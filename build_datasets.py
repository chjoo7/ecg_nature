#!/usr/bin/env python
# coding: utf-8

# In[82]:


from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import json
import numpy as np
import os
import random
import scipy.io as sio
import tqdm

import argparse
import json
import keras
import time
import sys             #chjoo added on Jul 28-2020

import network
import load
import util

MAX_EPOCHS = 100

STEP = 256


# In[85]:


get_ipython().system('pwd')


# In[77]:


def load_ecg_mat(ecg_file):
    return sio.loadmat(ecg_file)['val'].squeeze()


# In[78]:


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


# In[79]:


def split(dataset, dev_frac):
    dev_cut = int(dev_frac * len(dataset))
    random.shuffle(dataset)
    dev = dataset[:dev_cut]
    train = dataset[dev_cut:]
    return train, dev


# In[80]:


def make_json(save_path, dataset):
    with open(save_path, 'w') as fid:
        for d in dataset:
            datum = {'ecg' : d[0],
                     'labels' : d[1]}
            json.dump(datum, fid)
            fid.write('\n')


# In[81]:


if __name__ == "__main__":
    random.seed(2018)

    dev_frac = 0.1
    data_path = "/home/jovyan/ecg/examples/cinc17/data/training2017/"  #c.H.Joo --- 2020-07-30 changed from data/training2017 to /home/~/training2017
    label_path = "/home/jovyan/ecg/examples/cinc17/"      #C.H.Joo --- 2020-07-29
    dataset = load_all(data_path)
    train, dev = split(dataset, dev_frac)
    make_json("train.json", train)
    make_json("dev.json", dev)


# ## Train

# In[ ]:


def make_save_dir(dirname, experiment_name):
    start_time = str(int(time.time())) + '-' + str(random.randrange(1000))
    save_dir = os.path.join(dirname, experiment_name, start_time)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


# In[ ]:


def get_filename_for_saving(save_dir):
    return os.path.join(save_dir,
            "{val_loss:.3f}-{val_acc:.3f}-{epoch:03d}-{loss:.3f}-{acc:.3f}.hdf5")


# In[ ]:


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


# In[ ]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    parser.add_argument("--experiment", "-e", help="tag with experiment name",
                        default="default")
    args = parser.parse_args()
    print("args.config_file = {}".format(args.config_file))
    params = json.load(open(args.config_file, 'r'))
    train(args, params)


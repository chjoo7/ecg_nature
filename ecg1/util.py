import os
#import cPickle as pickle              #C.H.Joo --- 2020-07-30 
import pickle                          #C.H. Joo --- 2020-07-30 

def load(dirname):
    preproc_f = os.path.join(dirname, "preproc.bin")
    with open(preproc_f, 'r') as fid:
        preproc = pickle.load(fid)
    return preproc

def save(preproc, dirname):
    preproc_f = os.path.join(dirname, "preproc.bin")
    with open(preproc_f, 'w') as fid:
        print("preproc_f = {}, fid = {}".format(preproc_f, fid))  #C.H. Joo -- 2020-07-30
        pickle.dump(preproc, fid)

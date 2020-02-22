# intended to be run from Google Colab
# you obviously don't have the training data available so this
# code will error out at some point.
!pip install biopython
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from Bio import SeqIO
import gzip
import math
from google.colab import drive
from os import listdir
from os.path import join, exists
import re
import random
import h5py
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
%tensorflow_version 2.x
import tensorflow.python.keras
from tensorflow.python.keras.models import load_model, Model
from tensorflow.python.keras.layers import Dense, Dropout, Input, Conv1D, GlobalMaxPooling1D, BatchNormalization, MaxPooling1D
from tensorflow.python.keras.layers.merge import Average
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.optimizers import Adam

# mount files from drive
if not exists('/content/drive'):
    drive.mount('/content/drive')
# set seed to be the same
random.seed(0)

# functions for working with genomic sequence data
def _open(filename, mode='rt'):
    """
    helper function for opening files that allows for gzipped
    """
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

def one_hot_arr(arr):
    """
    returns a one hot encoding of an array of dna strings
    """
    enc = OneHotEncoder(categories=[['A','C','G','T']], sparse=False)
    return(np.array([one_hot_string(s) for s in arr]))

def one_hot_string(s, enc=None):
    """
    returns a one hot encoding of a dna string, s
    given an encoded, enc
    """
    assert(type(s) in [str, np.str_])
    if enc is None:
        enc = OneHotEncoder(categories=[['A','C','G','T']], sparse=False)
    # transform s to the correct shape
    s_arr = np.array(list(s)).reshape(-1,1)
    s_enc = enc.fit_transform(s_arr)
    return(s_enc)

def load_fasta(f):
    """
    returns an array of strings from loading a fasta file in f
    file can be gzipped
    For now, removes any non-ATCG characers from the sequence
    """
    seq_arr = []
    fasta = _open(f)
    for record in SeqIO.parse(fasta, 'fasta'):
        seq_arr.append(re.sub('[^ATCG]','',str(record.seq).upper()))
    return(np.array(seq_arr))

def shred_fasta(seq_arr, w, keep_frac=1.0):
    """
    shreds a fasta sequence into non-overlapping windows of size w
    seq_arr: aray of strings representing a fasta file
    optionally keep only a certain fraction of them to downsample bac
    w: window size 
    keep_frac: float, keep this proportion of windows 
    """
    shred_arr = []
    for seq in seq_arr:
        max_w = math.floor(len(seq)/ w)
        for j in range(max_w):
            seq_w=seq[(j*w):((j+1)*w)]
            shred_arr.append(seq_w)
    shred_arr = np.array(shred_arr)
    # subsample the windows
    if (keep_frac < 1.0):
        keep_mask = np.random.rand(shred_arr.shape[0]) < keep_frac
        shred_arr = shred_arr[keep_mask]
    return(shred_arr)

def rev_comp_one(arr):
    """ 
    Returns the reverse complement of a one-hot encoded dna sequence
    """
    assert(type(arr)==np.ndarray)
    assert(len(arr.shape)==2)
    assert(arr.shape[1]==4)
    arr_rev = arr[::-1,:]
    # switch cols: A-T, C-G (0-3, 2-1)
    arr_rev_comp = arr_rev[:,(3,2,1,0)]
    return(arr_rev_comp)

def rev_comp_many(many_arr):
    """
    wrapper for rev_comp_one on many sequence arrays
    many_arr: array of many sequences. shape=(seqs, 4, window_size)
    returns reverse complement of each sequence
    """
    # print(type(many_arr))
    assert(type(many_arr)==np.ndarray)
    assert(len(many_arr.shape) ==3)
    return(np.array([rev_comp_one(x) for x in many_arr]))

def load_many_fasta(f_list, w=500, keep_frac=1.0):
    """
    load and encode each fasta file in f_list
    Shreds into non-overlapping windows of size w
    optionally keep only a certain fraction of windows
    returns: forw_encoded, the one-hot encoded, shredded version of the sequences 
    """
    i = 1
    forw_encoded = np.ndarray((0, w, 4))
    for f in f_list:
        print("loading file ...  " + str(i) + ' of ' + str(len(f_list)))
        this_encoded = one_hot_arr(shred_fasta(load_fasta(f), w, keep_frac))
        forw_encoded = np.append(forw_encoded ,this_encoded, axis=0)
        i+=1
    # forw_encoded = np.array(forw_encoded)
    # forw_encoded = forw_encoded.reshape(-1, 4, w)
    # rev_encoded = rev_comp_many(forw_encoded)
    return(forw_encoded)

# params for loading data
n_train_phage = 1000
n_train_bact = 250
n_dev = 100
# phage genomes look like they're on average 2-3% as long as bacterial
# keep 10% and subsample down from that later
keep_bact_frac = 0.10
w = 1000
# load some training data directly from fasta files the first time
# or get it from the saved hdf5 file in subsequent examples
load_new = False
clear_mem = False
data_file= "/content/drive/My Drive/cs230_metagenomics/BugNet/bas_test/data_w" + str(w) + ".hdf5"
if load_new:
    print('Loading phage training data')
    with h5py.File(data_file, "w") as f:
        # save a hdf5 dataset for each pos/neg tain/dev
        train_phage_forw  = load_many_fasta(train_phage_files[0:n_train_phage], w=w, keep_frac=1.0)
        f.create_dataset('train_phage_forw', data=train_phage_forw, compression="gzip")
        if clear_mem:
            del train_phage_forw
        print('Loading bacteria training data')
        train_bact_forw  = load_many_fasta(train_bact_files[0:n_train_bact], w=w, keep_frac=keep_bact_frac)
        f.create_dataset('train_bact_forw', data=train_bact_forw, compression="gzip")
        if clear_mem:
            del train_bact_forw
        # dev set
        print('Loading phage dev data')
        dev_phage_forw  = load_many_fasta(dev_phage_files[0:n_dev], w=w, keep_frac=1.0)
        f.create_dataset('dev_phage_forw', data=dev_phage_forw, compression="gzip")
        if clear_mem:
            del dev_phage_forw
        print('Loading bacteria dev data')
        dev_bact_forw  = load_many_fasta(dev_bact_files[0:n_dev], w=w, keep_frac=keep_bact_frac)
        f.create_dataset('dev_bact_forw', data=dev_bact_forw, compression="gzip")
        if clear_mem:
            del dev_bact_forw
    # stop here
    if clear_mem:
        sys.exit()

# load the dataset from the hdf5 file
else:
    print('loading data')
    with  h5py.File(data_file, "r") as f:
        print(' ... training phage')
        train_phage_forw = np.array(f.get('train_phage_forw'))
        print(' ... training bact')
        train_bact_forw = np.array(f.get('train_bact_forw'))
        print(' ... dev phage')
        dev_phage_forw = np.array(f.get('dev_phage_forw'))
        print(' ... dev bact')
        dev_bact_forw = np.array(f.get('dev_bact_forw'))
    print('finished loading')

# reverse complemet each set 
print('making reverse complements')
train_phage_rev = rev_comp_many(train_phage_forw)
train_bact_rev = rev_comp_many(train_bact_forw)
dev_phage_rev = rev_comp_many(dev_phage_forw)
dev_bact_rev = rev_comp_many(dev_bact_forw)

# data is going to be imbalanced toward bacteria. correct for that here
# by subsampling
train_imbalance = train_phage_forw.shape[0] / train_bact_forw.shape[0]
dev_imbalance = dev_phage_forw.shape[0] / dev_bact_forw.shape[0]
if train_imbalance < 0.99:
    keep_mask = np.random.rand(train_bact_forw.shape[0]) < train_imbalance
    train_bact_forw = train_bact_forw[keep_mask]
    train_bact_rev = rev_comp_many(train_bact_forw)
if dev_imbalance < 0.99:
    keep_mask = np.random.rand(dev_bact_forw.shape[0]) < dev_imbalance
    dev_bact_forw = dev_bact_forw[keep_mask]
    dev_bact_rev = rev_comp_many(dev_bact_forw)

# imbalance should be fixed now
print(train_phage_forw.shape[0] / train_bact_forw.shape[0])
print(dev_phage_forw.shape[0] / dev_bact_forw.shape[0])

# set up X and Y vectors
# number of training and dev examples
n_train_pos = train_phage_forw.shape[0]
n_train_neg = train_bact_forw.shape[0]
n_train = n_train_pos + n_train_neg

n_dev_pos = dev_phage_forw.shape[0]
n_dev_neg = dev_bact_forw.shape[0]
n_dev = n_dev_pos + n_dev_neg

Y_train = np.append(np.ones(n_train_pos), np.zeros(n_train_neg), axis=0)
Y_dev = np.append(np.ones(n_dev_pos), np.zeros(n_dev_neg), axis=0)

X_train_forw = np.append(train_phage_forw, train_bact_forw, axis=0)
X_train_rev = np.append(train_phage_rev, train_bact_rev, axis=0)

X_dev_forw = np.append(dev_phage_forw, dev_bact_forw, axis=0)
X_dev_rev = np.append(dev_phage_rev, dev_bact_rev, axis=0)

### set up the model ###
# parameters for the model
dropout_pool = 0.1
dropout_dense = 0.1
channel_num  = 4
filter_len1=10
nb_filter1 = 1000
nb_dense = 1000
batch_size=32
train_epochs=20

# helper for creating model
def get_output(input_layer, hidden_layers):
    output = input_layer
    for hidden_layer in hidden_layers:
        output = hidden_layer(output)
    return output

# two inputs - forward and reverse complement
forward_input = Input(shape=(w, channel_num))
reverse_input = Input(shape=(w, channel_num))
# hidden layer - a single conv block
hidden_layers = [
    Conv1D(filters = nb_filter1, kernel_size = filter_len1, activation='relu'),
    GlobalMaxPooling1D(),
    Dropout(dropout_pool),
    Dense(nb_dense, activation='relu'),
    Dropout(dropout_dense),
    Dense(1, activation='sigmoid')
]
# two outputs
forward_output = get_output(forward_input, hidden_layers)     
reverse_output = get_output(reverse_input, hidden_layers)
# average forward and reverse complement
output = Average()([forward_output, reverse_output])
model = Model(inputs=[forward_input, reverse_input], outputs=output)
# print(model.summary())

# save checkpoints to here
model_dir = '/content/drive/My Drive/cs230_metagenomics/BugNet/bas_test/saved_models'
checkpointer = ModelCheckpoint(filepath=join(model_dir, 'dvf_w' + str(w) +'-{epoch:02d}-{val_accuracy:.2f}.hdf5'),
                               verbose=1,save_best_only=True, 
                               monitor='val_accuracy')
earlystopper = EarlyStopping(monitor='accuracy', min_delta=0.0001, patience=5, verbose=1)
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

train = True
if train:
    history = model.fit(x=[X_train_forw, X_train_rev], y=Y_train, 
        batch_size=batch_size, epochs=train_epochs, verbose=2,
        validation_data=([X_dev_forw, X_dev_rev], Y_dev),
        callbacks=[checkpointer, earlystopper])
else: 
    # load in predetermined weights
    listdir(model_dir)
    model.load_weights(join(model_dir, "weights-11-0.88.hdf5"))
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

# Make plots on model performance
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'dev'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'dev'], loc='upper left')
plt.show()

# confusion matrix
Y_dev_pred = model.predict([X_dev_forw, X_dev_rev])
matrix = confusion_matrix(Y_dev, Y_dev_pred > 0.5)
matrix_norm = matrix / np.sum(matrix)

print(matrix)
print(matrix_norm)

labels = ['bacteria', 'virus']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(matrix_norm, cmap ='Reds')
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

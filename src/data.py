import numpy as np
import torch
import pdb

def get_nli_hypoth(nlipath, max_train_sents, max_val_sents, max_test_sents):
  labels = {}
  hypoths = {}

  labels_to_int = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

  for data_split in ['train', 'val', 'test']:
    labels[data_split], hypoths[data_split] = [], []

    lbls = open("%s/cl_snli_%s_lbl_file" % (nlipath, data_split)).readlines()
    srcs = open("%s/cl_snli_%s_source_file" % (nlipath, data_split)).readlines()

    assert (len(lbls) == len(srcs), "SNIL %s labels and source files are not same length" % (data_split))

    for i in range(len(lbls)):
      lbl = lbls[i].strip()
      hypoth = srcs[i].split("|||")[-1].strip()

      if lbl not in labels_to_int:
        print "bad label: %s" % (lbl)
        continue

      labels[data_split].append(labels_to_int[lbl])
      hypoths[data_split].append(hypoth)


  train = {'lbls': labels['train'][0:max_train_sents], 'hypoths' : hypoths['train'][0:max_train_sents]}
  val = {'lbls': labels['val'][0:max_val_sents], 'hypoths' : hypoths['val'][0:max_val_sents]}
  test = {'lbls': labels['test'][0:max_test_sents], 'hypoths' : hypoths['test'][0:max_test_sents]}

  return train, val, test

def get_vocab(txt):
  vocab = set()
  for sent in txt:
    for word in sent.split():
      vocab.add(word)
  return vocab

def get_word_vecs(vocab, embdsfile):
  word_vecs = {}
  with open(embdsfile) as f:
    for line in f:
      word, vec = line.split(' ', 1)
      if word in vocab:
        word_vecs[word] = np.array(list(map(float, vec.split())))
  print('Found {0}(/{1}) words with vectors'.format(
            len(word_vecs), len(vocab)))
  return word_vecs



def build_vocab(txt, embdsfile):
  vocab = get_vocab(txt)
  word_vecs = get_word_vecs(vocab, embdsfile)
  print('Vocab size : {0}'.format(len(word_vecs)))
  return word_vecs 

def get_batch(batch, word_vec):
  # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
  lengths = np.array([len(x) for x in batch])
  max_len = np.max(lengths)
  embed = np.zeros((max_len, len(batch), len(word_vec[word_vec.keys()[0]])))

  for i in range(len(batch)):
    sent = batch[i].split()
    for j in range(len(batch[i].split())):
      embed[j, i, :] = word_vec[sent[j]]

  return torch.from_numpy(embed).float(), lengths

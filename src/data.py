import numpy as np
import torch
import pdb

def extract_from_file(lbls_file, srcs_file, max_sents, data_split):
  labels_to_int = None
  if "mpe" in lbls_file or "snli" in lbls_file or "multinli" in lbls_file or "sick" in lbls_file:
    labels_to_int = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
  elif "spr" in lbls_file or "dpr" in lbls_file or "fnplus" in lbls_file or "add_one" in lbls_file:
    labels_to_int = {'entailed': 0, 'not-entailed': 1}
  elif "scitail" in lbls_file:
    labels_to_int = {'entailed': 0, 'neutral': 1}
  else:
    print "Invalid lbls_file: %s" % (lbls_file)

  data = {'lbls': [], 'hypoths': []}

  lbls = open(lbls_file).readlines()
  srcs = open(srcs_file).readlines()
  assert (len(lbls) == len(srcs), "%s: %s labels and source files are not same length" % (lbls_file, data_split))

  added_sents = 0
  for i in range(len(lbls)):
    lbl = lbls[i].strip()
    hypoth = srcs[i].split("|||")[-1].strip()

    if lbl not in labels_to_int:
      print "bad label: %s" % (lbl)
      continue

    if added_sents >= max_sents:
      continue

    data['lbls'].append(labels_to_int[lbl])
    data['hypoths'].append(hypoth)  
    added_sents += 1

  return data

def get_nli_hypoth(train_lbls_file, train_src_file, val_lbls_file, val_src_file, \
                   test_lbls_file, test_src_file, max_train_sents, max_val_sents, max_test_sents):
  labels = {}
  hypoths = {}

  train = extract_from_file(train_lbls_file, train_src_file, max_train_sents, "train")
  val = extract_from_file(val_lbls_file, val_src_file, max_val_sents, "val")
  test = extract_from_file(test_lbls_file, test_src_file, max_test_sents, "test")

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
  vocab.add("OOV")
  word_vecs = get_word_vecs(vocab, embdsfile)
  print('Vocab size : {0}'.format(len(word_vecs)))
  return word_vecs 

def get_batch(batch, word_vec):
  # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
  lengths = np.array([len(x.split()) for x in batch])
  max_len = np.max(lengths)
  embed = np.zeros((max_len, len(batch), len(word_vec[word_vec.keys()[0]])))

  for i in range(len(batch)):
    sent = batch[i].split()
    for j in range(len(batch[i].split())):
      if sent[j] not in word_vec:
        embed[j, i, :] = word_vec["OOV"]
      else:
        embed[j, i, :] = word_vec[sent[j]]

  return torch.from_numpy(embed).float(), lengths

import os
import sys
import time
import argparse
import pdb

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn

from data import get_nli_hypoth
from data import build_vocab
from models import NLINet

def get_args():
  parser = argparse.ArgumentParser(description='Training NLI model based on just hypothesis sentence')

  # paths
  parser.add_argument("--embdfile", type=str, default='../data/embds/glove.840B.300d.txt', help="File containin the word embeddings")
  parser.add_argument("--nlipath", type=str, default='../data/snli_1.0/', help="NLI data path (SNLI or MultiNLI)")
  parser.add_argument("--outputdir", type=str, default='savedir/', help="Output directory")
  parser.add_argument("--outputmodelname", type=str, default='model.pickle')

  # data
  parser.add_argument("--max_train_sents", type=int, default=10000000, help="Maximum number of training examples")
  parser.add_argument("--max_val_sents", type=int, default=10000000, help="Maximum number of validation/dev examples")
  parser.add_argument("--max_test_sents", type=int, default=10000000, help="Maximum number of test examples")

  # training
  parser.add_argument("--n_epochs", type=int, default=20)
  parser.add_argument("--batch_size", type=int, default=64)
  parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
  parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
  parser.add_argument("--nonlinear_fc", type=float, default=0, help="use nonlinearity in fc")
  parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1")
  parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
  parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
  parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
  parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")

  # model
  parser.add_argument("--encoder_type", type=str, default='BLSTMEncoder', help="see list of encoders")
  parser.add_argument("--enc_lstm_dim", type=int, default=2048, help="encoder nhid dimension")
  parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
  parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
  parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")
  parser.add_argument("--pool_type", type=str, default='max', help="max or mean")

  # gpu
  parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
  parser.add_argument("--seed", type=int, default=1234, help="seed")

  params, _ = parser.parse_known_args()

  # print parameters passed, and all parameters
  print('\ntogrep : {0}\n'.format(sys.argv[1:]))
  print(params)

  return params

def get_model_configs(params, n_words):
  """
  MODEL
  """
  # model config
  config_nli_model = {
    'n_words'        :  n_words               ,
    'word_emb_dim'   :  params.word_emb_dim   ,
    'enc_lstm_dim'   :  params.enc_lstm_dim   ,
    'n_enc_layers'   :  params.n_enc_layers   ,
    'dpout_model'    :  params.dpout_model    ,
    'dpout_fc'       :  params.dpout_fc       ,
    'fc_dim'         :  params.fc_dim         ,
    'bsize'          :  params.batch_size     ,
    'n_classes'      :  params.n_classes      ,
    'pool_type'      :  params.pool_type      ,
    'nonlinear_fc'   :  params.nonlinear_fc   ,
    'encoder_type'   :  params.encoder_type   ,
    'use_cuda'       :  params.gpu_id > 0     ,
  }

  # model
  encoder_types = ['BLSTMEncoder']
                 #, 'BLSTMprojEncoder', 'BGRUlastEncoder',
                 #'InnerAttentionMILAEncoder', 'InnerAttentionYANGEncoder',
                 #'InnerAttentionNAACLEncoder', 'ConvNetEncoder', 'LSTMEncoder']
  assert params.encoder_type in encoder_types, "encoder_type must be in " + \
                                             str(encoder_types)

  return config_nli_model

def main(args):
  print "main"

  """
  SEED
  """
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)

  """
  DATA
  """
  train, val, test = get_nli_hypoth(args.nlipath, args.max_train_sents, \
                                    args.max_val_sents, args.max_test_sents)

  word_vecs = build_vocab(train['hypoths'] + val['hypoths'] + test['hypoths'] , args.embdfile)
  args.word_emb_dim = len(word_vecs[word_vecs.keys()[0]])

  nli_model_configs = get_model_configs(args, len(word_vecs))

  nli_net = NLINet(nli_model_configs)
  print(nli_net)

if __name__ == '__main__':
  args = get_args()
  main(args)

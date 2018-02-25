import os
import sys
import time
import argparse
import pdb

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn

from data import get_nli_hypoth, build_vocab, get_batch
from models import NLI_HYPOTHS_Net
from mutils import get_optimizer

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
  parser.add_argument("--gpu_id", type=int, default=-1, help="GPU ID")
  parser.add_argument("--seed", type=int, default=1234, help="seed")


  #misc
  parser.add_argument("--verbose", type=int, default=1, help="Verbose output")

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
    'use_cuda'       :  params.gpu_id > -1     ,
    'verbose'        :  params.verbose > 0    ,
  }

  # model
  encoder_types = ['BLSTMEncoder']
                 #, 'BLSTMprojEncoder', 'BGRUlastEncoder',
                 #'InnerAttentionMILAEncoder', 'InnerAttentionYANGEncoder',
                 #'InnerAttentionNAACLEncoder', 'ConvNetEncoder', 'LSTMEncoder']
  assert params.encoder_type in encoder_types, "encoder_type must be in " + \
                                             str(encoder_types)
  return config_nli_model

def trainepoch(epoch, train, optimizer, params, word_vec, nli_net, loss_fn):
  print('\nTRAINING : Epoch ' + str(epoch))
  nli_net.train()
  all_costs = []
  logs = []
  words_count = 0

  last_time = time.time()
  correct = 0.
  # shuffle the data
  permutation = np.random.permutation(len(train['hypoths']))

  hypoths, target = [], [] 
  for i in permutation:
    hypoths.append(train['hypoths'][i])
    target.append(train['lbls'][i])

  optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * params.decay if epoch>1\
      and 'sgd' in params.optimizer else optimizer.param_groups[0]['lr']
  print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))

  trained_sents = 0

  start_time = time.time()
  for stidx in range(0, len(hypoths), params.batch_size):
    # prepare batch
    hypoths_batch, hypoths_len = get_batch(hypoths[stidx:stidx + params.batch_size], word_vec)
    tgt_batch = None
    if params.gpu_id > -1: 
      hypoths_batch = Variable(hypoths_batch.cuda())
      tgt_batch = Variable(torch.LongTensor(target[stidx:stidx + params.batch_size])).cuda()
    else:
      hypoths_batch = Variable(hypoths_batch)
      tgt_batch = Variable(torch.LongTensor(target[stidx:stidx + params.batch_size]))

    k = hypoths_batch.size(1)  # actual batch size

    # model forward
    output = nli_net((hypoths_batch, hypoths_len))

    pred = output.data.max(1)[1]
    correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()
    assert len(pred) == len(hypoths[stidx:stidx + params.batch_size])

    # loss
    loss = loss_fn(output, tgt_batch)
    all_costs.append(loss.data[0])
    words_count += hypoths_batch.nelement() / params.word_emb_dim

    # backward
    optimizer.zero_grad()
    loss.backward()

    # gradient clipping (off by default)
    shrink_factor = 1
    total_norm = 0

    for p in nli_net.parameters():
      if p.requires_grad:
        p.grad.data.div_(k)  # divide by the actual batch size
        total_norm += p.grad.data.norm() ** 2
    total_norm = np.sqrt(total_norm)

    if total_norm > params.max_norm:
        shrink_factor = params.max_norm / total_norm
    current_lr = optimizer.param_groups[0]['lr'] # current lr (no external "lr", for adam)
    optimizer.param_groups[0]['lr'] = current_lr * shrink_factor # just for update

    # optimizer step
    optimizer.step()
    optimizer.param_groups[0]['lr'] = current_lr

    if len(all_costs) == 100:
      logs.append('{0} ; loss {1} ; sentence/s {2} ; words/s {3} ; accuracy train : {4}'.format(
                            stidx, round(np.mean(all_costs), 2),
                            int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                            int(words_count * 1.0 / (time.time() - last_time)),
                            round(100.*correct/(stidx+k), 2)))
      print(logs[-1])
      last_time = time.time()
      words_count = 0
      all_costs = []

    if params.verbose:
      trained_sents += k
      print "epoch: %d -- trained %d / %d sentences -- %f ms per sentence" % (epoch, trained_sents, len(hypoths),
                                                                            1000 * (time.time() - start_time) / trained_sents) 
      #sys.stdout.flush()

  train_acc = round(100 * correct/len(hypoths), 2)
  print('results : epoch {0} ; mean accuracy train : {1}'
          .format(epoch, train_acc))
  return train_acc, nli_net


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

  nli_net = NLI_HYPOTHS_Net(nli_model_configs)
  print(nli_net)

  # loss
  weight = torch.FloatTensor(args.n_classes).fill_(1)
  loss_fn = nn.CrossEntropyLoss(weight=weight)
  loss_fn.size_average = False

  # optimizer
  optim_fn, optim_params = get_optimizer(args.optimizer)
  optimizer = optim_fn(nli_net.parameters(), **optim_params)

  if args.gpu_id > -1:
    nli_net.cuda()
    loss_fn.cuda()

  """
  TRAIN
  """
  val_acc_best = -1e10
  adam_stop = False
  stop_training = False
  lr = optim_params['lr'] if 'sgd' in args.optimizer else None

  """
  Train model on Natural Language Inference task
  """
  epoch = 1

  while not stop_training and epoch <= args.n_epochs:
    train_acc, nli_net = trainepoch(epoch, train, optimizer, args, word_vecs, nli_net, loss_fn)
    #eval_acc = evaluate(epoch, 'valid')
    epoch += 1


if __name__ == '__main__':
  args = get_args()
  main(args)

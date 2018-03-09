import pdb
import argparse
import pandas as pd
from collections import defaultdict
import numpy as np

def get_args():
  parser = argparse.ArgumentParser(description='Evaluate NLI based on prediction files.')
  parser.add_argument('--gold_lbl', type=str, help='The gold train labels file')
  parser.add_argument('--pred_lbl', type=str, help='The predicted labels file')
  parser.add_argument('--hyp_src', type=str, help='File containing the hypothesis sentences')
  #/export/b02/apoliak/nli-hypothes-only/targeted-nli/cl_sprl_
  parser.add_argument('--data_split', type=str, help='Which split of datset is being used')
  parser.add_argument('--data_src', type=str, help='Which dataset is being used')

  parser.add_argument('--preds', type=int, help="Determine whether tosplit based on predictions or not")
  parser.add_argument('--vocab_thresh', type=int, default=25, help="Only include words that appear this many times")
  parser.add_argument('--percent_keep', type=int, default=40, help="Only keep words high percentages higher than this")
  parser.add_argument('--top_k', type=int, default=500, help="Top k examples to keep. k=100")

  args = parser.parse_args()
  return args

def get_sents(gold_f, pred_f, hyp_f, data_split):
  gold_lbls = open(gold_f).readlines()
  pred_lbls = open(pred_f).readlines()
  hyp_srcs = open(hyp_f).readlines()

  assert (len(gold_lbls) == len(pred_lbls) == len(hyp_srcs))

  sents = {"correct": {}, "wrong": {}}

  correct = {'total': 0}
  tot_lbl = {'total': 0}

  for i in range(len(gold_lbls)):
    gold_lbl = gold_lbls[i].strip()
    if gold_lbl not in tot_lbl:
      tot_lbl[gold_lbl] = 0
    tot_lbl[gold_lbl] += 1
    tot_lbl['total'] += 1

    if gold_lbl == pred_lbls[i].strip():
      correct['total'] += 1
      if gold_lbl not in correct:
        correct[gold_lbl] = 0
      correct[gold_lbl] += 1

      if gold_lbl not in sents["correct"]:
        sents["correct"][gold_lbl] = []
      sents["correct"][gold_lbl].append(hyp_srcs[i].split("|||")[-1].strip())

    else:
      if gold_lbl not in sents["wrong"]:
        sents["wrong"][gold_lbl] = []
      sents["wrong"][gold_lbl].append(hyp_srcs[i].split("|||")[-1].strip())

  #assert sum([sum(len(_) for _ in i.values()) for i in sents.values()   ]) == 1000 #pdb.set_trace()
  return sents

def get_plot(data, threshold):
  vocab = defaultdict(int)
  all_sents = {}
  vocab_for_lbl = {}
  #defaultdict(int)
  for key in data: #wrong, correct
    for lbl in data[key]: #label
      for sent in data[key][lbl]:
        for word in sent.split():
          word=word.replace(".", "").replace("?", "")
          vocab[word] += 1
          if lbl not in vocab_for_lbl:
            vocab_for_lbl[lbl] = defaultdict(int)
          vocab_for_lbl[lbl][word] += 1
  #assert sum([len(_) for _ in data['wrong'].values()]) +  sum([len(_) for _ in data['correct'].values()]) == 1000

  tmp_count = 0
  for key in data:
    for lbl in data[key]:
      if lbl not in all_sents:
        all_sents[lbl] = defaultdict(list)
      print len(data[key][lbl])
      for sent in data[key][lbl]:
        for word in sent.split():
          word=word.replace(".", "").replace("?", "")
          if sent in all_sents[lbl]: #and key == 'contradiction':
            #print "blah %s" % (key)
            tmp_count += 1
            #print tmp_count
          all_sents[lbl][sent].append(vocab_for_lbl[lbl][word] / float(vocab[word]))

  #Because there are duplicate hypothesis, the total number of sentences across all three label's graphs
  #will not be equal to the number of total sentences in the development set 

  df = pd.DataFrame(columns=['gold-lbl', 'percent', 'num_sents'])
  for i in np.arange(0.0, 1.0, 0.01):
    for lbl in all_sents:
      num_sents = sum([1 for sent in all_sents[lbl] if max(all_sents[lbl][sent]) >= i])
      df = df.append({'gold-lbl': lbl, 'percent': i, 'num_sents': num_sents}, ignore_index=True)

  pdb.set_trace()
  return df


def get_vocab_counts(data, use_preds, threshold, threshold_percent):
  all_vocab = {}
  all_toks_count = 0
  word_to_sents = {}
  threshold_percent = threshold_percent / 100.0
  print threshold_percent
  vocab_counts = {'correct': {}, 'wrong': {}}
  lbl_counts = {}
  for key in data:
    for lbl in data[key]:
      if lbl not in vocab_counts[key]:
        vocab_counts[key][lbl] = {}
        lbl_counts[lbl] = 0
        word_to_sents[lbl] = {}
      lbl_counts[lbl] += 1
      for sent in data[key][lbl]:
        for word in sent.split():
          word=word.replace(".", "").replace("?", "")
          if word not in vocab_counts[key][lbl]:
            vocab_counts[key][lbl][word] = 0
          vocab_counts[key][lbl][word] += 1
          if word not in all_vocab:
            all_vocab[word] = 0
          all_vocab[word] += 1
          all_toks_count += 1
          if word not in word_to_sents[lbl]:
            word_to_sents[lbl][word] = set()
          word_to_sents[lbl][word].add(sent)

  p_label = {}
  lbls_Z = sum([lbl_counts[key] for key in lbl_counts])
  for key in lbl_counts:
    p_label[key] = lbl_counts[key] / float(lbls_Z)
          
  df = None
  if not use_preds:
    df = pd.DataFrame(columns=['gold-lbl', 'word', 'percent', 'total', 'lbl_given_word', 'sents_with_word', 'sents_per_lbl'])
    percent_to_sent = {}
    for word in all_vocab:
      lbl_per_w = {}
      counts_p_lbl = {}
      for lbl in set(vocab_counts['correct'].keys() + vocab_counts['wrong'].keys()):
        correct, wrong = 0, 0
        if lbl in vocab_counts['correct']:
          if word in vocab_counts['correct'][lbl]:
            correct = vocab_counts['correct'][lbl][word]
        if lbl in vocab_counts['wrong']:
          if word in vocab_counts['wrong'][lbl]:
            wrong = vocab_counts['wrong'][lbl][word]
        count = (correct + wrong) / float(all_vocab[word])
        p_w = float(correct + wrong) / all_toks_count
        if p_w:
          lbl_per_w[lbl] = (count * p_label[lbl]) / p_w
        else:
          lbl_per_w[lbl] = 0
        counts_p_lbl[lbl] = count
        if word not in word_to_sents[lbl]:
          word_to_sents[lbl][word] = set()
      #pdb.set_trace()
      lbl_sum = sum([lbl_per_w[lbl] for lbl in lbl_per_w])
      for lbl in lbl_per_w:
        if lbl not in percent_to_sent:
          percent_to_sent[lbl] = {}
        if counts_p_lbl[lbl] not in percent_to_sent[lbl]:
          percent_to_sent[lbl][counts_p_lbl[lbl]] = set()
        if all_vocab[word] >= threshold and counts_p_lbl[lbl] >= threshold_percent:
          df = df.append({'gold-lbl': lbl, 'word': word, 'percent': counts_p_lbl[lbl], 'total': all_vocab[word], \
                         'lbl_given_word': lbl_per_w[lbl] / lbl_sum, \
                         'sents_with_word': len(word_to_sents[lbl][word].difference(percent_to_sent[lbl][counts_p_lbl[lbl]])), 'sents_per_lbl': lbl_counts[lbl]}, ignore_index = True)
          for sent in word_to_sents[lbl][word]:
            percent_to_sent[lbl][counts_p_lbl[lbl]].add(sent)

    return df, all_vocab

  df = pd.DataFrame(columns=['gold-lbl', 'correct', 'word', 'count'])
  for correct in vocab_counts:
    for lbl in vocab_counts[correct]:
      for word in vocab_counts[correct][lbl]:
        #all_vocab.add(word)
        df = df.append({'gold-lbl': lbl, 'correct': correct, 'word': word, 'count': vocab_counts[correct][lbl][word]}, ignore_index=True)
        
  for correct in vocab_counts:
    for lbl in vocab_counts[correct]:
      for word in all_vocab.difference(vocab_counts[correct][lbl].keys()):
        #print word
        df = df.append({'gold-lbl': lbl, 'correct': correct, 'word': word, 'count': 0}, ignore_index=True)
  

  return df, all_vocab


def main():
  
  args = get_args()

  if args.gold_lbl and args.pred_lbl and args.hyp_src:
    data = get_sents(args.gold_lbl, args.pred_lbl, args.hyp_src, args.data_split)
    #df, vocab = get_vocab_counts(data, args.preds, args.vocab_thresh, args.percent_keep)
    df = get_plot(data, args.vocab_thresh)
    #df_sorted = df.sort_values(by=['percent', 'total'], ascending=False)
    df.to_pickle("%s_sents_prob_count.pkl" % (args.data_src)) 
    '''
    #for lbl in set(df_sorted['gold-lbl']):
      #df = df_sorted[df_sorted['gold-lbl'] == lbl]
      #df = df.head(args.top_k)
    #pdb.set_trace()
      html = df.to_html()
      f_out = open("%s_top%d_mincount%d_minpercent%d_lnl-%s.html" % (args.data_src, args.top_k, args.vocab_thresh, args.percent_keep, lbl), "wb")
      f_out.write(html)
      f_out.close() 
    #df.to_pickle("%s_top%d_mincount%d_minpercent%d.pkl" % (args.data_src, args.top_k, args.vocab_thresh, args.percent_keep)) 

    df = df[df['percent'] == 1]
    #pdb.set_trace()
    #html = df.to_html()
    #f_out = open("%s_top%d_mincount%d_minpercent%d_lbl-%s_percent1.html" % (args.data_src, args.top_k, args.vocab_thresh, args.percent_keep, lbl), "wb")
    #f_out.write(html)
    #f_out.close() 
    '''

if __name__ == '__main__':
  main()

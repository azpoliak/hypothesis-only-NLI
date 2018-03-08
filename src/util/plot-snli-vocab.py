import pdb
import argparse
import pandas as pd

def get_args():
  parser = argparse.ArgumentParser(description='Evaluate NLI based on prediction files.')
  parser.add_argument('--gold_lbl', type=str, help='The gold train labels file')
  parser.add_argument('--pred_lbl', type=str, help='The predicted labels file')
  parser.add_argument('--hyp_src', type=str, help='File containing the hypothesis sentences')
  #/export/b02/apoliak/nli-hypothes-only/targeted-nli/cl_sprl_
  parser.add_argument('--data_split', type=str, help='Which split of datset is being used')

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


  for key in correct:  
    print "%s Accuracy on %s: %f" % (data_split, key, correct[key] / float(tot_lbl[key]))
  for key in tot_lbl:  
    print "%s MAJ on %s: %f" % (data_split, key, tot_lbl[key] / float(tot_lbl['total']))
  print

  return sents

def get_vocab_counts(data):
  all_vocab = set()
  vocab_counts = {'correct': {}, 'wrong': {}}
  for key in data:
    for lbl in data[key]:
      if lbl not in vocab_counts[key]:
        vocab_counts[key][lbl] = {}
      for sent in data[key][lbl]:
        for word in sent.split():
          if word not in vocab_counts[key][lbl]:
            vocab_counts[key][lbl][word] = 0
          vocab_counts[key][lbl][word] += 1
          

  df = pd.DataFrame(columns=['gold-lbl', 'correct', 'word', 'count'])
  for correct in vocab_counts:
    for lbl in vocab_counts[correct]:
      for word in vocab_counts[correct][lbl]:
        all_vocab.add(word)
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
    data.to_pickle("sentences.pkl")
    df, vocab = get_vocab_counts(data)
    df.to_pickle("tokens_count.pkl") 
    #pdb.set_trace()


if __name__ == '__main__':
  main()

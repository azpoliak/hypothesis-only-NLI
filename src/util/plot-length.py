import pdb
import argparse
import pandas as pd

def get_args():
  parser = argparse.ArgumentParser(description='Sentence lengths.')
  parser.add_argument('--gold_lbl', type=str, help='The gold train labels file')
  parser.add_argument('--pred_lbl', type=str, help='The predicted labels file')
  parser.add_argument('--hyp_src', type=str, help='File containing the hypothesis sentences')
  parser.add_argument('--data_split', type=str, help='Which split of datset is being used')
  parser.add_argument('--data_src', type=str, help='Which dataset is being used')

  parser.add_argument('--preds', type=int, help="Determine whether tosplit based on predictions or not")
  parser.add_argument('--percent_keep', type=int, default=40, help="Only keep high percentages higher than this")
  parser.add_argument('--top_k', type=int, default=100, help="Top k examples to keep. k=100")

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

  return sents


def get_sent_lens(data, use_preds):
 
  all_lens = {}
  sent_lens = {'correct': {}, 'wrong': {}}
  
  for key in data:
    for lbl in data[key]:
      if lbl not in sent_lens[key]:
        sent_lens[key][lbl] = {}
      for sent in data[key][lbl]:
        l = len(sent.split())
        if l not in sent_lens[key][lbl]:
          sent_lens[key][lbl][l] = 0
        sent_lens[key][lbl][l] += 1
        if l not in all_lens:
          all_lens[l] = 0
        all_lens[l] += 1

  df = None
  if not use_preds:
    df = pd.DataFrame(columns=['gold-lbl', 'len', 'count', 'total'])
    for l in all_lens:
      for lbl in set(sent_lens['correct'].keys() + sent_lens['wrong'].keys()):
        correct, wrong = 0, 0
        if lbl in sent_lens['correct']:
          if l in sent_lens['correct'][lbl]:
            correct = sent_lens['correct'][lbl][l]
        if lbl in sent_lens['wrong']:
          if l in sent_lens['wrong'][lbl]:
            wrong = sent_lens['wrong'][lbl][l]
        count = (correct + wrong) / float(all_lens[l])
        df = df.append({'gold-lbl': lbl, 'len': l, 'count': count, 'total': all_lens[l]}, ignore_index = True)

    return df, all_lens

  df = pd.DataFrame(columns=['gold-lbl', 'correct', 'len', 'count'])
  for correct in sent_lens:
    for lbl in sent_lens[correct]:
      for l in sent_lens[correct][lbl]:
        df = df.append({'gold-lbl': lbl, 'correct': correct, 'len': l, 'count': sent_lens[correct][lbl][l]}, ignore_index=True)

  for correct in sent_lens:
    for lbl in sent_lens[correct]:
      for l in all_lens.difference(sent_lens[correct][lbl].keys()):
        df = df.append({'gold-lbl': lbl, 'correct': correct, 'len': l, 'count': 0}, ignore_index=True)

  return df, all_lens



def main():
  
  args = get_args()

  if args.gold_lbl and args.pred_lbl and args.hyp_src:
    data = get_sents(args.gold_lbl, args.pred_lbl, args.hyp_src, args.data_split)
    df2, lengths = get_sent_lens(data, args.preds)
    df2 = df2.sort_values(by=['count', 'total'], ascending=False)
    df2 = df2.head(args.top_k)
    html2 = df2.to_html()
    f_out2 = open("%s_top%d_lens.html" % (args.data_src, args.top_k), "wb")
    f_out2.write(html2)
    f_out2.close()


if __name__ == '__main__':
  main()

import pdb
import argparse

def get_args():
  parser = argparse.ArgumentParser(description='Evaluate NLI based on prediction files.')
  parser.add_argument('--train_gold_lbl', type=str, help='The gold train labels file')
  parser.add_argument('--dev_gold_lbl', type=str, help='The gold dev labels file')
  parser.add_argument('--test_gold_lbl', type=str, help='The gold test labels file')
  parser.add_argument('--train_pred_lbl', type=str, help='The predicted train labels file')
  parser.add_argument('--dev_pred_lbl', type=str, help='The predicted dev labels file')
  parser.add_argument('--test_pred_lbl', type=str, help='The pred test labels file')
  parser.add_argument('--nli_data', type=str, help='The name of the NLI dataset')

  args = parser.parse_args()
  return args

def compute_acc(gold_f, pred_f, data_split, nli_data):
  gold_lbls = open(gold_f).readlines()
  pred_lbls = open(pred_f).readlines()

  assert (len(gold_lbls) == len(pred_lbls))

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

  for key in correct:  
    print "%s\t%s Accuracy on %s: %f" % (nli_data, data_split, key, correct[key] / float(tot_lbl[key]))
  for key in tot_lbl:  
    print "%s\t%s MAJ on %s: %f" % (nli_data, data_split, key, tot_lbl[key] / float(tot_lbl['total']))
 
  print "Remember: MAJ on total is the highest for the labels"
  print 
  print

def main():
  
  args = get_args()

  print "Results for %s" % (args.nli_data)

  if args.train_gold_lbl and args.train_pred_lbl:
    compute_acc(args.train_gold_lbl, args.train_pred_lbl, 'train', args.nli_data)
  if args.dev_gold_lbl and args.dev_pred_lbl:
    compute_acc(args.dev_gold_lbl, args.dev_pred_lbl, 'dev', args.nli_data)
  if args.test_gold_lbl and args.test_pred_lbl:
    compute_acc(args.test_gold_lbl, args.test_pred_lbl, 'test', args.nli_data)



if __name__ == '__main__':
  main()

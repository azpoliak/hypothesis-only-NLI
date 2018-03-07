import pdb
import argparse
import random

def get_args():
  parser = argparse.ArgumentParser(description='Evaluate NLI based on prediction files.')
  parser.add_argument('--train_gold_lbl', type=str, help='The gold train labels file')
  parser.add_argument('--train_source', type=str, help='The train source file')
  parser.add_argument('--dev_gold_lbl', type=str, help='The gold dev labels file')
  parser.add_argument('--dev_source', type=str, help='The dev source file')
  parser.add_argument('--test_gold_lbl', type=str, help='The gold test labels file')
  parser.add_argument('--test_source', type=str, help='The test source file')
  parser.add_argument('--train_pred_lbl', type=str, help='The predicted train labels file')
  parser.add_argument('--dev_pred_lbl', type=str, help='The predicted dev labels file')
  parser.add_argument('--test_pred_lbl', type=str, help='The pred test labels file')
  parser.add_argument('--nli_data', type=str, help='The name of the NLI dataset')

  args = parser.parse_args()
  return args

def compute_acc(gold_f, source_f, pred_f, data_split):
  gold_lbls = open(gold_f).readlines()
  pred_lbls = open(pred_f).readlines()
  source_example = open(source_f).readlines()

  assert (len(gold_lbls) == len(pred_lbls))

  correct = {'total': 0}
  tot_lbl = {'total': 0}

  examples_entailment = []
  examples_neutral = []
  examples_contradiction = []

  examples_entailment_wrong = []
  examples_neutral_wrong = []
  examples_contradiction_wrong = []

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

      if gold_lbl == "entailment":
        examples_entailment.append(source_example[i])
      elif gold_lbl == "neutral":
        examples_neutral.append(source_example[i])
      elif gold_lbl == "contradiction":
        examples_contradiction.append(source_example[i])
    else:
      if pred_lbls[i].strip() == "entailment":
        examples_entailment_wrong.append(source_example[i])
      elif pred_lbls[i].strip() == "neutral":
        examples_neutral_wrong.append(source_example[i])
      elif pred_lbls[i].strip() == "contradiction":
        examples_contradiction_wrong.append(source_example[i])

   

  for key in correct:  
    print "%s Accuracy on %s: %f" % (data_split, key, correct[key] / float(tot_lbl[key]))
  for key in tot_lbl:  
    print "%s MAJ on %s: %f" % (data_split, key, tot_lbl[key] / float(tot_lbl['total']))
  
  print len(examples_entailment)
  print len(examples_neutral)
  print len(examples_contradiction)

  random.shuffle(examples_entailment)
  random.shuffle(examples_neutral)
  random.shuffle(examples_contradiction)

  random.shuffle(examples_entailment_wrong)
  random.shuffle(examples_neutral_wrong)
  random.shuffle(examples_contradiction_wrong)

  write_file("snli_entailment_FULL.txt", examples_entailment)
  write_file("snli_neutral_FULL.txt", examples_neutral)
  write_file("snli_contradiction_FULL.txt", examples_contradiction)
  write_file("snli_entailment_wrong_FULL.txt", examples_entailment_wrong)
  write_file("snli_neutral_wrong_FULL.txt", examples_neutral_wrong)
  write_file("snli_contradiction_wrong_FULL.txt", examples_contradiction_wrong)

def write_file(filename, examples):
  f_out = open(filename, "wb")
  for ex in examples:
    f_out.write(str(ex))
    f_out.write("\n")
  f_out.close()

def main():
  
  args = get_args()

  print "Results for %s" % (args.nli_data)

  if args.train_gold_lbl and args.train_pred_lbl:
    compute_acc(args.train_gold_lbl, args.train_source, args.train_pred_lbl, 'train')
  if args.dev_gold_lbl and args.dev_pred_lbl:
    compute_acc(args.dev_gold_lbl, args.dev_source, args.dev_pred_lbl, 'dev')
  if args.test_gold_lbl and args.test_pred_lbl:
    compute_acc(args.test_gold_lbl, args.test_source, args.test_pred_lbl, 'test')



if __name__ == '__main__':
  main()

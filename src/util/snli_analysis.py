import pdb
import argparse
import random
import re
import matplotlib.pyplot as plt
import pandas as pd

def get_args():
  parser = argparse.ArgumentParser(description='Evaluate NLI based on prediction files.')
  parser.add_argument('--nli_data', type=str, help='The name of the NLI dataset')
  parser.add_argument('--data_split', type=str, help='Which split of datset is being used')

  args = parser.parse_args()
  return args

def hist(distr, label, correct):
  
  plt.figure(figsize=(20,10))
  plt.bar(distr.keys(), distr.values())
  plt.ylabel("Sentence length")
  plt.xlabel("# Sentences")
  plt.title("Histogram of sentence lengths: " + label + "/" + correct)
  plt.savefig("histo_sentence_lengths_"+label+"_"+correct)


def calc(filename):
  
  examples = []

  reg = '(.*)\|\|\|(.*)'
  pattern = re.compile(reg)

  lines = [line.rstrip('\n') for line in open(filename)]
  lengths = []
  lengths_distr = {}
 
  for l in lines: 
    result = pattern.match(l)
    if result is None:
      continue
    result = result.group(2)
    #print result
    sentence_length = len(result.split())
    if sentence_length not in lengths_distr:
      lengths_distr[sentence_length] = 0
    lengths_distr[sentence_length] += 1
    lengths.append(sentence_length)
   
  #print lengths
  #print filename
  #print "Average hypothesis length: " , sum(lengths)/len(lengths)
  #print lengths_distr
  return lengths_distr

def main():
  args = get_args()

  lbls = ["entailment", "neutral", "contradiction"]
  correct = ["correct", "wrong"]

  for lbl in lbls:
    for c in correct:
      filename = str(str(args.nli_data)+"_"+lbl+"_"+str(args.data_split)+".txt")

      len_distr = calc(filename)
      hist(len_distr, lbl, c)

  plt.show()  
  

if __name__ == '__main__':
  main()

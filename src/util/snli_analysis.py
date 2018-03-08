# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import string
import math
import pdb
import argparse
import random
import re
import matplotlib.pyplot as plt
import pandas as pd


from textblob import TextBlob as tb

def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)



tokenize = lambda doc: doc.lower().split(" ")

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


def sent_len(filename):
  
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


def sublinear_term_frequency(term, tokenized_document):
  count = tokenized_document.count(term)
  if count == 0:
    return 0
  return 1 + math.log(count)

def inverse_document_frequencies(tokenized_documents):
  idf_values = {}
  all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
  for tkn in all_tokens_set:
    contains_token = map(lambda doc: tkn in doc, tokenized_documents)
    idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(sum(contains_token)))
  return idf_values


def tf_idf(filename):

  reg = '(.*)\|\|\|(.*)'
  pattern = re.compile(reg)

  lines = [line.rstrip('\n') for line in open(filename)]
  all_documents = []
  bloblist = []

  for l in lines:
    result = pattern.match(l)
    if result is None:
      continue
    result = result.group(2)
#    all_documents.append(result)
    bloblist.append(tb(result))

#  tokenized_documents = [tokenize(d) for d in all_documents]
#  idf = inverse_document_frequencies(tokenized_documents)
#  tfidf_documents = []
#  for document in tokenized_documents:
#    doc_tfidf = []
#    for term in idf.keys():
#      tf = sublinear_term_frequency(term, document)
#      doc_tfidf.append(tf * idf[term])
#    tfidf_documents.append(doc_tfidf)
#  return tfidf_documents

  for i, blob in enumerate(bloblist):
    print("Top words in document {}".format(i + 1))
    scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:3]:
      print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))













def main():
  args = get_args()

  lbls = ["entailment", "neutral", "contradiction"]
  correct = ["correct", "wrong"]

  for lbl in lbls:
    for c in correct:
      c1 = "_wrong"
      if c == "correct": c1 = ''
      filename = str(str(args.nli_data)+"_"+lbl+"_"+str(args.data_split)+c1+".txt")

      #len_distr = sent_len(filename)
      #hist(len_distr, lbl, c)
      tf_idf_repr = tf_idf(filename)
      print len(tf_idf_repr[0])
  #plt.show()  
  

if __name__ == '__main__':
  main()

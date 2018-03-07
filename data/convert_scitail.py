import pandas as pd
import nltk
import operator

tag_set = set()

for f in ["train", "dev", "test"]:

  hyp2label = {}

  lbls, hypoths = [], []
  for line in open("scitail/SciTailV1/tsv_format/scitail_1.0_%s.tsv" % (f)):
    if f == "dev":
      f = "val"

    line = line.strip().split("\t")
    assert(len(line) == 3) 
    tag = line[-1]
    if tag == "entails":
      tag = "entailment"
    tag_set.add(tag)
    lbls.append(tag)
    hyp = line[1]
    hypoths.append(hyp) 

    if f == 'train':
      if hyp not in hyp2label:
        hyp2label[hyp] = {}
      if tag not in hyp2label[hyp]:
        hyp2label[hyp][tag] = 0
      hyp2label[hyp][tag] += 1

  lbl_out = open("scitail/cl_scitail_%s_lbl_file" % (f), "wb")
  source_out = open("scitail/cl_scitail_%s_source_file" % (f), "wb")
  for i in range(len(lbls)):
    lbl_out.write(lbls[i].strip() + "\n")
    source_out.write("|||" + " ".join(nltk.word_tokenize(hypoths[i].strip())) + "\n")

  lbl_out.close()
  source_out.close()

  if f == 'train':
    lbl_out = open("scitail/cl_scitail_train-no-dup_lbl_file", "wb")
    source_out = open("scitail/cl_scitail_train-no-dup_source_file", "wb")
    used_hyps = set()
    for i in range(len(lbls)):
      hyp = hypoths[i]
      if hyp in used_hyps:
        continue
      source_out.write("|||" + " ".join(nltk.word_tokenize(hypoths[i].strip())) + "\n")
      lbl_out.write(max(hyp2label[hyp].iteritems(), key=operator.itemgetter(1))[0].strip() + "\n")
      used_hyps.add(hyp)

  lbl_out.close()
  source_out.close()


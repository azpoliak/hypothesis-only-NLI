import pandas as pd
import nltk

tag_set = set()
for f in ["train", "dev", "test"]:
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

  lbl_out = open("scitail/cl_scitail_%s_lbl_file" % (f), "wb")
  source_out = open("scitail/cl_scitail_%s_source_file" % (f), "wb")
  for i in range(len(lbls)):
    lbl_out.write(lbls[i].strip() + "\n")
    source_out.write("|||" + " ".join(nltk.word_tokenize(hypoths[i].strip())) + "\n")

  lbl_out.close()
  source_out.close()

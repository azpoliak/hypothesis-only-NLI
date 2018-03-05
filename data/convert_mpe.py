import pandas as pd
import nltk
import pdb

for f in ["train", "dev", "test"]:
  line_count = -1
  lbls, hypoths = [], []
  for line in open("mpe/mpe_%s.txt" % (f)):
    line_count += 1
    if line_count == 0:
      continue
    line = line.split("\t")
    assert (len(line) == 10, "MPE %s file has a bad line at line numbers %d" % (f, line_count))
    lbls.append(line[-1].strip().split()[-1])
    hypoths.append("|||" + " ".join(nltk.word_tokenize(line[5].strip())))

  if f == "dev":
    f = "val"

  assert(len(hypoths) == len(set(hypoths)), "A hypothesis appears more than once")

  assert(len(lbls) == len(hypoths), "Number of labels and hypothesis for MPE %s do not match" % (f))
  lbl_out = open("mpe/cl_mpe_%s_lbl_file" % (f), "wb")
  source_out = open("mpe/cl_mpe_%s_source_file" % (f), "wb")
  for i in range(len(lbls)):
    lbl_out.write(lbls[i].strip() + "\n")
    source_out.write(hypoths[i] + "\n")

  lbl_out.close()
  source_out.close()


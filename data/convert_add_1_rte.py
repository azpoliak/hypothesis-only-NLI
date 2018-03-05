import pandas as pd
import nltk
import pdb


def convert_label(score, is_test):
  score = float(score)
  if is_test:
    if score <= 3:
      return "not-entailed"
    elif score >= 4:
      return "entailed"
    return

  if score < 3.5:
      return "not-entailed"
  return "entailed"

for f in ["train", "dev", "test"]:
  hyps_dict = {}
  line_count = -1
  lbls, hypoths = [], []
  for line in open("add-one-rte/AN-composition/addone-entailment/splits/data.%s" % (f)):
    line_count += 1
    line = line.split("\t")
    assert (len(line) == 7) # "add one rte %s file has a bad line" % (f))
    lbl = convert_label(line[0], f == "test")
    if not lbl:
      continue
    lbls.append(lbl)
    hyp = line[-1].replace("<b><u>", "").replace("</u></b>", "").strip()
    hypoths.append("|||" + " ".join(nltk.word_tokenize(hyp)))
    if hyp not in hyps_dict:
      hyps_dict[hyp] = 0
    hyps_dict[hyp] += 1

  if f == "dev":
    f = "val"

  print "In %s, there are %d hypothesis sentences with more than 1 context: " % (f, len([item for key,item in hyps_dict.items() if item > 1]))
  #assert(len(hypoths) == len(set(hypoths))) #, "A hypothesis appears more than once")
  assert(len(lbls) == len(hypoths)) #, "Number of labels and hypothesis for MPE %s do not match" % (f))

  lbl_out = open("add-one-rte/cl_add_one_rte_%s_lbl_file" % (f), "wb")
  source_out = open("add-one-rte/cl_add_one_rte_%s_source_file" % (f), "wb")
  for i in range(len(lbls)):
    lbl_out.write(lbls[i].strip() + "\n")
    source_out.write(hypoths[i] + "\n")

  lbl_out.close()
  source_out.close()



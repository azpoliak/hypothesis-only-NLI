import pandas as pd
import nltk
import pdb
import csv
import random

def convert_lbl(num):
  if num == 1:
    return 'contradiction'
  if num == 5:
    return 'entailment'
  return 'neutral' #  return num #  if 

lbl_count = {}

with open('joci/joci.csv', 'rb') as csvfile:
  pairs = []
  spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
  line_num = -1
  for row in spamreader:
    line_num += 1
    if line_num == 0:
      continue
    hyp_src = row[4]
    if "AGCI" not in hyp_src:
      continue
    hyp, lbl = row[1], convert_lbl(int(row[2]))
    if lbl not in lbl_count:
      lbl_count[lbl] = 0
    lbl_count[lbl] += 1
    pairs.append(("|||"+hyp.strip(), lbl))

print lbl_count
random.shuffle(pairs)

sent_num = 0
for data_pair in [("train", .80), ("val", .10), ("test", .10)]:
  f = data_pair[0]

  lbl_out = open("joci/cl_joci_%s_lbl_file" % (f), "wb")
  source_out = open("joci/cl_joci_%s_source_file" % (f), "wb")
  for i in range(sent_num, sent_num + (int(len(pairs) * data_pair[1]))):
    lbl_out.write(str(pairs[i][1]) + "\n")
    source_out.write(pairs[i][0] + "\n")

  sent_num = sent_num + int(len(pairs) * data_pair[1])
  print sent_num

  lbl_out.close()
  source_out.close()


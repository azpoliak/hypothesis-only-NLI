import pandas as pd
import nltk

test = {'lbls': [], 'hypoths': []}
train = {'lbls': [], 'hypoths': []}
dev = {'lbls': [], 'hypoths': []}
DATA = {"TEST": test, "TRAIN" : train, "TRIAL": dev}

tag_set = set()

line_count = -1
for line in open("sick/SICK.txt"):
  line_count += 1
  if line_count == 0:
    continue
  line = line.split("\t")
  tag = line[-1].strip()
  if tag not in DATA.keys():
    print "Bad tag: %s" % (tag)
  hyp = "|||" + " ".join(nltk.word_tokenize(line[2].strip()))
  lbl = line[3]
  DATA[tag]['lbls'].append(lbl)
  DATA[tag]['hypoths'].append(hyp)
  tag_set.add(line[-1])


print tag_set

for pair in [("TEST", "test"), ("TRAIN", "train"), ("TRIAL", "val")]:
  print "Number of %s examples: %d" % (pair[1], len(DATA[pair[0]]['lbls']))
  lbl_out = open("sick/cl_sick_%s_lbl_file" % (pair[1]), "wb")
  source_out = open("sick/cl_sick_%s_source_file" % (pair[1]), "wb")
  for i in range(len(DATA[pair[0]]['lbls'])):
    lbl_out.write(DATA[pair[0]]['lbls'][i].strip() + "\n")
    source_out.write(DATA[pair[0]]['hypoths'][i].strip() + "\n")

  source_out.close()
  lbl_out.close()

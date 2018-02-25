
def get_nli_hypoth(nlipath):
  labels = {}
  hypoths = {}

  labels_to_int = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

  for data_split in ['train', 'val', 'test']:
    labels[data_split], hypoths[data_split] = [], []

    lbls = open("%s/cl_snli_%s_lbl_file" % (nlipath, data_split)).readlines()
    srcs = open("%s/cl_snli_%s_source_file" % (nlipath, data_split)).readlines()

    assert (len(lbls) == len(srcs), "SNIL %s labels and source files are not same length" % (data_split))

    for i in range(len(lbls)):
      lbl = lbls[i].strip()
      hypoth = srcs[i].split("|||")[-1].strip()

      if lbl not in labels_to_int:
        print "bad label: %s" % (lbl)
        continue

      labels[data_split].append(labels_to_int[lbl])
      hypoths[data_split].append(hypoth)


  train = {'lbls': labels['train'], 'hypoths' : hypoths['train']}
  val = {'lbls': labels['val'], 'hypoths' : hypoths['val']}
  test = {'lbls': labels['test'], 'hypoths' : hypoths['test']}

  return train, val, test

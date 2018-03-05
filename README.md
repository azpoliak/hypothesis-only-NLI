# Hypothesis-only NLI baselines

*Add description about the repo & project*

## Requirements
All code in the repo relies on pytho2.7. Running `pip install -r requirements.txt` from the root of the project will install the required python packages.

This project relies on [pytorch](http://pytorch.org/). 




## Data
We provide a bash script that can be used to downlod all data used in our experiments. The script also cleans and processes the data.
To get and process the data, run `data/get_data.sh`.  

## Training

To train a hypothesis-only NLI model, use `src/train.py`.

All command line arguments are initialized with default values. If you ran `get_data.sh` as described above, all of the paths will be set directly and you can just run `src/train.py`. 

The most useful command line arguments are:

- `embdfile` - File containin the word embeddings
- `outputdir` - Output directory to store the model after training
- `train_lbls_file` NLI train data labels file
- `train_src_file`  NLI train data source file
- `val_lbls_file`  NLI validation (dev) data labels file
- `val_src_file`   NLI validation (dev) data source file
- `test_lbls_file` NLI test data labels file
- `test_src_file`  NLI test data source file 

To see a description of more command line arguments, run `src/train.py --help`.


## Bibligoraphy

If you use our code, please cite us using the following citation

```
Enter bibtex here
```

We additionally provide the bibliographies for the datasets that we use as well. If you use any of those datasets, please cite them as well.

```
@InProceedings{white-EtAl:2017:I17-1,
  author    = {White, Aaron Steven  and  Rastogi, Pushpendre  and  Duh, Kevin  and  Van Durme, Benjamin},
  title     = {Inference is Everything: Recasting Semantic Resources into a Unified Evaluation Framework},
  booktitle = {Proceedings of the Eighth International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
  month     = {November},
  year      = {2017},
  address   = {Taipei, Taiwan},
  publisher = {Asian Federation of Natural Language Processing},
  pages     = {996--1005},
  url       = {http://www.aclweb.org/anthology/I17-1100}
}

@article{williams2017broad,
  title={A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference},
  author={Williams, Adina and Nangia, Nikita and Bowman, Samuel R},
  journal={arXiv preprint arXiv:1704.05426},
  year={2017}
}

@inproceedings{snli:emnlp2015,
        Author = {Bowman, Samuel R. and Angeli, Gabor and Potts, Christopher and Manning, Christopher D.},
        Booktitle = {Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
        Publisher = {Association for Computational Linguistics},
        Title = {A large annotated corpus for learning natural language inference},
        Year = {2015}
}

@InProceedings{lai-bisk-hockenmaier:2017:I17-1,
  author    = {Lai, Alice  and  Bisk, Yonatan  and  Hockenmaier, Julia},
  title     = {Natural Language Inference from Multiple Premises},
  booktitle = {Proceedings of the Eighth International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
  month     = {November},
  year      = {2017},
  address   = {Taipei, Taiwan},
  publisher = {Asian Federation of Natural Language Processing},
  pages     = {100--109},
  url       = {http://www.aclweb.org/anthology/I17-1011}
}

@inproceedings{scitail,
     author = {Tushar Khot and Ashish Sabharwal and Peter Clark},
     booktitle = {AAAI},
     title = {{SciTail}: A Textual Entailment Dataset from Science Question Answering},
     year = {2018}
}

@InProceedings{P16-1204,
  author =      "Pavlick, Ellie
                and Callison-Burch, Chris",
  title =       "Most ``babies'' are ``little'' and most ``problems'' are ``huge'': Compositional      Entailment in Adjective-Nouns    ",
  booktitle =   "Proceedings of the 54th Annual Meeting of the Association for      Computational Linguistics (Volume 1: Long Papers)    ",
  year =        "2016",
  publisher =   "Association for Computational Linguistics",
  pages =       "2164--2173",
  location =    "Berlin, Germany",
  doi =         "10.18653/v1/P16-1204",
  url =         "http://www.aclweb.org/anthology/P16-1204"
}

@inproceedings{MARELLI14.363.L14-1314,
    author = {Marco Marelli and Stefano Menini and Marco Baroni and Luisa Bentivogli and Raffaella bernardi and Roberto Zamparelli},
    url = {http://www.lrec-conf.org/proceedings/lrec2014/pdf/363_Paper.pdf},
    note = {ACL Anthology Identifier: L14-1314},
    title = {A SICK cure for the evaluation of compositional distributional semantic models},
    booktitle = {Proceedings of the Ninth International Conference on Language Resources and Evaluation (LREC'14)},
    year = {2014},
    month = {May},
    date = {26-31},
    address = {Reykjavik, Iceland},
    publisher = {European Language Resources Association (ELRA)},
    isbn = {978-2-9517408-8-4},
    pages = {216--223}
}
```


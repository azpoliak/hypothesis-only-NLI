import pdb
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def hist(df, label):
  df = df.loc[df['gold-lbl'] == label]
  #print df
  small_df = df.sort_values(by=['count'], ascending=False)[:10]
  print small_df
  x_word = [item for item in small_df['word']]
  x_lbl = [item for item in small_df['gold-lbl']]
  y = [item for item in small_df['count']]
  
  xn = range(len(x_word))
  plt.bar(xn, y)
  plt.xticks(xn, x_word, rotation=60)
  plt.ylabel("Count")
  plt.xlabel("Token")
  plt.title("Histogram of token counts: " + label)
  plt.show()


def main():
  
  df = pd.read_pickle('tokens_count.pkl')

  hist(df, "entailment")
  hist(df, "neutral")
  hist(df, "contradiction")


if __name__ == '__main__':
  main()


import pdb
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#stopwords = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']
stopwords = ['.', ',', 'a', 'an', 'A', 'the', 'The', 'Two', 'two', 'Three', 'three', 'is', 'are', 'in', 'on', 'of', 'at', 'to', 'for', 'with', 'and', 'there', 'There', 'by', 'from', 'has', 'man', 'woman', 'person', 'men', 'women', 'boy', 'girl', 'people', 'People', 'his', 'her', 'there', 'some', '\'s']

def hist(df, label, correct):
  df = df.loc[(df['gold-lbl'] == label) & (df['correct'] == correct)]
  #print df
  small_df = df.sort_values(by=['count'], ascending=False)[:40]
  print small_df
  
  x_word = [item for item in small_df['word']]
  x_lbl = [item for item in small_df['gold-lbl']]
  y = [item for item in small_df['count']]
  xn = range(len(x_word))
  
  plt.figure(figsize=(20,10))
  plt.bar(xn, y)
  plt.xticks(xn, x_word, rotation=60)
  plt.ylabel("Count")
  plt.xlabel("Token")
  plt.title("Histogram of token counts: " + label + "/" + correct)
  plt.savefig("histo_token_counts_"+label+"_"+correct)


def main():
  
  df = pd.read_pickle('tokens_count.pkl')
  df = df.loc[~df['word'].isin(stopwords)]

  hist(df, "entailment", "correct")
  hist(df, "entailment", "wrong")
  hist(df, "neutral", "correct")
  hist(df, "neutral", "wrong")
  hist(df, "contradiction", "correct")
  hist(df, "contradiction", "wrong")
  plt.show()  
  
if __name__ == '__main__':
  main()


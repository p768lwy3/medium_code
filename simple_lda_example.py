"""
Source: https://de.dariah.eu/tatom/topic_model_visualization.html
"""

# Import
import jieba as jb,  numpy as np, os, pandas as pd, random, re
import matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns
from matplotlib.font_manager import FontProperties
from os import path
from subprocess import check_output
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import get_single_color_func, WordCloud

stopwords = [',', '?', '、', '。', '“', '”', '《', '》', '！', '，', '：', '；', '？', 
             '（', '）', ',', ':', 'hi', 'auntie', 'ok', '向左走', '向右走', '大家', '利申', 
             '雖然', '但係', '乜', '一齊', '可以', '應該', '好多', '已經', '因為', '邊個',
             '好似', '而家', '一定', '之前', '即刻', '好過', '仲有', '如果', '其實', '一半',
             '有人', '個人', '一次', '無人', '好好', '根本', '一樣', '成日', '問題', '不過',
             '有時', '之後', '沒有', '所以', '不如', '個個', '無法']
mpl.rcParams['font.size'] = 12
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['figure.subplot.bottom'] = .1
ChineseFont1 = FontProperties(fname='./hanazono-20160201 (1)/HanaMinA.ttf')
sns.set(color_codes=True)

def sent_token(sent, StopWords=True, RemoveHttp=True):
  if RemoveHttp == True:
    sent = re.sub(r'^https?:\/\/.*[\r\n]*', '', sent, flags=re.MULTILINE)
  words = '/'.join(jb.cut(sent)).split('/')
  if StopWords == True:
    words = [w for w in words if w not in stopwords]
  return words

def tfidfvectorizer(words_list, max_features=1000, n_top_words=20, n_components=5, return_model=False):
  def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
      print("Topic #%d:" % topic_idx)
      print(" ".join([feature_names[i]
                      for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

  sents = []
  for words in words_list:
    sents.append(' '.join(words))
  vtr = TfidfVectorizer(max_df=0.95, min_df=3,
                        max_features=max_features)
  vtr_sents = vtr.fit_transform(sents)

  lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                  learning_method='online',
                                  learning_offset=50,
                                  random_state=12345)
  lda.fit(vtr_sents)
  if return_model == True:
    return lda, vtr_sents, vtr
  vtr_feature_names = vtr.get_feature_names()
  #print_top_words(lda, vtr_feature_names, n_top_words)
  lda_words_list = []
  for topic_idx, topic in enumerate(lda.components_):
    term = [topic_idx, [vtr_feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]]
    lda_words_list.append(term)
  return lda_words_list

def proportion_of_topic(model, feature_names, num_top_words=10, num_topics=5):
  components = model.components_.T
  word_topic = components / np.sum(components, axis=0)
  fontsize_base = 70 / np.max(word_topic)
  for i in range(num_topics):
    plt.subplot(1, num_topics, i + 1)
    plt.ylim(0, num_top_words + 0.5)
    plt.xticks([])
    plt.yticks([])
    plt.title('Topic #{}'.format(i))
    top_words_idx = np.argsort(word_topic[:, i])[::-1]
    top_words_idx = top_words_idx[:num_top_words]
    top_words = []
    for idx in top_words_idx:
      top_words.append(feature_names[idx])
    top_words_shares = word_topic[top_words_idx, i]
    for j, (word, share) in enumerate(zip(top_words, top_words_shares)):
      plt.text(0.3, num_top_words-j-0.5, word, fontsize=fontsize_base*share, fontproperties=ChineseFont1)
  plt.show()

class GroupedColorFunc(object):
  # Source: https://amueller.github.io/word_cloud/auto_examples/colored_by_group.html?highlight=color_func
  def __init__(self, color_to_words, default_color):
    self.color_func_to_words = [
        (get_single_color_func(color), set(words))
        for (color, words) in color_to_words.items()]
    self.default_color_func = get_single_color_func(default_color)

  def get_color_func(self, word):
    """Returns a single_color_func associated with the word"""
    try:
      color_func = next(
          color_func for (color_func, words) in self.color_func_to_words
          if word in words)
    except StopIteration:
      color_func = self.default_color_func
    return color_func

  def __call__(self, word, **kwargs):
    return self.get_color_func(word)(word, **kwargs)

def word_clouds(terms, groupbycolor=True, num_top_words=20, num_topics=5):
  def terms_to_wordcounts(terms, multiplier=1000):
    wordcounts = ''
    for i in terms:
      for j in range(num_top_words):
        wordcounts += ' '.join(int(((num_top_words - j) * multiplier)) * [i[1][j]])
    return wordcounts
  wordcounts = terms_to_wordcounts(terms)
  font_path = './hanazono-20160201 (1)/HanaMinA.ttf'
  wordcloud = WordCloud(font_path=font_path, background_color='white', collocations=False).generate(wordcounts)
  if groupbycolor == True:
    default_color = 'grey'
    color_list = []
    r = lambda: random.randint(200, 255)
    for i in range(num_topics):
      color_list.append('#%02X%02X%02X' % (r(), r(), r()))
    color_to_words = {color_list[i]: terms[i][1] for i in range(num_topics)}
    grouped_color_func = GroupedColorFunc(color_to_words, default_color)
    wordcloud.recolor(color_func=grouped_color_func)
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis("off")
  plt.show()

def simple_desc():
  nums_list = []
  data_files = []
  folder = './input/'
  for subdir, dirs, files in os.walk(folder):
    for f in files:
      data_files.append(subdir + f)
  for f in data_files:
    df = pd.read_csv(f)
    nums_list.append(len(df))
  nums_list = np.array(nums_list)
  sns.distplot(nums_list, bins=20, kde=False)
  plt.xlabel('Numbers of Reply')
  plt.ylabel('Freqency')
  plt.show()

def main():
  words_list = []
  data_files = []
  folder = './input/'
  for subdir, dirs, files in os.walk(folder):
    for f in files:
      data_files.append(subdir + f)
  for f in data_files:
    df = pd.read_csv(f)
    sents = df.response.values
    for sent in sents:
      if not isinstance(sent, str):
        continue
      sent = sent.split('|')
      for s in sent:
        words_list.append(sent_token(s))
  lda_words_list = tfidfvectorizer(words_list)
  word_clouds(lda_words_list)
  #sklearn_lda, sklearn_wordvec, sklearn_wordvecmodel = tfidfvectorizer(words_list, return_model=True)
  #proportion_of_topic(sklearn_lda, sklearn_wordvecmodel.get_feature_names())
      

if __name__ == '__main__':
  main()
  #simple_desc()

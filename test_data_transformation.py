# -*- coding: utf-8 -*-
# encoding=utf8
import sys, json, re
import numpy as np
from vaderSentiment import vaderSentiment as vs

import threading
from threading import Lock, Thread


lock = Lock()



def take_next_free():
    global hashset
    lock.acquire()
    if len(hashset) <= 0:
        res = -1
    else:
        res = hashset.pop()
    lock.release()
    return res

class runnable_class:

    def __init__(self, id):
        self.id = id

    def foo(self):
        global index
        global train_x
        global train_y
        global train_idd
        global train_data
        global test_data
        global test_idd
        global affect
        global all_train_x
        global all_train_y
        N = index
        #for t in range(N):
        if True:
            t = self.currently_handling_index
            for i in range(15):
                print t, i

                p = prediction(all_train_x[i], all_train_y[i], [np.ma.concatenate([test_x[i][t]])])
                print "   #", test_idd[t]
                if not ("ainl" in test_data[affect][test_idd[t]]):
                    test_data[affect][test_idd[t]]["ainl"] = []
                test_data[affect][test_idd[t]]["ainl"].append(p[0])
        print self.id, self.currently_handling_index, "finished"

    def run(self):
        global hashset
        while (True):
            self.currently_handling_index = take_next_free()
            if self.currently_handling_index == -1:
                break
            self.foo()

analyzer = vs.SentimentIntensityAnalyzer()
reload(sys)
sys.setdefaultencoding('utf8')
affect_list = ["anger", "fear", "joy", "sadness"]
import os
cwd = os.getcwd()
print cwd
with open(os.path.join(cwd, 'dataset', 'task1', 'train', 'dataset_json_task_1.txt')) as data_file:
    train_data = json.load(data_file)
with open(os.path.join(cwd, 'dataset', 'task1', 'development', 'dataset_json_development.txt')) as data_file:
    development_data = json.load(data_file)
'''with open(os.path.join(cwd, 'dataset', 'task1', 'test', 'dataset_json_test.txt')) as data_file:
    test_data = json.load(data_file)
'''

test_data = development_data


import sklearn.ensemble, sklearn.metrics#, sklearn.cross_validation
from sklearn.metrics import mean_squared_error, r2_score
import scipy
import math
import numpy as np
import time

def average(x):
    assert len(x) > 0
    return float(sum(x)) / len(x)

def pearson_def(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff

    return diffprod / math.sqrt(xdiff2 * ydiff2)

######################################################################################

dict_bag_of_words = {}
dict_curr_number = 0

def bag_of_words(word):
    global dict_curr_number
    if not word in dict_bag_of_words:
        dict_bag_of_words[word] = dict_curr_number
        dict_curr_number += 1
for affect in affect_list:
    for idd in train_data[affect]:
        tmp = train_data[affect][idd]["atext_clear_lower"].split(" ")
        for word in tmp:
            bag_of_words(word)
n_bag_of_words = len(dict_bag_of_words)
print n_bag_of_words
def make_vector_from_bag_of_words(sentence):
    tmp_list = sentence.split(" ")
    res = [0] * n_bag_of_words
    for word in tmp_list:
        res[dict_bag_of_words[word]] += 1
    return res


###################################################################################



with open(os.path.join(cwd, 'NRC-Sentiment-Emotion-Lexicons', 'AutomaticallyGeneratedLexicons', 'NRC-Emoticon-AffLexNegLex-v1.0', 'Emoticon-AFFLEX-NEGLEX-unigrams.txt')) as data_file:
    rnc_Emoticon_AFFLEX_NEGLEX_unigrams = data_file.readlines()

dict_rnc_Emoticon_AFFLEX_NEGLEX_unigrams = {}

for row in rnc_Emoticon_AFFLEX_NEGLEX_unigrams:
   row = row.split("\t")[0:2]
   dict_rnc_Emoticon_AFFLEX_NEGLEX_unigrams[row[0]] = float(row[1])

def func_rnc_Emoticon_AFFLEX_NEGLEX_unigrams(sentence):
   #minn = 0.0
   avg = 0.0
   #maxx = 0.0
   #less_than_avg_cnt = 0.0
   #more_than_avg_cnt = 0.0
   NN = 2
   list_of_NN_min = []
   list_of_NN_max = []   
   sentence_tmp = sentence.strip().replace("  ", " ").split(" ")
   n = len(sentence_tmp)
   for i in range(n):
     for x in range(2):
       if x == 0: w = sentence_tmp[i]
       elif i < n-1: w = sentence_tmp[i] + " " + sentence_tmp[i+1]
       else: continue
       if w in dict_rnc_Emoticon_AFFLEX_NEGLEX_unigrams:
          val = dict_rnc_Emoticon_AFFLEX_NEGLEX_unigrams[w]
          #minn = min(minn, val)
          if len(list_of_NN_min) < NN or (len(list_of_NN_min) >= NN and val < list_of_NN_min[NN-1]):
             list_of_NN_min.append(val)
             list_of_NN_min.sort()
             if (len(list_of_NN_min) > NN):
                 list_of_NN_min = list_of_NN_min[:-1]
          elif len(list_of_NN_max) < NN or (len(list_of_NN_max) >= NN and val > list_of_NN_max[0]):
             list_of_NN_max.append(val)
             list_of_NN_max.sort()
             if (len(list_of_NN_max) > NN):
                 list_of_NN_max = list_of_NN_max[1:]
          avg += val
          
          #maxx = max(maxx, val)
   '''
   for i in range(n):
     for x in range(2):
       if x == 0: w = sentence_tmp[i]
       elif i < n-1: w = sentence_tmp[i] + " " + sentence_tmp[i+1]
       else: continue
       if w in dict_rnc_Emoticon_AFFLEX_NEGLEX_unigrams:
          val = dict_rnc_Emoticon_AFFLEX_NEGLEX_unigrams[w]
          if val < avg:
            less_than_avg_cnt += 1.0
          else:
            more_than_avg_cnt += 1.0
   '''
   #return [minn, less_than_avg_cnt/(float(n)), avg, more_than_avg_cnt/(float(n)), maxx]
   n_lmin = len(list_of_NN_min)
   n_lmax = len(list_of_NN_max)
   how_much_insufficient_min = NN - n_lmin
   how_much_insufficient_max = NN - n_lmax
   a = 0.0
   if n_lmin > 0:
       a = list_of_NN_min[n_lmin-1]
   for i in range(how_much_insufficient_min):
       list_of_NN_min.append(a)
   if n_lmax > 0:
       a = list_of_NN_max[n_lmax-1]
   for i in range(how_much_insufficient_max):
       list_of_NN_max.append(a) 
   list_of_NN_min.sort()
   list_of_NN_max.sort() 
   return [avg] + list_of_NN_min + list_of_NN_max


##########################################################################################




with open(os.path.join(cwd, 'DepecheMood_V1.0', 'DepecheMood_freq.txt')) as data_file:
    depecheMood = data_file.readlines()
depecheMood = depecheMood[1:]

import re
dict_depecheMood = {}
count = 0.0
tag_clear_previous = "zzz@@@zzz"
for row in depecheMood:
   items = row.split("\t")
   vector = map(lambda x: float(re.sub("[^0-9.-]", "", x)), items[1:])
   tag = items[0]
   tag_clear = tag[:tag.rfind("#")]
   if tag_clear_previous != tag_clear:
      if tag_clear_previous in dict_depecheMood:
         dict_depecheMood[tag_clear_previous] = map(lambda x: x / count, dict_depecheMood[tag_clear_previous]) 
      count = 0.0
   if tag_clear in dict_depecheMood:
      vector_current = dict_depecheMood[tag_clear]
      dict_depecheMood[tag_clear] = map(lambda x, y: x + y, vector, vector_current)
      count += 1.0
   else:
      dict_depecheMood[tag_clear] = vector
      count = 1.0
   tag_clear_previous = tag_clear


def func_depecheMood(sentence):
   words = sentence.strip().replace("  ", " ").split(" ")
   n = len(words)
   matches_found = 0.0
   res_vector = [0.1] * 8
   for preffix in ["", "#"]:
      for word in words:
         if preffix+word in dict_depecheMood:
             res_vector = map(lambda x, y: x + y, dict_depecheMood[preffix+word], res_vector)
             matches_found += 1.0

   if matches_found == 0.0:
      to_ret = res_vector
   else:
      to_ret = map(lambda x: x / matches_found, res_vector)
   #print to_ret
   return to_ret

###############################################################################################

class word_embeddings_class:

   def __init__(self, dimensions_number, name):
         self.default_vector = (np.array([0.0] * dimensions_number, dtype=np.float16)).astype(np.float16)
         self.word_emb_dict = {}
         self.name = name


   def load(self):
      if self.name == "biu":
         f = open(os.path.join(cwd, 'word_embeddings', 'biu', "lexsub_words"))
      elif self.name == "google":
         f = open(os.path.join(cwd, 'word_embeddings', 'google', "GoogleNews-vectors-negative300.txt"))
      elif self.name == "glove_twitter":
         f = open(os.path.join(cwd, 'word_embeddings', 'glove', "glove.twitter.27B", "glove.twitter.27B.200d.txt"))
      else:
         f = open(os.path.join(cwd, 'word_embeddings', 'glove', "glove.840B.300d.txt"))
      lines = f.readlines()
      if self.name == "biu" or self.name == "google":
         lines = lines[1:] 
      for row in lines:
         tmp = row.split(" ")
         if self.name == "biu":
            self.word_emb_dict[tmp[0]] = (np.array(tmp[1:-1], dtype = np.float16)).astype(np.float16)
         else:
            self.word_emb_dict[tmp[0]] = (np.array(tmp[1:], dtype = np.float16)).astype(np.float16)


   def are_lists_equal(self, a, b):
      if len(a) != len(b):
         return False
      for i in range(len(a)):
         if a[i] != b[i]:
            return False
      return True


   def get_word_emb(self, sentence):
      res_n_hashed = np.float16(0.0)
      res_n_nonhashed = np.float16(0.0)
      res_vector = self.default_vector
      maximum_likelyhood_estimate_vector = (self.default_vector).astype(np.float16)
      sentence_by_word = sentence.split(" ")
      hashed_flag = False
      for word in sentence_by_word:
         if self.is_hashed_word(word):
            hashed_flag = True
            res_vector = (np.add(res_vector, self.get_word_vector(word), dtype=np.float16)).astype(np.float16)
            if not self.are_lists_equal(res_vector, self.default_vector):
               res_n_hashed += 1.0
         if self.is_hashed_word(word):
            word = word[1:]
         maximum_likelyhood_estimate_vector = (np.add(maximum_likelyhood_estimate_vector, self.get_word_vector(self.clear_the_word(word)), dtype=np.float16)).astype(np.float16)
         if not self.are_lists_equal(maximum_likelyhood_estimate_vector, self.default_vector):
            res_n_nonhashed += 1.0
      tmp = (np.divide(maximum_likelyhood_estimate_vector, (res_n_nonhashed+1.0), dtype=np.float16)).astype(np.float16)
      if hashed_flag:
         return [res_n_hashed, (np.divide(res_vector, (res_n_hashed+1.0), dtype=np.float16)).astype(np.float16), tmp]
      else:
         return [res_n_hashed, tmp, tmp]


   def get_word_emb_min_max(self, sentence, min_or_max):
      res_n_hashed = 1.0
      res_n_nonhashed = 1.0
      res_vector = self.default_vector
      maximum_likelyhood_estimate_vector = self.default_vector
      sentence_by_word = sentence.split(" ")
      hashed_flag = False
      for word in sentence_by_word:
         if self.is_hashed_word(word):
            hashed_flag = True
            res_vector = self.min_or_max_func(res_vector, self.get_word_vector(word), min_or_max)
            if not self.are_lists_equal(res_vector, self.default_vector):
               res_n_hashed += 1.0
         elif not hashed_flag:
            maximum_likelyhood_estimate_vector = self.min_or_max_func(maximum_likelyhood_estimate_vector, self.get_word_vector(self.clear_the_word(word)), min_or_max)
            if not self.are_lists_equal(maximum_likelyhood_estimate_vector, self.default_vector):
               res_n_nonhashed += 1.0
      if hashed_flag:
         return res_vector
      else:
         return maximum_likelyhood_estimate_vector


   def min_or_max_func(self, res_vector, current_word_vector, min_or_max):
      n = len(res_vector)
      aggregated = 0
      for i in range(n):
         aggregated = current_word_vector[i] - res_vector[i]
      if min_or_max == 'min':
         if aggregated < 0:
            return current_word_vector
         else:
            return res_vector
      else:
         if aggregated > 0:
            return current_word_vector
         else:
            return res_vector
      return res_vector


   def clear_the_word(self, word):
      letters = list(word)
      return "".join(filter(lambda x: (x>="a" and x<="z") or x == "-", letters))

   
   def is_hashed_word(self, word):
      return word[0:1] == "#"
   

   def get_word_vector(self, word):
      word = self.clear_the_word(word)
      if word in self.word_emb_dict:
         return (np.array(self.word_emb_dict[word], dtype=np.float16)).astype(np.float16)
      return (self.default_vector).astype(np.float16)


print "loading word embedding vectors", time.strftime('%X %x %Z')
'''
biu_word_embeddings = word_embeddings_class(600, "biu")
biu_word_embeddings.load()
'''

print 2
glove_twitter_word_embeddings = word_embeddings_class(200, "glove_twitter")
glove_twitter_word_embeddings.load()

print 3
glove_common_crawl_word_embeddings = word_embeddings_class(300, "glove_common_crawl")
glove_common_crawl_word_embeddings.load()

print 4
google_word_embeddings = word_embeddings_class(300, "google")
google_word_embeddings.load()

print "finished word embedding vectors", time.strftime('%X %x %Z')

##########################################################################################

def cross_validation(number_of_validations, train_set, classes_dataset):
  acc_r = 0.0
  acc_s = 0.0
  train_size = len(train_set) / number_of_validations
  for i in range(number_of_validations):
    test_range = range(i*train_size, (i+1)*train_size)
    train_range = list(set(range(len(train_set))).difference(test_range))
    #print "1: ", test_range
    #print "2: ", train_range

    test_subset_data = list(np.array(train_set)[test_range])
    test_subset_classes = list(np.array(classes_dataset)[test_range])
    train_subset_data = list(np.array(train_set)[train_range])
    train_subset_classes = list(np.array(classes_dataset)[train_range])

    clf = sklearn.ensemble.GradientBoostingRegressor(n_estimators=100, max_depth=3)
    clf.fit(train_subset_data, train_subset_classes)
    p = clf.predict(test_subset_data)

    r_row, p_value = scipy.stats.pearsonr(p, test_subset_classes)
    s_row, p_value = scipy.stats.spearmanr(p, test_subset_classes)
    print len(test_subset_data), len(train_subset_data)
    print r_row, s_row
    acc_r += r_row
    acc_s += s_row
  return acc_r / float(number_of_validations), acc_s / float(number_of_validations)



##########################################################################################

def prediction(train_x, train_y, test_x):
    clf = sklearn.ensemble.GradientBoostingRegressor(n_estimators=100, max_depth=3)
    #clf = sklearn.ensemble.RandomForestRegressor()#n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)
    #print "===???", len(train_x), len(train_y)
    #print " "
    #print train_x
    #print " "
    #print test_x
    #print train_y
    #print "   train_x =  "
    #print train_x
    #print "   train_y =  "
    #print train_y
    #print " "
    clf.fit(train_x, train_y)
    #print "here1a"
    #print "       test_X =   ", test_x
    p = clf.predict(test_x)
    #print "here2a"
    return p



##########################################################################################




result_test_set = []
print '-' * 50
dictTextProcessingCom = { "neg": 0, "neutral": 1, "pos": 2 }
for YYY in range(1):
 index = 0
 print '===== ===='
 print 'YYY =', YYY
 print " "
 for affect in affect_list:
  print affect

  N = len(train_data[affect])

  #train_x = [[[] for _ in range(N)] for _ in range(15)]
  #train_y = [[[] for _ in range(N)] for _ in range(15)]

  train_x = [[] for _ in range(15)]
  train_y = [[] for _ in range(15)]
  all_train_x = [[] for _ in range(15)]
  all_train_y = [[] for _ in range(15)]
  train_idd = []

  #test_x = [[[] for _ in range(N)] for _ in range(15)]
  #test_y = [[[] for _ in range(N)] for _ in range(15)]

  test_x = [[] for _ in range(15)]
  test_y = [[] for _ in range(15)]
  test_idd = []

  index = 0

  for train_part in [train_data]:
    is_train_data = train_part == train_data
    print "bool: ", is_train_data
    for idd in train_part[affect]:
      train_tweet_json = train_part[affect][idd]
      sentence = train_tweet_json["atext_clear_lower"]
      #vec = func_rnc_Emoticon_AFFLEX_NEGLEX_unigrams(sentence)
      #vec = [vec[0], vec[2], vec[4]]
      #vec = [vec[2]] 
      #tmp = vec
      #print tmp
      if (not is_train_data) or (is_train_data and train_tweet_json["remained"] != 0):
        #try:
        #cnt += 1
        #if cnt % 500 == 0: print cnt
        #tmp0 = biu_word_embeddings.get_word_emb(train_data[i]['list'][x]["atext_clear_lower"])

        train_idd.append(idd)

        tmp1 = glove_twitter_word_embeddings.get_word_emb(sentence)
        tmp2 = glove_common_crawl_word_embeddings.get_word_emb(sentence)
        tmp3 = google_word_embeddings.get_word_emb(sentence)
        
        tmp = (
            np.ma.concatenate([
                func_depecheMood(sentence)
            ])
        )

        train_x[0].append(tmp)
        train_y[0].append(float(train_tweet_json['magnitude']))


        tmp = (
            np.ma.concatenate([
                analyzer.polarity_scores(sentence)
            ])
        )

        train_x[1].append(tmp)
        train_y[1].append(float(train_tweet_json['magnitude']))


        tmp = (
            np.ma.concatenate([
                # vec,
                np.array([train_tweet_json["metrics"]["textProcessingCom"]["probability"]["neg"]], dtype=np.float16),
                np.array([train_tweet_json["metrics"]["textProcessingCom"]["probability"]["neutral"]], dtype=np.float16),
                np.array([train_tweet_json["metrics"]["textProcessingCom"]["probability"]["pos"]], dtype=np.float16)
            ])
        )

        train_x[2].append(tmp)
        train_y[2].append(float(train_tweet_json['magnitude']))


        tmp = (
            np.ma.concatenate([
                # vec,
                np.array([tmp1[0]], dtype=np.float16),
                # + list(tmp0[1].astype(np.float16))
                tmp1[1],
                tmp1[2],
            ])
        )

        train_x[3].append(tmp)
        train_y[3].append(float(train_tweet_json['magnitude']))

        tmp = (
            np.ma.concatenate([
                # vec,
                np.array([tmp1[0]], dtype=np.float16),
                # + list(tmp0[1].astype(np.float16))
                tmp2[1],
                tmp2[2],
            ])
        )

        train_x[4].append(tmp)
        train_y[4].append(float(train_tweet_json['magnitude']))

        tmp = (
            np.ma.concatenate([
                # vec,
                np.array([tmp1[0]], dtype=np.float16),
                tmp3[1],
                tmp3[2]
            ])
        )

        train_x[5].append(tmp)
        train_y[5].append(float(train_tweet_json['magnitude']))

        tmp = (
            np.ma.concatenate([
                np.array([tmp1[0]], dtype=np.float16),
                # + list(tmp0[1].astype(np.float16))
                tmp1[1],
            ])
        )

        train_x[6].append(tmp)
        train_y[6].append(float(train_tweet_json['magnitude']))

        tmp = (
            np.ma.concatenate([
                np.array([tmp1[0]], dtype=np.float16),
                tmp2[1],
            ])
        )

        train_x[7].append(tmp)
        train_y[7].append(float(train_tweet_json['magnitude']))

        tmp = (
            np.ma.concatenate([
                np.array([tmp1[0]], dtype=np.float16),
                # + list(tmp0[1].astype(np.float16))
                tmp3[1]
            ])
        )

        train_x[8].append(tmp)
        train_y[8].append(float(train_tweet_json['magnitude']))

        tmp = (
            np.ma.concatenate([
                np.array([tmp1[0]], dtype=np.float16),
                # + list(tmp0[1].astype(np.float16))
                tmp1[1],
                tmp1[2],
                tmp2[1],
                tmp2[2],
                tmp3[1],
                tmp3[2]
            ])
        )

        train_x[9].append(tmp)
        train_y[9].append(float(train_tweet_json['magnitude']))

        tmp = (
            np.ma.concatenate([
                # vec,
                func_depecheMood(sentence),
                analyzer.polarity_scores(sentence),
                np.array([train_tweet_json["metrics"]["textProcessingCom"]["probability"]["neg"]], dtype=np.float16),
                np.array([train_tweet_json["metrics"]["textProcessingCom"]["probability"]["neutral"]], dtype=np.float16),
                np.array([train_tweet_json["metrics"]["textProcessingCom"]["probability"]["pos"]], dtype=np.float16)
            ])
        )

        train_x[10].append(tmp)
        train_y[10].append(float(train_tweet_json['magnitude']))

        tmp = (
            np.ma.concatenate([
                # vec,
                func_depecheMood(sentence),
                analyzer.polarity_scores(sentence),
                np.array([train_tweet_json["metrics"]["textProcessingCom"]["probability"]["neg"]], dtype=np.float16),
                np.array([train_tweet_json["metrics"]["textProcessingCom"]["probability"]["neutral"]], dtype=np.float16),
                np.array([train_tweet_json["metrics"]["textProcessingCom"]["probability"]["pos"]], dtype=np.float16),
                np.array([tmp1[0]], dtype=np.float16),
                # + list(tmp0[1].astype(np.float16))
                tmp1[1],
                tmp1[2],
                tmp2[1],
                tmp2[2],
                tmp3[1],
                tmp3[2]
            ])
        )

        train_x[11].append(tmp)
        train_y[11].append(float(train_tweet_json['magnitude']))


        tmp = (
            np.ma.concatenate([
                # vec,
                func_depecheMood(sentence),
                analyzer.polarity_scores(sentence)
            ])
        )

        train_x[12].append(tmp)
        train_y[12].append(float(train_tweet_json['magnitude']))

        tmp = (
            np.ma.concatenate([
                # vec,
                func_depecheMood(sentence),
                analyzer.polarity_scores(sentence),
                np.array([tmp1[0]], dtype=np.float16),
                # + list(tmp0[1].astype(np.float16))
                tmp1[1],
                tmp1[2],
                tmp2[1],
                tmp2[2],
                tmp3[1],
                tmp3[2]
            ])
        )

        train_x[13].append(tmp)
        train_y[13].append(float(train_tweet_json['magnitude']))


        tmp = (
            np.ma.concatenate([
                # vec,
                np.array([train_tweet_json["metrics"]["textProcessingCom"]["probability"]["neg"]], dtype=np.float16),
                np.array([train_tweet_json["metrics"]["textProcessingCom"]["probability"]["neutral"]], dtype=np.float16),
                np.array([train_tweet_json["metrics"]["textProcessingCom"]["probability"]["pos"]], dtype=np.float16),
                np.array([tmp1[0]], dtype=np.float16),
                # + list(tmp0[1].astype(np.float16))
                tmp1[1],
                tmp1[2],
                tmp2[1],
                tmp2[2],
                tmp3[1],
                tmp3[2]
            ])
        )

        train_x[14].append(tmp)
        train_y[14].append(float(train_tweet_json['magnitude']))

        index += 1
        '''
        print affect
        for i in range(15):
            p = prediction(train_x, train_y, test_x[i])

            print "i=", i
            r_row, p_value = scipy.stats.pearsonr(p, test_y[i])
            print r_row, p_value
        '''
    '''
    print len(train_x[0])
    print len(train_x[0][:5] + train_x[0][7:])
    print " "
    print train_x[0][5:8]
    '''
    N = len(train_x[0])

    for i in range(15):
        all_train_x[i] = [train_x[i][h] for h in range(0, N)]
        all_train_y[i] = [train_y[i][h] for h in range(0, N)]

    index_test = 0
    for idd in test_data[affect]:
        tweet_json = test_data[affect][idd]
        sentence = tweet_json["atext_clear_lower"]
        test_idd.append(idd)
        #vec = func_rnc_Emoticon_AFFLEX_NEGLEX_unigrams(sentence)
        #vec = [vec[0], vec[2], vec[4]]
        #vec = [vec[2]] 
        #tmp = vec
        #print tmp
        
        tmp1 = glove_twitter_word_embeddings.get_word_emb(sentence)
        tmp2 = glove_common_crawl_word_embeddings.get_word_emb(sentence)
        tmp3 = google_word_embeddings.get_word_emb(sentence)
        
        
        tmp = (
            np.ma.concatenate([
                func_depecheMood(sentence)
            ])
        )
          
        test_x[0].append(tmp)
        test_y[0].append(float(tweet_json['magnitude']))

        tmp = (
            np.ma.concatenate([
                analyzer.polarity_scores(sentence)
            ])
        )

        test_x[1].append(tmp)
        test_y[1].append(float(tweet_json['magnitude']))

        tmp = (
          np.ma.concatenate([
             #vec,
             np.array([tweet_json["metrics"]["textProcessingCom"]["probability"]["neg"]], dtype=np.float16),
             np.array([tweet_json["metrics"]["textProcessingCom"]["probability"]["neutral"]], dtype=np.float16),
             np.array([tweet_json["metrics"]["textProcessingCom"]["probability"]["pos"]], dtype=np.float16)
           ])
        )

        test_x[2].append(tmp)
        test_y[2].append(float(tweet_json['magnitude']))

        tmp = (
          np.ma.concatenate([
             #vec,
             np.array([tmp1[0]], dtype=np.float16),
          #+ list(tmp0[1].astype(np.float16))
             tmp1[1],
             tmp1[2],
           ])
        )

        test_x[3].append(tmp)
        test_y[3].append(float(tweet_json['magnitude']))

        tmp = (
          np.ma.concatenate([
             #vec,
             np.array([tmp1[0]], dtype=np.float16),
          #+ list(tmp0[1].astype(np.float16))
             tmp2[1],
             tmp2[2],
           ])
        )

        test_x[4].append(tmp)
        test_y[4].append(float(tweet_json['magnitude']))

        tmp = (
          np.ma.concatenate([
             #vec,
             np.array([tmp1[0]], dtype=np.float16),
             tmp3[1],
             tmp3[2]
           ])
        )

        test_x[5].append(tmp)
        test_y[5].append(float(tweet_json['magnitude']))

        tmp = (
          np.ma.concatenate([
             np.array([tmp1[0]], dtype=np.float16),
          #+ list(tmp0[1].astype(np.float16))
             tmp1[1],
           ])
        )

        test_x[6].append(tmp)
        test_y[6].append(float(tweet_json['magnitude']))

        tmp = (
          np.ma.concatenate([
             np.array([tmp1[0]], dtype=np.float16),
             tmp2[1],
           ])
        )

        test_x[7].append(tmp)
        test_y[7].append(float(tweet_json['magnitude']))

        tmp = (
          np.ma.concatenate([
             np.array([tmp1[0]], dtype=np.float16),
          #+ list(tmp0[1].astype(np.float16))
             tmp3[1]
           ])
        )

        test_x[8].append(tmp)
        test_y[8].append(float(tweet_json['magnitude']))

        tmp = (
          np.ma.concatenate([
             np.array([tmp1[0]], dtype=np.float16),
          #+ list(tmp0[1].astype(np.float16))
             tmp1[1],
             tmp1[2],
             tmp2[1],
             tmp2[2],
             tmp3[1],
             tmp3[2]
           ])
        )

        test_x[9].append(tmp)
        test_y[9].append(float(tweet_json['magnitude']))

        tmp = (
          np.ma.concatenate([
             #vec,
             func_depecheMood(sentence),
             analyzer.polarity_scores(sentence),
             np.array([tweet_json["metrics"]["textProcessingCom"]["probability"]["neg"]], dtype=np.float16),
             np.array([tweet_json["metrics"]["textProcessingCom"]["probability"]["neutral"]], dtype=np.float16),
             np.array([tweet_json["metrics"]["textProcessingCom"]["probability"]["pos"]], dtype=np.float16)
           ])
        )

        test_x[10].append(tmp)
        test_y[10].append(float(tweet_json['magnitude']))

        tmp = (
          np.ma.concatenate([
             #vec,
             func_depecheMood(sentence),
             analyzer.polarity_scores(sentence),
             np.array([tweet_json["metrics"]["textProcessingCom"]["probability"]["neg"]], dtype=np.float16),
             np.array([tweet_json["metrics"]["textProcessingCom"]["probability"]["neutral"]], dtype=np.float16),
             np.array([tweet_json["metrics"]["textProcessingCom"]["probability"]["pos"]], dtype=np.float16),
             np.array([tmp1[0]], dtype=np.float16),
          #+ list(tmp0[1].astype(np.float16))
             tmp1[1],
             tmp1[2],
             tmp2[1],
             tmp2[2],
             tmp3[1],
             tmp3[2]
           ])
        )

        test_x[11].append(tmp)
        test_y[11].append(float(tweet_json['magnitude']))


        tmp = (
            np.ma.concatenate([
                # vec,
                func_depecheMood(sentence),
                analyzer.polarity_scores(sentence)
            ])
        )

        test_x[12].append(tmp)
        test_y[12].append(float(tweet_json['magnitude']))

        tmp = (
            np.ma.concatenate([
                # vec,
                func_depecheMood(sentence),
                analyzer.polarity_scores(sentence),
                np.array([tmp1[0]], dtype=np.float16),
                # + list(tmp0[1].astype(np.float16))
                tmp1[1],
                tmp1[2],
                tmp2[1],
                tmp2[2],
                tmp3[1],
                tmp3[2]
            ])
        )

        test_x[13].append(tmp)
        test_y[13].append(float(tweet_json['magnitude']))


        tmp = (
          np.ma.concatenate([
             #vec,
             np.array([tweet_json["metrics"]["textProcessingCom"]["probability"]["neg"]], dtype=np.float16),
             np.array([tweet_json["metrics"]["textProcessingCom"]["probability"]["neutral"]], dtype=np.float16),
             np.array([tweet_json["metrics"]["textProcessingCom"]["probability"]["pos"]], dtype=np.float16),
             np.array([tmp1[0]], dtype=np.float16),
          #+ list(tmp0[1].astype(np.float16))
             tmp1[1],
             tmp1[2],
             tmp2[1],
             tmp2[2],
             tmp3[1],
             tmp3[2]
           ])
        )

        test_x[14].append(tmp)
        test_y[14].append(float(tweet_json['magnitude']))

        index_test += 1

    hashset = range(index_test)
    threads = [threading.Thread(target=runnable_class(z).run) for z in range(8)]
    for i in range(8):
        threads[i].start()

    for thread in threads:
        thread.join()

    a = json.dumps(test_data, sort_keys=True, indent=2, separators=(',', ': '), ensure_ascii=False)
    f = open(os.path.join(cwd, 'dataset', 'task1', 'test_ainl', 'dataset_json_'+affect+'.txt'), 'w+')
    f.write(a)
    f.close()











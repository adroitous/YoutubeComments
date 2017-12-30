import json
from collections import namedtuple
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import sentiwordnet as swn
from nltk import word_tokenize
from nltk import NaiveBayesClassifier
from itertools import combinations
import nltk

with open('video_dict.json') as json_data:
    video_dict = json.load(json_data)

def not_int(s):
    try:
        int(s)
        return False
    except ValueError:
        return True

def sent_classifier(video_dict, video_ids):
    sid = SentimentIntensityAnalyzer()
    all_comments = []
    for vid in video_ids:
        all_comments += video_dict[vid]
    sent_dict = [(sid.polarity_scores(comment[0])['compound'], comment[1]) for comment in all_comments]
    # print(sent_dict)
    ## combo is a tuple containing the combined pair
    ## combo[n] indicates the n-th element in that combo
    ## combo[n][0] gives the polarity score while combo[n][1] gives the # of likes
    pair_combo = [({'diff' : combo[0][0] - combo[1][0]}, (1 if combo[0][1] > combo[1][1] else 0)) for combo in list(combinations(sent_dict, 2)) if combo[0][1] != combo[1][1]]
    classifier = NaiveBayesClassifier.train(pair_combo)
    # classifier.show_most_informative_features()
    print(nltk.classify.accuracy(classifier, pair_combo))
    return classifier

def sent_test(classifier, sent1, sent2):
    sid = SentimentIntensityAnalyzer()
    ss1 = sid.polarity_scores(sent1)['compound']
    ss2 = sid.polarity_scores(sent2)['compound']
    if ss1 > ss2: return 0
    else: return 1
#    return classifier.classify({'diff':ss1-ss2})
